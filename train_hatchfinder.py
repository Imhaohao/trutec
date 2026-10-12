"""Smallest custom masked fine-tuning loop for HatchFinder."""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import inference_hatchfinder as adapter

PAD_MULTIPLE = 64
DEFAULT_SAVE_EVERY = 200
DEFAULT_STEPS = 600
DEFAULT_TILE_SIZE = 512
DEFAULT_LR = 1e-5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def get_validation_document_ids(data_root: Path) -> set[str]:
    candidates = [Path("artifacts/splits/development.json"),
                  Path("artifacts/splits/development-real.json"),
                  data_root / "val.json", Path("dataset/val.json")]
    forbidden_ids: set[str] = set()
    for candidate in candidates:
        if candidate.is_file():
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            examples = payload.get("examples", payload) if isinstance(payload, dict) else payload
            if isinstance(examples, list):
                forbidden_ids.update(str(r["document_id"]) for r in examples if isinstance(r, dict) and "document_id" in r)
    return forbidden_ids


def check_split_safety(manifest_rows: list[dict[str, Any]], data_root: Path) -> set[str]:
    forbidden_ids = get_validation_document_ids(data_root)
    training_ids: set[str] = {str(r["document_id"]) for r in manifest_rows if isinstance(r, dict) and "document_id" in r}
    overlap = training_ids & forbidden_ids
    if overlap:
        raise ValueError(f"Training documents overlap with forbidden evaluation documents: {sorted(overlap)}")
    return training_ids


class SingleImageCache:
    """Cache strictly one drawing and its reviewed masks to stay under memory limits."""

    def __init__(self, data_root: Path) -> None:
        self.data_root = data_root
        self._image_rel: str | None = None
        self._rgb: np.ndarray | None = None
        self._row_id: str | None = None
        self._positive: np.ndarray | None = None
        self._known: np.ndarray | None = None

    def get(self, row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        from evaluate import reviewed_domains

        image_rel = str(row["image"])
        if self._image_rel != image_rel or self._rgb is None:
            self._rgb = adapter.read_rgb_drawing(self.data_root, row)
            self._image_rel = image_rel
            self._row_id, self._positive, self._known = None, None, None

        row_id = str(row["id"])
        if self._row_id != row_id or self._positive is None or self._known is None:
            positive, known, _, _ = reviewed_domains(row, self.data_root)
            self._row_id, self._positive, self._known = row_id, positive, known

        return self._rgb, self._positive, self._known


def group_queries_by_document(rows: list[dict[str, Any]]) -> tuple[list[str], dict[str, list[dict[str, Any]]]]:
    docs_map: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        docs_map.setdefault(str(row["document_id"]), []).append(row)
    doc_ids = sorted(docs_map.keys())
    for d in doc_ids:
        docs_map[d].sort(key=lambda r: str(r["id"]))
    return doc_ids, docs_map


def choose_crop_center(
    positive_mask: np.ndarray, negative_mask: np.ndarray, class_choice: int, rng: np.random.Generator
) -> tuple[int, int]:
    pos_coords = np.argwhere(positive_mask)
    neg_coords = np.argwhere(negative_mask)
    candidates = (pos_coords if len(pos_coords) > 0 else neg_coords) if class_choice == 0 else (
        neg_coords if len(neg_coords) > 0 else pos_coords
    )
    if len(candidates) == 0:
        raise ValueError("No reviewed known pixels available for sampling")
    chosen = candidates[int(rng.integers(0, len(candidates)))]
    return int(chosen[0]), int(chosen[1])


def calculate_tile_bounds(py: int, px: int, height: int, width: int, tile_size: int) -> tuple[int, int, int, int]:
    y0 = max(0, min(py - tile_size // 2, height - tile_size)) if height >= tile_size else 0
    valid_h = tile_size if height >= tile_size else height
    x0 = max(0, min(px - tile_size // 2, width - tile_size)) if width >= tile_size else 0
    valid_w = tile_size if width >= tile_size else width
    return y0, x0, valid_h, valid_w


def crop_and_pad_tile(
    rgb: np.ndarray, known_mask: np.ndarray, positive_mask: np.ndarray,
    y0: int, x0: int, valid_h: int, valid_w: int, tile_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sub_rgb = rgb[y0:y0 + valid_h, x0:x0 + valid_w]
    sub_known = known_mask[y0:y0 + valid_h, x0:x0 + valid_w]
    sub_pos = positive_mask[y0:y0 + valid_h, x0:x0 + valid_w]

    if valid_h == tile_size and valid_w == tile_size:
        return sub_rgb, sub_known.astype(np.float32), sub_pos.astype(np.float32)

    tile_rgb = np.full((tile_size, tile_size, 3), adapter.WHITE, dtype=np.uint8)
    tile_rgb[:valid_h, :valid_w] = sub_rgb
    tile_known = np.zeros((tile_size, tile_size), dtype=np.float32)
    tile_known[:valid_h, :valid_w] = sub_known
    tile_pos = np.zeros((tile_size, tile_size), dtype=np.float32)
    tile_pos[:valid_h, :valid_w] = sub_pos
    return tile_rgb, tile_known, tile_pos


def sample_training_step(
    rng: np.random.Generator,
    doc_ids: list[str],
    docs_map: dict[str, list[dict[str, Any]]],
    cache: SingleImageCache,
    tile_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    doc_id = doc_ids[int(rng.integers(0, len(doc_ids)))]
    queries = docs_map[doc_id]
    row = queries[int(rng.integers(0, len(queries)))]

    rgb, positive_mask, known_mask = cache.get(row)
    py, px = choose_crop_center(positive_mask, known_mask & ~positive_mask, int(rng.integers(0, 2)), rng)

    y0, x0, valid_h, valid_w = calculate_tile_bounds(py, px, int(rgb.shape[0]), int(rgb.shape[1]), tile_size)
    tile_rgb, tile_known, tile_pos = crop_and_pad_tile(
        rgb, known_mask, positive_mask, y0, x0, valid_h, valid_w, tile_size
    )

    box = (int(row["query_box"][0]), int(row["query_box"][1]), int(row["query_box"][2]), int(row["query_box"][3]))
    return tile_rgb, tile_known, tile_pos, adapter.extract_query(rgb, box), row


def choose_coverage_crop_center(
    positive_mask: np.ndarray, negative_mask: np.ndarray, branch: str, rng: np.random.Generator
) -> tuple[int, int]:
    if branch == "positive":
        candidates = np.argwhere(positive_mask)
        if len(candidates) == 0:
            raise ValueError("No reviewed positive pixels available for positive branch")
    elif branch == "negative":
        candidates = np.argwhere(negative_mask)
        if len(candidates) == 0:
            raise ValueError("No reviewed negative pixels available for negative branch")
    else:
        raise ValueError(f"Unknown branch {branch!r}, expected 'positive' or 'negative'")
    chosen = candidates[int(rng.integers(0, len(candidates)))]
    return int(chosen[0]), int(chosen[1])


def inspect_row_branch_eligibility(row: dict[str, Any], data_root: Path) -> tuple[bool, bool]:
    from evaluate import reviewed_domains

    positive, known, _, _ = reviewed_domains(row, data_root)
    has_pos = bool(positive.any())
    has_neg = bool((known & ~positive).any())
    return has_pos, has_neg


@dataclass
class CoverageBranchState:
    """One branch's remaining per-document query queues for the current epoch."""

    document_order: deque[str] = field(default_factory=deque)
    query_queues: dict[str, deque[dict[str, Any]]] = field(default_factory=dict)


class CoverageSampler:
    """Bounded deterministic coverage sampler covering eligible documents and queries."""

    def __init__(
        self,
        rows: list[dict[str, Any]],
        data_root: Path,
        cache: SingleImageCache,
        tile_size: int,
        seed: int = 42,
    ) -> None:
        self.rows = rows
        self.data_root = data_root
        self.cache = cache
        self.tile_size = tile_size
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self._eligibility_cache: dict[str, tuple[bool, bool]] = {}
        self.doc_ids, self.docs_map = group_queries_by_document(rows)

        self._branch_states = {
            "positive": CoverageBranchState(),
            "negative": CoverageBranchState(),
        }
        self._step_count = 0
        self.last_branch: str | None = None
        self.branch_counts: dict[str, int] = {"positive": 0, "negative": 0}
        self.selected_query_ids: list[str] = []
        self.selected_document_ids: list[str] = []

    def _start_epoch(self, branch: str) -> None:
        state = self._branch_states[branch]
        query_queues: dict[str, deque[dict[str, Any]]] = {}
        for doc_id in self.doc_ids:
            eligible_queries = [
                row
                for row in self.docs_map[doc_id]
                if self._is_row_eligible(row, branch)
            ]
            if eligible_queries:
                self.rng.shuffle(eligible_queries)
                query_queues[doc_id] = deque(eligible_queries)

        if not query_queues:
            raise ValueError(f"No eligible documents or rows exist for {branch} branch")

        document_order = list(query_queues)
        self.rng.shuffle(document_order)
        state.query_queues = query_queues
        state.document_order = deque(document_order)

    def _is_row_eligible(self, row: dict[str, Any], branch: str) -> bool:
        row_id = str(row["id"])
        if row_id not in self._eligibility_cache:
            self._eligibility_cache[row_id] = inspect_row_branch_eligibility(row, self.data_root)
        has_pos, has_neg = self._eligibility_cache[row_id]
        return has_pos if branch == "positive" else has_neg

    def select_query(self, branch: str) -> dict[str, Any]:
        if branch not in ("positive", "negative"):
            raise ValueError(f"Unknown branch {branch!r}, expected 'positive' or 'negative'")
        state = self._branch_states[branch]
        if not state.document_order:
            self._start_epoch(branch)

        doc_id = state.document_order.popleft()
        query_queue = state.query_queues[doc_id]
        row = query_queue.popleft()
        if query_queue:
            state.document_order.append(doc_id)
        else:
            del state.query_queues[doc_id]
        return row

    def sample_step(
        self, branch: str | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        if branch is None:
            branch = "positive" if (self._step_count % 2 == 0) else "negative"
        if branch not in self._branch_states:
            raise ValueError(f"Unknown branch {branch!r}, expected 'positive' or 'negative'")

        row = self.select_query(branch)
        self._step_count += 1
        self.last_branch = branch
        self.branch_counts[branch] += 1
        self.selected_query_ids.append(str(row["id"]))
        self.selected_document_ids.append(str(row["document_id"]))

        rgb, positive_mask, known_mask = self.cache.get(row)
        negative_mask = known_mask & ~positive_mask
        py, px = choose_coverage_crop_center(positive_mask, negative_mask, branch, self.rng)

        y0, x0, valid_h, valid_w = calculate_tile_bounds(
            py, px, int(rgb.shape[0]), int(rgb.shape[1]), self.tile_size
        )
        tile_rgb, tile_known, tile_pos = crop_and_pad_tile(
            rgb, known_mask, positive_mask, y0, x0, valid_h, valid_w, self.tile_size
        )
        box = (
            int(row["query_box"][0]),
            int(row["query_box"][1]),
            int(row["query_box"][2]),
            int(row["query_box"][3]),
        )
        return tile_rgb, tile_known, tile_pos, adapter.extract_query(rgb, box), row


def masked_loss(logits: Any, target: Any, loss_mask: Any, smooth: float = 1e-6) -> tuple[Any, Any, Any | None]:
    import torch
    import torch.nn.functional as F

    if logits.shape != target.shape or logits.shape != loss_mask.shape:
        raise ValueError(f"Shape mismatch: logits {tuple(logits.shape)}, target {tuple(target.shape)}, mask {tuple(loss_mask.shape)}")

    loss_map = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce_loss = (loss_map * loss_mask).sum() / loss_mask.sum().clamp_min(1.0)

    probs = torch.sigmoid(logits) * loss_mask
    masked_target = target * loss_mask
    non_empty = masked_target.sum(dim=(1, 2, 3)) > 0
    if not non_empty.any():
        return bce_loss, bce_loss, None

    p, t = probs[non_empty], masked_target[non_empty]
    intersection = (p * t).sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + smooth) / (p.sum(dim=(1, 2, 3)) + t.sum(dim=(1, 2, 3)) + smooth)
    dice_loss = 1.0 - dice.mean()
    return bce_loss + dice_loss, bce_loss, dice_loss


def configure_trainable_scope(model: Any, scope: str) -> list[Any]:
    if scope not in ("head", "all"):
        raise ValueError(f"trainable_scope must be 'head' or 'all', got {scope!r}")

    for param in model.parameters():
        param.requires_grad = True
    if scope == "head":
        for encoder_name in ("drawing_encoder", "hatch_encoder"):
            if hasattr(model, encoder_name):
                for param in getattr(model, encoder_name).parameters():
                    param.requires_grad = False

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise ValueError(f"No trainable parameters found for scope {scope!r}")
    return trainable_params


def freeze_encoders(model: Any) -> list[Any]:
    """Keep the original head-only helper available to existing callers."""
    return configure_trainable_scope(model, "head")


def seed_torch(seed: int) -> Any:
    import torch

    torch.manual_seed(seed)
    return torch


def frozen_module_names(model: Any) -> list[str]:
    frozen = []
    for name, module in model.named_children():
        params = list(module.parameters())
        if params and all(not param.requires_grad for param in params):
            frozen.append(name)
    return frozen


def sha256_model_state(model: Any) -> str:
    import torch

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.view(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def extract_model_config(model: Any) -> dict[str, Any]:
    return model.config.model.model_dump(mode="json")


def save_atomic_checkpoint(model: Any, output_path: Path) -> None:
    import torch

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "model_config": extract_model_config(model),
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    }
    tmp_path = output_path.parent / f"{output_path.name}.tmp.{os.getpid()}"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, output_path)


def build_provenance(
    pretrained_sha256: str, manifest_sha256: str, fit_doc_ids: Sequence[str], seed: int, steps: int,
    lr: float, tile_size: int, trainable_params_count: int, elapsed_seconds: float, loss_history: list[float],
    sampling: str = "random",
    trainable_scope: str = "head",
    torch_seed: int | None = None,
    starting_model_sha256: str | None = None,
    frozen_modules: Sequence[str] | None = None,
    selected_query_ids: Sequence[str] | None = None,
    selected_document_ids: Sequence[str] | None = None,
    branch_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    counts = dict(branch_counts) if branch_counts is not None else None
    return {
        "source_pretrained_sha256": pretrained_sha256,
        "starting_model_sha256": starting_model_sha256,
        "manifest_sha256": manifest_sha256,
        "fit_document_ids": sorted(fit_doc_ids),
        "seed": seed,
        "torch_seed": torch_seed,
        "steps": steps,
        "lr": lr,
        "tile_size": tile_size,
        "trainable_parameters": trainable_params_count,
        "trainable_scope": trainable_scope,
        "frozen_modules": list(frozen_modules) if frozen_modules is not None else None,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "sampling_mode": sampling,
        "selected_document_ids": [str(value) for value in (selected_document_ids or [])],
        "selected_query_ids": [str(value) for value in (selected_query_ids or [])],
        "branch_counts": counts,
        "loss_summary": {
            "initial_loss": round(loss_history[0], 6) if loss_history else None,
            "final_loss": round(loss_history[-1], 6) if loss_history else None,
            "mean_loss": round(float(np.mean(loss_history)), 6) if loss_history else None,
            "min_loss": round(float(np.min(loss_history)), 6) if loss_history else None,
            "max_loss": round(float(np.max(loss_history)), 6) if loss_history else None,
        },
    }


def execute_training_step(
    model: Any, optimizer: Any, trainable_params: list[Any], device: str,
    tile_rgb: np.ndarray, tile_known: np.ndarray, tile_pos: np.ndarray,
    hatch_rgb: np.ndarray, tile_size: int, step: int,
) -> tuple[float, float, float | None]:
    import torch

    drawing_tensor = adapter.rgb_to_tensor(tile_rgb, device)
    input_mask = torch.ones((1, 1, tile_size, tile_size), dtype=torch.float32, device=device)
    hatch_tensor = adapter.rgb_to_tensor(hatch_rgb, device)
    target_tensor = torch.from_numpy(tile_pos).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)
    loss_mask_tensor = torch.from_numpy(tile_known).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)

    optimizer.zero_grad()
    logits = model(drawing_tensor, input_mask, hatch_tensor)
    loss, bce_loss, dice_loss = masked_loss(logits, target_tensor, loss_mask_tensor)

    if not bool(torch.isfinite(loss).item()):
        raise ValueError(f"Non-finite loss {loss.item()} at step {step}")
    loss.backward()

    for p in trainable_params:
        if p.grad is not None and not bool(torch.isfinite(p.grad).all()):
            raise ValueError(f"Non-finite gradient detected at step {step}")
    optimizer.step()
    return float(loss.item()), float(bce_loss.item()), (float(dice_loss.item()) if dice_loss is not None else None)


def select_device(requested: str | None) -> str:
    import torch

    return adapter.select_device(requested, bool(torch.backends.mps.is_available()))


def load_pretrained_model(pretrained_path: Path, device: str) -> Any:
    from hatchfinder_mps import MPSHatchFinder

    model = MPSHatchFinder(load_model_path=pretrained_path, device="cpu")
    return model.to("mps") if device == "mps" else model


def validate_train_args(args: argparse.Namespace) -> None:
    if args.tile_size <= 0 or args.tile_size % PAD_MULTIPLE != 0:
        raise ValueError(f"Tile size must be a positive multiple of 64, got {args.tile_size}")
    if args.steps <= 0 or not np.isfinite(args.lr) or args.lr <= 0 or args.save_every <= 0:
        raise ValueError("steps, lr, and save_every must all be positive")
    if getattr(args, "sampling", "random") not in ("random", "coverage"):
        raise ValueError(f"sampling must be 'random' or 'coverage', got {args.sampling}")
    if getattr(args, "trainable_scope", "head") not in ("head", "all"):
        raise ValueError(f"trainable_scope must be 'head' or 'all', got {args.trainable_scope}")


def train(
    manifest_path: Path, data_root: Path, pretrained_path: Path, output_path: Path, device: str,
    steps: int = DEFAULT_STEPS, tile_size: int = DEFAULT_TILE_SIZE, lr: float = DEFAULT_LR,
    seed: int = 42, save_every: int = DEFAULT_SAVE_EVERY, model: Any | None = None,
    sampling: str = "random",
    trainable_scope: str = "head",
) -> dict[str, Any]:
    if sampling not in ("random", "coverage"):
        raise ValueError(f"Unknown sampling mode {sampling!r}, expected 'random' or 'coverage'")
    if trainable_scope not in ("head", "all"):
        raise ValueError(f"trainable_scope must be 'head' or 'all', got {trainable_scope!r}")

    torch = seed_torch(seed)
    from evaluate import load_manifest

    start_time = time.monotonic()
    manifest_sha256 = sha256_file(manifest_path)
    pretrained_sha256 = sha256_file(pretrained_path) if pretrained_path.exists() else "unhashed"

    rows = load_manifest(manifest_path, require_labels=True)
    fit_doc_ids = check_split_safety(rows, data_root)
    doc_ids, docs_map = group_queries_by_document(rows)

    if model is None:
        model = load_pretrained_model(pretrained_path, device)
    starting_model_sha256 = sha256_model_state(model)
    model.train()

    trainable_params = configure_trainable_scope(model, trainable_scope)
    frozen_modules = frozen_module_names(model)
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    cache = SingleImageCache(data_root)
    loss_history: list[float] = []

    sampler = (
        CoverageSampler(rows, data_root, cache, tile_size, seed)
        if sampling == "coverage"
        else None
    )
    selected_queries: list[str] = []
    selected_docs: list[str] = []

    for step in range(1, steps + 1):
        if sampler is not None:
            tile_rgb, tile_known, tile_pos, hatch_rgb, row = sampler.sample_step()
        else:
            tile_rgb, tile_known, tile_pos, hatch_rgb, row = sample_training_step(rng, doc_ids, docs_map, cache, tile_size)

        selected_queries.append(str(row["id"]))
        selected_docs.append(str(row["document_id"]))

        if float(tile_known.sum()) == 0.0:
            continue

        loss_val, bce_val, dice_val = execute_training_step(
            model, optimizer, trainable_params, device,
            tile_rgb, tile_known, tile_pos, hatch_rgb, tile_size, step,
        )
        loss_history.append(loss_val)

        if step % 10 == 0:
            rec = {
                "step": step, "loss": round(loss_val, 6),
                "bce_loss": round(bce_val, 6),
                "dice_loss": round(dice_val, 6) if dice_val is not None else None,
                "elapsed": round(time.monotonic() - start_time, 2),
            }
            print(json.dumps(rec), flush=True)

        if step % save_every == 0 and step < steps:
            save_atomic_checkpoint(model, output_path.parent / f"{output_path.stem}-step-{step}.pt")

    model.eval()
    elapsed = time.monotonic() - start_time
    save_atomic_checkpoint(model, output_path)

    branch_counts = dict(sampler.branch_counts) if sampler is not None else None
    provenance = build_provenance(
        pretrained_sha256, manifest_sha256, sorted(fit_doc_ids), seed, steps, lr, tile_size,
        sum(p.numel() for p in trainable_params), elapsed, loss_history,
        sampling=sampling,
        trainable_scope=trainable_scope,
        torch_seed=seed,
        starting_model_sha256=starting_model_sha256,
        frozen_modules=frozen_modules,
        selected_query_ids=selected_queries,
        selected_document_ids=selected_docs,
        branch_counts=branch_counts,
    )
    provenance.update({
        "implementation_sha256": sha256_file(Path(__file__)),
        "output_sha256": sha256_file(output_path),
        "updated_steps": len(loss_history),
        "training_queries": len(rows),
        "optimizer": "AdamW", "weight_decay": 1e-4,
    })
    output_path.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return provenance


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/splits/fit.json"), help="Manifest path")
    parser.add_argument("--data-root", type=Path, default=Path("dataset"), help="Dataset root path")
    parser.add_argument("--pretrained", type=Path, default=Path("models/hatchfinder-3.5m.pt"), help="Pretrained model")
    parser.add_argument("--output", type=Path, default=Path("models/hatchfinder-fit.pt"), help="Destination weights")
    parser.add_argument("--device", choices=("mps", "cpu"), default=None, help="Target device")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="Number of training steps")
    parser.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE, help="Tile size (multiple of 64)")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="AdamW learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    parser.add_argument("--save-every", type=int, default=DEFAULT_SAVE_EVERY, help="Checkpoint frequency")
    parser.add_argument("--sampling", choices=("random", "coverage"), default="random", help="Sampling strategy")
    parser.add_argument(
        "--trainable-scope", choices=("head", "all"), default="head",
        help="Train only the prediction head (legacy default) or all model parameters",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(None if argv is None else list(argv))
    validate_train_args(args)
    device = select_device(args.device)
    train(
        manifest_path=args.manifest, data_root=args.data_root, pretrained_path=args.pretrained,
        output_path=args.output, device=device, steps=int(args.steps), tile_size=int(args.tile_size),
        lr=float(args.lr), seed=int(args.seed), save_every=int(args.save_every),
        sampling=str(args.sampling),
        trainable_scope=str(args.trainable_scope),
    )


if __name__ == "__main__":
    main()
