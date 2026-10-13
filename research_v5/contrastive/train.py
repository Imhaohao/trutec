"""Train the query-conditioned embedding on the fit split only."""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = "0.4"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.7"

import argparse
import json
import math
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

from grids import SAMPLING_NOTES, DocumentStore, iter_documents, load_fit_rows
from infer import pick_device
from loss import contrastive_loss
from model import DESIGN_NOTES, EMBED_DIM, HatchEncoder, parameter_count

WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0


def main() -> None:
    args = parse_args()
    torch.set_num_threads(4)
    device = pick_device(args.device)
    _set_seed(args.seed)
    store = DocumentStore(load_fit_rows(args.manifest), args.data_root)
    model = HatchEncoder().to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    print(f"parameters {parameter_count(model)} device {device}", flush=True)
    history = _run_steps(model, opt, store, args, device)
    _write_progress(args, model, history, device)
    print(f"checkpoint {args.checkpoint}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--loss-log", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--crop", type=int, default=512)
    parser.add_argument("--steps-per-visit", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--max-rss-gb", type=float, default=11.0)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    return parser.parse_args()


def _run_steps(model, opt, store, args, device) -> list[dict]:
    rng = np.random.default_rng(args.seed)
    docs = iter_documents(store.document_ids(), args.steps_per_visit, rng)
    history: list[dict] = []
    step = 0
    attempts = 0
    started = time.perf_counter()
    while step < args.steps:
        attempts += 1
        _too_many_skips(attempts, args.steps)
        stats = _attempt(model, opt, store, args, device, rng, next(docs))
        if stats is None:
            continue
        step += 1
        stats = _decorate(stats, step, attempts, started)
        history.append(stats)
        if _due(step, args):
            _write_progress(args, model, history, device)
            _print_step(stats)
        _maybe_abort(stats, args, model, history, device)
    return history


def _attempt(model, opt, store, args, device, rng, doc_id):
    batch = store.sample(doc_id, rng, args.crop)
    if batch is None:
        return None
    return _apply_step(model, opt, batch, args, device)


def _apply_step(model, opt, batch, args, device):
    query = _batch_tensor(batch["query"], device)
    target = _batch_tensor(batch["target"], device)
    labels = torch.from_numpy(np.ascontiguousarray(batch["labels"])).to(device)
    mask = torch.from_numpy(np.ascontiguousarray(batch["query_mask"])).to(device)
    opt.zero_grad(set_to_none=True)
    anchor = _anchor(model(query)[0], mask)
    if anchor is None:
        return None
    cells = model(target)[0].permute(1, 2, 0).reshape(-1, EMBED_DIM)
    loss, stats = contrastive_loss(anchor, cells, labels.reshape(-1), args.temperature)
    if not math.isfinite(stats["loss"]):
        raise RuntimeError("non-finite loss")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    opt.step()
    return stats


def _anchor(embedding: torch.Tensor, mask: torch.Tensor):
    ys, xs = torch.nonzero(mask, as_tuple=True)
    if ys.numel() == 0:
        return None
    return torch.nn.functional.normalize(embedding[:, ys, xs].mean(dim=1), dim=0)


def _batch_tensor(array: np.ndarray, device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(array)).unsqueeze(0).to(device)


def _decorate(stats: dict, step: int, attempts: int, started: float) -> dict:
    enriched = dict(stats)
    enriched["step"] = step
    enriched["attempts"] = attempts
    enriched["seconds"] = time.perf_counter() - started
    enriched.update(_memory())
    return enriched


def _due(step: int, args) -> bool:
    return step == 1 or step % args.log_every == 0 or step == args.steps


def _maybe_abort(stats, args, model, history, device) -> None:
    rss = stats["rss_gb"]
    driver = stats["mps_driver_gb"]
    if rss <= args.max_rss_gb and driver <= args.max_rss_gb:
        return
    _write_progress(args, model, history, device)
    raise MemoryError(f"memory rss={rss:.2f} GB mps={driver:.2f} GB limit={args.max_rss_gb}")


def _write_progress(args, model, history, device) -> None:
    _save_checkpoint(model, args.checkpoint)
    payload = _payload(args, model, history, device)
    _write_json(args.loss_log, payload)


def _payload(args, model, history, device) -> dict:
    return {
        "seed": args.seed,
        "steps": args.steps,
        "completed_steps": history[-1]["step"] if history else 0,
        "lr": args.lr,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip": GRAD_CLIP,
        "temperature": args.temperature,
        "crop": args.crop,
        "steps_per_visit": args.steps_per_visit,
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "device": str(device),
        "parameters": parameter_count(model),
        "embedding_dim": EMBED_DIM,
        "notes": list(DESIGN_NOTES) + list(SAMPLING_NOTES),
        "losses": history,
    }


def _save_checkpoint(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _print_step(stats: dict) -> None:
    print(
        f"step {stats['step']} loss {stats['loss']:.4f} nce {stats['nce']:.4f} "
        f"bce {stats['bce']:.4f} pos_score {stats['pos_score']} neg_score {stats['neg_score']} "
        f"rss {stats['rss_gb']:.2f} mps {stats['mps_driver_gb']:.2f} sec {stats['seconds']:.1f}",
        flush=True,
    )


def _memory() -> dict:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_gb = rss / (1024 ** 3) if sys.platform == "darwin" else rss / (1024 ** 2)
    current = 0.0
    driver = 0.0
    if torch.backends.mps.is_available():
        current = torch.mps.current_allocated_memory() / (1024 ** 3)
        driver = torch.mps.driver_allocated_memory() / (1024 ** 3)
    return {"rss_gb": rss_gb, "mps_current_gb": current, "mps_driver_gb": driver}


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def _too_many_skips(attempts: int, steps: int) -> None:
    if attempts > steps * 20:
        raise RuntimeError("too many skipped steps; crops are not finding both classes")


if __name__ == "__main__":
    main()
