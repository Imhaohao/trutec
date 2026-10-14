"""Train the large-context hatch model on the fit split only."""

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

from .constants import GRAD_CLIP, LR, SEED, WEIGHT_DECAY
from .data import FitSampler
from .infer import pick_device
from .loss import masked_loss
from .model import HatchContextNet, parameter_count


def main() -> None:
    args = parse_args()
    torch.set_num_threads(4)
    device = pick_device(args.device)
    _set_seed(args.seed)
    sampler = FitSampler(args.manifest, args.data_root, args.seed, args.half_crop, args.steps_per_visit)
    model = HatchContextNet().to(device)
    if args.init is not None:
        model.load_state_dict(torch.load(args.init, map_location=device, weights_only=True))
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.steps, 1))
    print(f"parameters {parameter_count(model)} device {device} half_crop {args.half_crop}", flush=True)
    history = _run_steps(model, opt, sched, sampler, args, device)
    _write_progress(args, model, history, device)
    print(f"checkpoint {args.checkpoint}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--loss-log", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--half-crop", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--steps-per-visit", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--max-rss-gb", type=float, default=11.5)
    parser.add_argument("--init", type=Path, default=None)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    return parser.parse_args()


def _run_steps(model, opt, sched, sampler, args, device) -> list[dict]:
    history: list[dict] = []
    step = 0
    attempts = 0
    started = time.perf_counter()
    while step < args.steps:
        attempts += 1
        _too_many_skips(attempts, args.steps)
        stats = _attempt(model, opt, sched, sampler, args, device)
        if stats is None:
            continue
        step += 1
        stats = _decorate(stats, step, attempts, started, opt)
        history.append(stats)
        if _due(step, args):
            _write_progress(args, model, history, device)
            _print_step(stats)
        _maybe_abort(stats, args, model, history, device)
    return history


def _attempt(model, opt, sched, sampler, args, device):
    batch = sampler.sample_batch(args.batch)
    if batch is None:
        return None
    return _apply_step(model, opt, sched, batch, device)


def _apply_step(model, opt, sched, batch, device):
    drawing = _tensor(batch["drawing"], device)
    query = _tensor(batch["query"], device)
    known = _tensor(batch["known"], device)
    positive = _tensor(batch["positive"], device)
    opt.zero_grad(set_to_none=True)
    loss, stats = masked_loss(model(drawing, query), known, positive)
    if not math.isfinite(stats["loss"]):
        raise RuntimeError("non-finite loss")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    opt.step()
    sched.step()
    return stats


def _tensor(array: np.ndarray, device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(array)).to(device)


def _decorate(stats: dict, step: int, attempts: int, started: float, opt) -> dict:
    enriched = dict(stats)
    enriched["step"] = step
    enriched["attempts"] = attempts
    enriched["seconds"] = time.perf_counter() - started
    enriched["lr"] = float(opt.param_groups[0]["lr"])
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
    _write_json(args.loss_log, _payload(args, model, history, device))


def _payload(args, model, history, device) -> dict:
    return {
        "seed": args.seed,
        "steps": args.steps,
        "completed_steps": history[-1]["step"] if history else 0,
        "lr": args.lr,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip": GRAD_CLIP,
        "batch": args.batch,
        "half_crop": args.half_crop,
        "native_crop": args.half_crop * 2,
        "steps_per_visit": args.steps_per_visit,
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "init": None if args.init is None else str(args.init),
        "device": str(device),
        "parameters": parameter_count(model),
        "peak_rss_gb": max((row["rss_gb"] for row in history), default=0.0),
        "seconds_per_step": _seconds_per_step(history),
        "losses": history,
    }


def _seconds_per_step(history: list[dict]) -> float | None:
    if not history:
        return None
    return history[-1]["seconds"] / history[-1]["step"]


def _save_checkpoint(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cpu_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save(cpu_state, path)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _print_step(stats: dict) -> None:
    print(
        f"step {stats['step']} loss {stats['loss']:.4f} bce {stats['bce']:.4f} "
        f"dice {stats['dice']:.4f} known {stats['known_cells']} lr {stats['lr']:.6f} "
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
    if attempts > steps * 30:
        raise RuntimeError("too many skipped steps; crops are not finding known cells")


if __name__ == "__main__":
    main()
