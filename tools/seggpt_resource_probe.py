"""Run one local-checkpoint, synthetic-only SegGPT MPS resource probe."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import platform
from pathlib import Path
import resource
import re
import subprocess
import time
from typing import Any

import numpy as np

from seggpt_matching import (
    INPUT_SIDE,
    PATCHES_PER_VIEW,
    TOTAL_PATCHES,
    SegGptError,
    encode_pair,
    ensure_mps_fallback_disabled,
    load_local_model,
    predict_encoded,
    select_mps,
)

RSS_LIMIT_BYTES = 12 * 1024**3


class ProbeGateError(SegGptError):
    """Resource gate failure carrying the proof collected before stopping."""

    def __init__(self, message: str, proof: dict[str, Any]) -> None:
        super().__init__(message)
        self.proof = proof


def synthetic_case() -> tuple[np.ndarray, np.ndarray]:
    """Build deterministic image arrays without touching project data."""
    y, x = np.indices((128, 160))
    query = np.where(((x // 8 + y // 8) % 2) == 0, 36, 220).astype(np.uint8)
    target_y, target_x = np.indices((INPUT_SIDE, INPUT_SIDE))
    target = ((target_x * 17 + target_y * 31) % 256).astype(np.uint8)
    return target, query


def process_peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if platform.system() == "Darwin" else peak * 1024)


def mps_memory_snapshot(torch_module: Any) -> dict[str, int | None]:
    mps = torch_module.mps
    return {
        "current_allocated_bytes": _mps_counter(mps, "current_allocated_memory"),
        "driver_allocated_bytes": _mps_counter(mps, "driver_allocated_memory"),
        "recommended_max_bytes": _mps_counter(mps, "recommended_max_memory"),
    }


def run_probe(checkpoint_dir: Path) -> dict[str, Any]:
    """Load verified local weights and run one batch-one synthetic forward."""
    ensure_mps_fallback_disabled()
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SegGptError("PyTorch must be installed to run the SegGPT resource probe") from exc
    device = select_mps(torch)
    torch.mps.set_per_process_memory_fraction(0.75)
    torch.mps.synchronize()
    memory = {"before_load": mps_memory_snapshot(torch)}
    rss = {"before_load": process_peak_rss_bytes()}
    swap_before = system_swap_snapshot()
    start_load = time.perf_counter()
    model, processor, pins = load_local_model(checkpoint_dir, torch)
    torch.mps.synchronize()
    load_seconds = time.perf_counter() - start_load
    memory["after_load"] = mps_memory_snapshot(torch)
    rss["after_load"] = process_peak_rss_bytes()
    if rss["after_load"] > RSS_LIMIT_BYTES:
        proof = _failure_proof("post_load_rss_limit", rss, None)
        raise ProbeGateError("post-load process RSS exceeds 12 GiB; forward skipped", proof)
    target, query = synthetic_case()
    encoded, mask_positions, _ = encode_pair(processor, target, query, device, torch)
    _validate_tensor_shapes(encoded)
    mask_counts = _mask_counts(mask_positions)
    _validate_mask_counts(mask_counts)
    forward_start = time.perf_counter()
    scores = predict_encoded(
        model, processor, encoded, mask_positions, (INPUT_SIDE, INPUT_SIDE), device, torch
    )
    torch.mps.synchronize()
    forward_seconds = time.perf_counter() - forward_start
    memory["after_forward"] = mps_memory_snapshot(torch)
    rss["after_forward"] = process_peak_rss_bytes()
    checks = _output_checks(scores)
    if not all(checks.values()):
        raise SegGptError(f"synthetic output failed resource-probe checks: {checks}")
    swap_after = system_swap_snapshot()
    swap_delta = swap_after["swapout_pages"] - swap_before["swapout_pages"]
    if swap_after["page_size_bytes"] != swap_before["page_size_bytes"] or swap_delta < 0:
        raise SegGptError("system swap counter changed incompatibly during the resource probe")
    resource_gates = {
        "system_swapouts_unchanged": swap_delta == 0,
        "process_peak_rss_within_limit": max(rss.values()) <= RSS_LIMIT_BYTES,
    }
    if not all(resource_gates.values()):
        proof = _failure_proof("post_forward_resource_limit", rss, {
            "before": swap_before, "after": swap_after, "swapout_delta_pages": swap_delta
        }, forward_performed=True)
        proof["resource_gates"] = resource_gates
        raise ProbeGateError("resource gate failed: system swapouts must stay unchanged and process RSS <= 12 GiB", proof)
    parameters = list(model.parameters())
    return {
        "schema": "seggpt-synthetic-resource-probe/v1",
        "scope": "one batch-one synthetic forward; no dataset, labels, optimizer, or training",
        "model": pins,
        "runtime": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pillow": _package_version("Pillow"),
            "torch": str(torch.__version__),
            "transformers": _package_version("transformers"),
            "torchvision": _package_version("torchvision"),
            "device": device,
            "per_process_memory_fraction": 0.75,
            "mps_cpu_fallback_enabled": False,
            "eval_mode": not bool(model.training),
            "all_parameters_frozen": all(not parameter.requires_grad for parameter in parameters),
            "parameter_dtype": sorted({str(parameter.dtype) for parameter in parameters}),
            "load_seconds": load_seconds,
            "forward_seconds": forward_seconds,
            "mps_memory_snapshots": memory,
            "system_swap": {
                "before": swap_before,
                "after": swap_after,
                "swapout_delta_pages": swap_delta,
                "swapout_delta_bytes": swap_delta * swap_before["page_size_bytes"],
            },
            "process_peak_rss_snapshots_bytes": rss,
            "process_rss_stop_limit_bytes": RSS_LIMIT_BYTES,
            "resource_gates": resource_gates,
            "memory_limitations": (
                "MPS counters are snapshots, not device high-water marks. Use external /usr/bin/time -l output "
                "for process peak footprint and process-swap evidence. RSS is a separate internal gate; stop above 12 GiB or if system swapouts rise."
            ),
        },
        "synthetic_forward": {
            "query_shape": [int(query_h) for query_h in query.shape],
            "target_shape": [int(side) for side in target.shape],
            "processor_tensor_shapes": {key: list(value.shape) for key, value in encoded.items()},
            "bool_masked_pos_shape": list(mask_positions.shape),
            "bool_masked_pos_counts": mask_counts,
            "score_shape": list(scores.shape),
            "score_range": [float(scores.min()), float(scores.max())],
            "score_semantics": "normalized binary palette-distance margin; not a calibrated probability",
            "checks": checks,
            "mask_contract": {
                "total_patches": TOTAL_PATCHES,
                "prompt_visible_patches": PATCHES_PER_VIEW,
                "target_masked_patches": PATCHES_PER_VIEW,
                "labels_passed": False,
                "feature_ensemble": False,
            },
        },
    }


def _validate_tensor_shapes(encoded: Any) -> None:
    expected = (1, 3, INPUT_SIDE, INPUT_SIDE)
    actual = {key: tuple(value.shape) for key, value in encoded.items()}
    if set(actual) != {"pixel_values", "prompt_pixel_values", "prompt_masks"}:
        raise SegGptError("processor must return exactly the three SegGPT inference tensors")
    if any(shape != expected for shape in actual.values()):
        raise SegGptError(f"processor tensors violate the batch-one 448 contract: {actual}")


def _mask_counts(mask: Any) -> dict[str, int]:
    if hasattr(mask, "detach"):
        values = mask.detach().to("cpu").numpy()
    else:
        values = np.asarray(mask)
    return {
        "total": int(values.size),
        "prompt_visible": int((~values[..., :PATCHES_PER_VIEW]).sum()),
        "prompt_masked": int(values[..., :PATCHES_PER_VIEW].sum()),
        "target_masked": int(values[..., PATCHES_PER_VIEW:].sum()),
        "target_visible": int((~values[..., PATCHES_PER_VIEW:]).sum()),
    }


def _validate_mask_counts(counts: dict[str, int]) -> None:
    expected = {
        "total": TOTAL_PATCHES,
        "prompt_visible": PATCHES_PER_VIEW,
        "prompt_masked": 0,
        "target_masked": PATCHES_PER_VIEW,
        "target_visible": 0,
    }
    if counts != expected:
        raise SegGptError(f"patch mask does not match the pinned default contract: {counts}")


def _output_checks(scores: np.ndarray) -> dict[str, bool]:
    return {
        "shape": scores.shape == (INPUT_SIDE, INPUT_SIDE),
        "finite": bool(np.isfinite(scores).all()),
        "bounded": bool(((scores >= 0) & (scores <= 1)).all()),
    }


def _failure_proof(
    reason: str,
    rss: dict[str, int],
    swap: dict[str, Any] | None,
    forward_performed: bool = False,
) -> dict[str, Any]:
    return {
        "schema": "seggpt-synthetic-resource-probe/v1",
        "status": "failed",
        "failure_reason": reason,
        "process_peak_rss_snapshots_bytes": rss,
        "process_rss_limit_bytes": RSS_LIMIT_BYTES,
        "system_swap": swap,
        "synthetic_forward": {"performed": forward_performed},
    }


def _mps_counter(mps: Any, name: str) -> int | None:
    counter = getattr(mps, name, None)
    return int(counter()) if callable(counter) else None


def system_swap_snapshot() -> dict[str, Any]:
    captured = datetime.now(timezone.utc).isoformat()
    try:
        result = subprocess.run(["vm_stat"], check=True, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        raise SegGptError("Cannot capture system swapout baseline with vm_stat") from exc
    page_match = re.search(r"page size of (\d+) bytes", result.stdout)
    swap_match = re.search(r"^Swapouts:\s*(\d+)\.?\s*$", result.stdout, re.MULTILINE)
    if page_match is None or swap_match is None:
        raise SegGptError("vm_stat output lacks page size or Swapouts counter")
    return {
        "captured_at_utc": captured,
        "page_size_bytes": int(page_match.group(1)),
        "swapout_pages": int(swap_match.group(1)),
    }


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        proof = run_probe(args.checkpoint_dir)
    except ProbeGateError as exc:
        serialized = json.dumps(exc.proof, indent=2) + "\n"
        if args.output:
            args.output.write_text(serialized, encoding="utf-8")
        else:
            print(serialized, end="")
        raise SystemExit(str(exc)) from exc
    except SegGptError as exc:
        raise SystemExit(str(exc)) from exc
    serialized = json.dumps(proof, indent=2) + "\n"
    if args.output:
        args.output.write_text(serialized, encoding="utf-8")
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
