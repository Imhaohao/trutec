"""Run the bounded, label-free cadence feature resource and equivalence pilot."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate import EvaluationError, load_manifest
from hatch_cadence import LagState, cadence_features, cadence_features_tiled, estimate_query_lag
from inference import read_drawing
from tools.dino_patch_support import FIT_MANIFEST_SHA256

PROTOCOL = ROOT / "artifacts/research-v4/cadence-feature-protocol.json"
PILOT_SOURCE = Path(__file__).resolve()
PATCH_STRIDE = 4
CORE_SIZE = 672
POINT_SEED = 42
POINT_CAP = 4096


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"Cannot read JSON: {path}") from exc
    if not isinstance(value, dict):
        raise EvaluationError(f"Expected JSON object: {path}")
    return value


def _manifest_rows(path: Path) -> tuple[list[dict[str, Any]], str]:
    digest = _sha_file(path)
    if digest != FIT_MANIFEST_SHA256:
        raise EvaluationError("Cadence pilot requires the frozen fit manifest")
    return load_manifest(path, require_labels=False), digest


def _verify_support(support: dict[str, Any], rows: list[dict[str, Any]], manifest_sha: str) -> dict[str, dict[str, Any]]:
    if support.get("schema") != "cadence-fit-support/v1":
        raise EvaluationError("Unsupported cadence fit-support schema")
    if support.get("manifest_sha256") != manifest_sha or support.get("protocol_sha256") != _sha_file(PROTOCOL):
        raise EvaluationError("Fit-support manifest or protocol hash mismatch")
    source_hashes = support.get("source_sha256")
    if not isinstance(source_hashes, dict):
        raise EvaluationError("Fit-support report lacks source hashes")
    for relative in ("hatch_cadence.py", "tools/cadence_fit_support.py"):
        if source_hashes.get(relative) != _sha_file(ROOT / relative):
            raise EvaluationError(f"Fit-support source hash mismatch: {relative}")
    reports = support.get("per_query")
    if not isinstance(reports, list):
        raise EvaluationError("Fit-support report lacks query records")
    by_id: dict[str, dict[str, Any]] = {}
    for record in reports:
        if not isinstance(record, dict) or not isinstance(record.get("id"), str):
            raise EvaluationError("Fit-support query IDs must be strings")
        by_id[record["id"]] = record
    row_ids = {str(row["id"]) for row in rows}
    if len(by_id) != len(reports) or set(by_id) != row_ids:
        raise EvaluationError("Fit-support query IDs do not match the pinned fit manifest")
    row_docs = {str(row["id"]): str(row["document_id"]) for row in rows}
    if any(record.get("document_id") != row_docs[ident] for ident, record in by_id.items()):
        raise EvaluationError("Fit-support document IDs differ from the fit manifest")
    gate = support.get("support_gate")
    if not isinstance(gate, dict) or gate.get("passed") is not True:
        raise EvaluationError("Cadence numerical support gate did not pass")
    stable_dual = [r for r in reports if r.get("stable") is True and r.get("dual_class") is True]
    dual_documents = {str(r["document_id"]) for r in stable_dual}
    if (len(stable_dual) < 30 or len(dual_documents) < 10
            or gate.get("stable_dual_queries") != len(stable_dual)
            or gate.get("stable_dual_documents") != len(dual_documents)):
        raise EvaluationError("Cadence stable dual-class support is below the frozen gate")
    return by_id


def _pick_row(rows: list[dict[str, Any]], reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    stable_ids = {ident for ident, report in reports.items() if report.get("stable") is True}
    eligible = [row for row in rows if str(row["id"]) in stable_ids]
    if not eligible:
        raise EvaluationError("No fit query has a stable available lag")
    images: dict[str, list[dict[str, Any]]] = {}
    for row in eligible:
        images.setdefault(str(row["image"]), []).append(row)
    selected_image = max(images, key=lambda image: (
        max(int(row["width"]) * int(row["height"]) for row in images[image]), image))
    return min(images[selected_image], key=lambda row: str(row["id"]))


def _lag_from_report(gray: np.ndarray, row: dict[str, Any], report: dict[str, Any]) -> tuple[np.ndarray, Any]:
    x0, y0, x1, y1 = row["query_box"]
    crop = gray[y0:y1, x0:x1]
    crop_digest = _sha_bytes(crop.tobytes())
    lag = estimate_query_lag(crop)
    canonical_lag = json.loads(json.dumps(asdict(lag)))
    if (list(crop.shape) != report.get("crop_shape") or crop_digest != report.get("crop_pixel_sha256")
            or report.get("stable") is not lag.available or canonical_lag != report.get("lag")):
        raise EvaluationError("Selected query crop or recomputed lag differs from fit-support evidence")
    return crop, lag


def _boundary_points(grid_h: int, grid_w: int, core_cells: int) -> np.ndarray:
    points: set[tuple[int, int]] = {(0, 0), (0, grid_w - 1), (grid_h - 1, 0), (grid_h - 1, grid_w - 1)}
    for y0 in range(0, grid_h, core_cells):
        y1 = min(y0 + core_cells, grid_h) - 1
        for x0 in range(0, grid_w, core_cells):
            x1 = min(x0 + core_cells, grid_w) - 1
            points.update(((y0, x0), (y0, x1), (y1, x0), (y1, x1)))
    return np.asarray(sorted(points), dtype=np.int64).reshape(-1, 2)


def select_reference_points(grid_h: int, grid_w: int, core_size: int = CORE_SIZE,
                            point_cap: int = POINT_CAP, seed: int = POINT_SEED) -> np.ndarray:
    if min(grid_h, grid_w, core_size, point_cap) <= 0 or core_size % PATCH_STRIDE:
        raise ValueError("Grid dimensions, cap, and stride-aligned core size must be positive")
    core_cells = core_size // PATCH_STRIDE
    boundary = _boundary_points(grid_h, grid_w, core_cells)
    grid_count = grid_h * grid_w
    boundary_flat = boundary[:, 0] * grid_w + boundary[:, 1]
    remaining_count = grid_count - len(boundary_flat)
    sample_count = min(point_cap, remaining_count)
    if sample_count:
        rng = np.random.default_rng(seed)
        candidates = np.setdiff1d(np.arange(grid_count, dtype=np.int64), boundary_flat, assume_unique=False)
        picks = rng.choice(candidates, size=sample_count, replace=False)
        sampled = np.column_stack((picks // grid_w, picks % grid_w))
        return np.unique(np.vstack((boundary, sampled)), axis=0)
    return boundary


def _system_swap_snapshot() -> dict[str, Any]:
    before = datetime.now(timezone.utc).isoformat()
    try:
        result = subprocess.run(["vm_stat"], check=True, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        raise EvaluationError("Cannot capture system swapout baseline with vm_stat") from exc
    page_match = re.search(r"page size of (\d+) bytes", result.stdout)
    swap_match = re.search(r"^Swapouts:\s*(\d+)\.?\s*$", result.stdout, re.MULTILINE)
    if page_match is None or swap_match is None:
        raise EvaluationError("vm_stat output lacks page size or Swapouts counter")
    return {"captured_at_utc": before, "page_size_bytes": int(page_match.group(1)),
            "swapout_pages": int(swap_match.group(1))}


def _all_coordinates(grid_h: int, grid_w: int) -> np.ndarray:
    gy: np.ndarray
    gx: np.ndarray
    gy, gx = np.indices((grid_h, grid_w), dtype=np.int64)
    return np.column_stack((gy.ravel(), gx.ravel()))


def _feature_grid(gray: np.ndarray, lag: Any) -> tuple[np.ndarray, np.ndarray]:
    grid_h, grid_w = math.ceil(gray.shape[0] / PATCH_STRIDE), math.ceil(gray.shape[1] / PATCH_STRIDE)
    coords = _all_coordinates(grid_h, grid_w)
    values = cadence_features_tiled(gray, lag, coords).reshape(grid_h, grid_w, 5)
    if values.shape != (grid_h, grid_w, 5) or not np.isfinite(values).all():
        raise EvaluationError("Tiled cadence grid has invalid shape or non-finite features")
    if np.any((values < 0) | (values > 1)):
        raise EvaluationError("Tiled cadence features lie outside [0,1]")
    if not lag.available and not np.array_equal(values, np.broadcast_to(
            np.asarray([0.5, 0.5, 0.5, 0.5, 0.0], np.float32), values.shape)):
        raise EvaluationError("Unavailable lag did not use the explicit feature fallback")
    return coords, values


def _verify_unavailable_fallback(gray: np.ndarray) -> bool:
    unavailable = LagState(None, None, (None, None, None, None))
    observed = cadence_features_tiled(gray, unavailable, np.asarray([[0, 0]], dtype=np.int64))[0]
    expected = np.asarray([0.5, 0.5, 0.5, 0.5, 0.0], dtype=np.float32)
    if not np.array_equal(observed, expected):
        raise EvaluationError("Unavailable cadence lag did not use the explicit feature fallback")
    return True


def _compare_paths(gray: np.ndarray, lag: Any, coords: np.ndarray, dense: np.ndarray) -> dict[str, Any]:
    grid_h, grid_w = dense.shape[:2]
    points = select_reference_points(grid_h, grid_w)
    direct = cadence_features(gray, lag, points)
    sparse = cadence_features_tiled(gray, lag, points)
    dense_selected = dense[points[:, 0], points[:, 1]]
    direct_error = float(np.max(np.abs(direct - dense_selected))) if len(points) else 0.0
    sparse_error = float(np.max(np.abs(sparse - dense_selected))) if len(points) else 0.0
    if direct_error > 1e-6 or sparse_error > 1e-6:
        raise EvaluationError("Direct/tiled or sparse/dense cadence equivalence exceeded 1e-6")
    if coords.shape != (grid_h * grid_w, 2):
        raise EvaluationError("Dense cadence grid coordinates are misaligned")
    return {"reference_points": int(len(points)), "reference_seed": POINT_SEED,
            "direct_max_abs_diff": direct_error, "sparse_dense_max_abs_diff": sparse_error}


def run_pilot(manifest: Path, data_root: Path, support_path: Path, output: Path) -> dict[str, Any]:
    rows, manifest_sha = _manifest_rows(manifest)
    support = _json(support_path)
    reports = _verify_support(support, rows, manifest_sha)
    row = _pick_row(rows, reports)
    baseline = _system_swap_snapshot()
    started = time.monotonic()
    try:
        gray = read_drawing(data_root, row)
    except (OSError, ValueError) as exc:
        raise EvaluationError(f"Cannot read selected fit drawing: {row['image']}") from exc
    if gray.dtype != np.uint8 or gray.ndim != 2 or gray.shape != (row["height"], row["width"]):
        raise EvaluationError("Selected fit drawing must match its native uint8 manifest geometry")
    crop, lag = _lag_from_report(gray, row, reports[str(row["id"])])
    fallback_verified = _verify_unavailable_fallback(crop)
    coords, tiled = _feature_grid(gray, lag)
    equivalence = _compare_paths(gray, lag, coords, tiled)
    finished = time.monotonic()
    after = _system_swap_snapshot()
    if after["page_size_bytes"] != baseline["page_size_bytes"]:
        raise EvaluationError("vm_stat page size changed during pilot")
    delta_pages = after["swapout_pages"] - baseline["swapout_pages"]
    if delta_pages < 0:
        raise EvaluationError("System swapout counter decreased during pilot")
    input_digest = _sha_bytes(gray.tobytes())
    crop_digest = _sha_bytes(crop.tobytes())
    proof = {
        "schema": "cadence-resource-pilot/v1",
        "status": "system_swap_passed_process_gate_pending" if delta_pages == 0 else "failed",
        "manifest_sha256": manifest_sha, "protocol_sha256": _sha_file(PROTOCOL),
        "support_sha256": _sha_file(support_path),
        "source_sha256": {"hatch_cadence.py": _sha_file(ROOT / "hatch_cadence.py"),
                           "tools/cadence_resource_pilot.py": _sha_file(PILOT_SOURCE)},
        "drawing": row["image"], "query_id": row["id"], "document_id": row["document_id"],
        "drawing_pixel_sha256": input_digest, "query_crop_pixel_sha256": crop_digest,
        "query_crop_shape": list(crop.shape), "query_lag": asdict(lag),
        "lag_available": lag.available,
        "native_grid": {"rows": tiled.shape[0], "columns": tiled.shape[1],
                        "stride": PATCH_STRIDE, "center_phase_yx": [2, 2]},
        "feature_shape": list(tiled.shape), "feature_min": tiled.min(axis=(0, 1)).tolist(),
        "feature_max": tiled.max(axis=(0, 1)).tolist(), "all_features_finite": bool(np.isfinite(tiled).all()),
        "unavailable_fallback_verified": fallback_verified,
        **equivalence,
        "wall_seconds": finished - started,
        "system_swap": {"before": baseline, "after": after, "swapout_delta_pages": delta_pages,
                        "swapout_delta_bytes": delta_pages * baseline["page_size_bytes"]},
        "process_swap_count": None,
        "peak_process_footprint_bytes": None,
        "process_resource_measurement": "external /usr/bin/time -l wrapper required",
        "process_resource_gate_passed": None,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(proof, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    if delta_pages != 0:
        raise EvaluationError("Cadence pilot increased system swapouts; failed proof was saved")
    return proof


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--support", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = run_pilot(args.manifest, args.data_root, args.support, args.output)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
