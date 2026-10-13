"""Native v6 inference: cross-fitted context stacker over crop HGB, then the fixed color gate.

Every pixel reads the stage-2 probability of its stride-8 cell (the same lookup the
development scorer uses), is multiplied by the w16 linear color gate (strength 0.6),
and is selected when the gated score reaches the frozen threshold. Requests carry no
labels; only the drawing, the query box and the model files are read.
"""
from __future__ import annotations

import bootstrap  # noqa: F401

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from PIL import Image

import drawing
from color.color_features import (
    apply_gate,
    build_image_color_cache,
    compute_sample_mismatch,
    extract_robust_query_signature,
)
from constants import CACHE, GRID_STRIDE
from evaluate_dev import load_original
from model import positive_probability

COLOR_WINDOW = 16
COLOR_STRENGTH = 0.6
BAND_ROWS = 256
DEFAULT_STAGE2 = CACHE / "stage2_model.joblib"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path, help="Label-free request manifest")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--threshold", required=True, type=float)
    parser.add_argument("--stage2", type=Path, default=DEFAULT_STAGE2)
    return parser.parse_args()


def load_requests(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["examples"] if isinstance(payload, dict) else payload
    allowed = ("id", "image", "width", "height", "query_box", "context_boxes")
    return [{key: row[key] for key in allowed if key in row} for row in rows]


def native_cell_lookup(grid: np.ndarray, height: int, width: int) -> np.ndarray:
    rows = np.minimum(np.arange(height) // GRID_STRIDE, grid.shape[0] - 1)
    cols = np.minimum(np.arange(width) // GRID_STRIDE, grid.shape[1] - 1)
    return grid[rows[:, None], cols[None, :]]


def stage2_grid(stage2, volume: np.ndarray) -> np.ndarray:
    height, width, channels = volume.shape
    return positive_probability(stage2, volume.reshape(-1, channels)).reshape(height, width)


def gated_native(score: np.ndarray, rgb: np.ndarray, color_maps, box) -> np.ndarray:
    signature = extract_robust_query_signature(rgb, tuple(int(v) for v in box))
    height, width = score.shape
    columns = np.arange(width)
    gated = np.empty_like(score, dtype=np.float32)
    for top in range(0, height, BAND_ROWS):
        bottom = min(top + BAND_ROWS, height)
        yy, xx = np.meshgrid(np.arange(top, bottom), columns, indexing="ij")
        mismatch = compute_sample_mismatch(signature, color_maps, yy.ravel(), xx.ravel())
        band = score[top:bottom].ravel()
        gated[top:bottom] = apply_gate(band, mismatch, gate_type="linear", strength=COLOR_STRENGTH).reshape(
            bottom - top, width
        )
    return gated


def predict_drawing(requests: list[dict], stage1, stage2, data_root: Path, emit) -> None:
    """Call emit(request, gated_native_score) once per request on this drawing."""
    drawing.DATA_ROOT = data_root
    spec = {key: requests[0][key] for key in ("id", "image", "width", "height")}
    grids: list[np.ndarray] = []
    drawing.consume_volumes(
        spec,
        [request["query_box"] for request in requests],
        [stage1] * len(requests),
        lambda index, volume: grids.append(stage2_grid(stage2, volume)),
    )
    with Image.open(data_root / spec["image"]) as image:
        rgb = np.asarray(image.convert("RGB"))
    color_maps = build_image_color_cache(rgb, window_sizes=(COLOR_WINDOW,))[COLOR_WINDOW]
    for request, grid in zip(requests, grids):
        native = native_cell_lookup(grid, int(request["height"]), int(request["width"]))
        emit(request, gated_native(native, rgb, color_maps, request["query_box"]))


def write_mask(path: Path, selected: np.ndarray) -> None:
    Image.fromarray(np.where(selected, 255, 0).astype(np.uint8), mode="L").save(path)


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage1, _metadata = load_original()
    stage2 = joblib.load(args.stage2)["model"]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for request in load_requests(args.inputs):
        grouped[request["image"]].append(request)
    runtime = {"threshold": args.threshold, "stage2": str(args.stage2), "drawings": []}

    def emit(request: dict, gated: np.ndarray) -> None:
        write_mask(args.output_dir / f"{request['id']}.png", gated >= args.threshold)

    for image, requests in grouped.items():
        drawing_started = time.perf_counter()
        predict_drawing(requests, stage1, stage2, args.data_root, emit)
        runtime["drawings"].append(
            {"image": image, "queries": len(requests), "seconds": time.perf_counter() - drawing_started}
        )
        print(f"wrote {len(requests)} masks for {image}", flush=True)
    runtime["elapsed_seconds"] = time.perf_counter() - started
    (args.output_dir / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
