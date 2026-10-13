"""v10 = frozen v7 scores with a hard ink-color veto.

The query's colour signature and the 16-pixel window colour maps are the ones the v7
colour gate already uses (research_v5/color/color_features.py). Where the window
mismatch exceeds VETO_LEVEL, the score is set to zero instead of only being scaled.
Batch-independent: each request uses only its own drawing and box.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "combo"))
import bootstrap  # noqa: E402,F401

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from collections import defaultdict  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from bootstrap import STRUCTURE_MODEL  # noqa: E402
from color.color_features import (  # noqa: E402
    build_image_color_cache,
    compute_sample_mismatch,
    extract_robust_query_signature,
)
from evaluate_dev import load_original  # noqa: E402
from pipeline import cleaner_by_name, visit_maps  # noqa: E402

CLEANER = "g2"
COLOR_WINDOW = 16
VETO_LEVEL = 0.7
BAND_ROWS = 256
REQUEST_FIELDS = ("id", "image", "width", "height", "query_box", "context_boxes")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--threshold", required=True, type=float)
    return parser.parse_args()


def load_requests(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["examples"] if isinstance(payload, dict) else payload
    return [{key: row[key] for key in REQUEST_FIELDS if key in row} for row in rows]


def vetoed(score: np.ndarray, rgb: np.ndarray, color_maps, box) -> np.ndarray:
    signature = extract_robust_query_signature(rgb, tuple(int(v) for v in box))
    height, width = score.shape
    columns = np.arange(width)
    result = score.copy()
    for top in range(0, height, BAND_ROWS):
        bottom = min(top + BAND_ROWS, height)
        yy, xx = np.meshgrid(np.arange(top, bottom), columns, indexing="ij")
        mismatch = compute_sample_mismatch(signature, color_maps, yy.ravel(), xx.ravel()).reshape(bottom - top, width)
        band = result[top:bottom]
        band[mismatch > VETO_LEVEL] = 0.0
    return result


def predict_drawing(requests: list[dict], stage1, stage2, data_root: Path, emit) -> None:
    with Image.open(data_root / requests[0]["image"]) as handle:
        rgb = np.asarray(handle.convert("RGB"))
    color_maps = build_image_color_cache(rgb, window_sizes=(COLOR_WINDOW,))[COLOR_WINDOW]

    def visit(request: dict, gated: np.ndarray) -> None:
        emit(request, vetoed(gated, rgb, color_maps, request["query_box"]))

    visit_maps(requests, stage1, stage2, cleaner_by_name(CLEANER), data_root, visit)


def write_mask(path: Path, selected: np.ndarray) -> None:
    Image.fromarray(np.where(selected, 255, 0).astype(np.uint8), mode="L").save(path)


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage1, _metadata = load_original()
    stage2 = joblib.load(STRUCTURE_MODEL)["model"]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for request in load_requests(args.inputs):
        grouped[request["image"]].append(request)
    runtime = {"threshold": args.threshold, "drawings": []}

    def emit(request: dict, score: np.ndarray) -> None:
        write_mask(args.output_dir / f"{request['id']}.png", score >= args.threshold)

    for image, requests in grouped.items():
        drawing_started = time.perf_counter()
        predict_drawing(requests, stage1, stage2, args.data_root, emit)
        runtime["drawings"].append({"image": image, "queries": len(requests),
                                    "seconds": time.perf_counter() - drawing_started})
        print(f"wrote {len(requests)} masks for {image}", flush=True)
    runtime["elapsed_seconds"] = time.perf_counter() - started
    (args.output_dir / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
