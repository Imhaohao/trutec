"""v9 = frozen v7 scores, with scores inside detected symbol circles set to zero.

Large labeled symbol bubbles are found with a fixed Hough setting (hough.py). When a
query's own crop contains no such circle centre, its score inside every detected circle
on the drawing is set to zero, because those bubbles are separate symbols, not the
query's hatch. Batch-independent: each request uses only its own drawing and box.
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

sys.path.insert(0, str(HERE))
from hough import circle_mask, detect_circles, query_shows_circles  # noqa: E402

from bootstrap import STRUCTURE_MODEL  # noqa: E402
from evaluate_dev import load_original  # noqa: E402
from pipeline import cleaner_by_name, visit_maps  # noqa: E402

CLEANER = "g2"
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


def drawing_circles(data_root: Path, image: str) -> np.ndarray:
    with Image.open(data_root / image) as handle:
        gray = np.asarray(handle.convert("L"))
    return detect_circles(gray)


def suppressed(score: np.ndarray, circles: np.ndarray, box) -> np.ndarray:
    if len(circles) == 0 or query_shows_circles(circles, box):
        return score
    result = score.copy()
    result[circle_mask(score.shape, circles)] = 0.0
    return result


def predict_drawing(requests: list[dict], stage1, stage2, data_root: Path, emit) -> None:
    circles = drawing_circles(data_root, requests[0]["image"])

    def visit(request: dict, gated: np.ndarray) -> None:
        emit(request, suppressed(gated, circles, request["query_box"]))

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
