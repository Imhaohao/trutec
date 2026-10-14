"""Native v7 inference: cleaned descriptor, structure stage 2, then the v6 color gate.

Every pixel reads the stage-2 probability of its stride-8 cell, is multiplied by
the w16 linear color gate (strength 0.6), and is selected when the gated score
reaches the given threshold. Requests carry no labels.
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

from bootstrap import STRUCTURE_MODEL, log
from evaluate_dev import load_original
from pipeline import cleaner_by_name, visit_maps

BAND_NOTE = "stride-8 cell lookup, w16 linear color gate at 0.6"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path, help="Label-free request manifest")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--threshold", required=True, type=float)
    parser.add_argument("--cleaner", required=True, help="identity, v6, rule_d, g1, g2, or g3")
    parser.add_argument("--stage2", type=Path, default=STRUCTURE_MODEL)
    return parser.parse_args()


def load_requests(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["examples"] if isinstance(payload, dict) else payload
    allowed = ("id", "image", "width", "height", "query_box", "context_boxes")
    return [{key: row[key] for key in allowed if key in row} for row in rows]


def load_stage2(path: Path):
    stored = joblib.load(path)
    if isinstance(stored, dict):
        return stored["model"]
    return stored


def predict_drawing(requests: list[dict], stage1, stage2, cleaner_name: str, data_root: Path, emit) -> None:
    """Call emit(request, gated_native_score) once per request on this drawing."""
    visit_maps(requests, stage1, stage2, cleaner_by_name(cleaner_name), data_root, emit)


def write_mask(path: Path, selected: np.ndarray) -> None:
    Image.fromarray(np.where(selected, 255, 0).astype(np.uint8), mode="L").save(path)


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage1, _metadata = load_original()
    stage2 = load_stage2(args.stage2)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for request in load_requests(args.inputs):
        grouped[request["image"]].append(request)
    runtime = _runtime_header(args)

    def emit(request: dict, gated: np.ndarray) -> None:
        write_mask(args.output_dir / f"{request['id']}.png", gated >= args.threshold)

    for image, requests in grouped.items():
        _one_drawing(image, requests, stage1, stage2, args, emit, runtime)
    runtime["elapsed_seconds"] = time.perf_counter() - started
    (args.output_dir / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")


def _runtime_header(args) -> dict:
    return {
        "threshold": args.threshold,
        "cleaner": args.cleaner,
        "stage2": str(args.stage2),
        "gate": BAND_NOTE,
        "drawings": [],
    }


def _one_drawing(image, requests, stage1, stage2, args, emit, runtime) -> None:
    started = time.perf_counter()
    predict_drawing(requests, stage1, stage2, args.cleaner, args.data_root, emit)
    runtime["drawings"].append(
        {"image": image, "queries": len(requests), "seconds": time.perf_counter() - started}
    )
    log(f"wrote {len(requests)} masks for {image}")


if __name__ == "__main__":
    main()
