"""Native v11 inference: g2-cleaned descriptor, all-data stage 2, then the v6 color gate.

Same request interface as research_v5/combo/inference_v7.py. The stage-2 file
defaults to the all-data hardneg_0.75 model. The w16 linear color gate at
strength 0.6 is the one inside combo/pipeline.py.
"""
from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "combo"))

import bootstrap  # noqa: E402,F401

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from collections import defaultdict  # noqa: E402

from bootstrap import log  # noqa: E402
from evaluate_dev import load_original  # noqa: E402
from inference_v7 import load_requests, load_stage2, write_mask  # noqa: E402
from pipeline import cleaner_by_name, visit_maps  # noqa: E402

BAND_NOTE = "stride-8 cell lookup, w16 linear color gate at 0.6"
DEFAULT_STAGE2 = Path(__file__).resolve().parent / "stage2_alldata_hardneg_0.75.joblib"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path, help="Label-free request manifest")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--threshold", required=True, type=float)
    parser.add_argument("--cleaner", required=True, help="identity, v6, rule_d, g1, g2, or g3")
    parser.add_argument("--stage2", type=Path, default=DEFAULT_STAGE2)
    return parser.parse_args()


def predict_drawing(requests: list[dict], stage1, stage2, cleaner_name: str, data_root: Path, emit) -> None:
    """Call emit(request, gated_native_score) once per request on this drawing."""
    visit_maps(requests, stage1, stage2, cleaner_by_name(cleaner_name), data_root, emit)


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

    def emit(request: dict, gated) -> None:
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
