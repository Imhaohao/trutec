"""Frozen v14 entry point: v12 all-data stage 1, CV-chosen stage-2 capacity, color gate.

Reads a label-free request manifest and writes one binary mask per request.
The cleaner is guarded rule g2 and the score threshold is 0.25. Model files
come from --model-dir (default: models/v14 next to this script).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_THREAD_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
for _key in _THREAD_KEYS:
    os.environ[_key] = "4"

ROOT = Path(__file__).resolve().parent
COMBO = ROOT / "research_v5" / "combo"
sys.path.insert(0, str(COMBO))

import bootstrap  # noqa: E402,F401

import argparse
import json
import time
from collections import defaultdict

import joblib
from bootstrap import log, peak_gb
from inference_v7 import load_requests, load_stage2, write_mask
from pipeline import cleaner_by_name, visit_maps

CLEANER = "g2"
THRESHOLD = 0.25
STAGE1_FEATURES = 184
STAGE1_NAME = "stage1-alldata-v12.joblib"
STAGE2_NAME = "stage2-structure-hgb-v14.joblib"
MODEL_FILES = (
    STAGE1_NAME,
    "stage1-alldata-v12.json",
    STAGE2_NAME,
    "stage2-structure-hgb-v14.json",
)
DEFAULT_MODEL_DIR = ROOT / "models" / "v14"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path, help="Label-free request manifest")
    parser.add_argument("--data-root", required=True, type=Path, help="Root for relative drawing paths")
    parser.add_argument("--output-dir", required=True, type=Path, help="Destination of <id>.png predictions")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="Directory of frozen v14 weights")
    return parser.parse_args(argv)


def model_paths(model_dir: Path) -> tuple[Path, Path]:
    directory = Path(model_dir).resolve()
    missing = [name for name in MODEL_FILES if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError("missing model files in " + str(directory) + ": " + ", ".join(missing))
    return directory / STAGE1_NAME, directory / STAGE2_NAME


def load_stage1(path: Path):
    stored = joblib.load(path)
    model = stored["model"] if isinstance(stored, dict) else stored
    if int(model.n_features_in_) != STAGE1_FEATURES:
        raise RuntimeError(f"stage1 features {int(model.n_features_in_)} != {STAGE1_FEATURES}")
    return model


def predict_drawing(requests: list[dict], stage1, stage2, cleaner: str, data_root: Path, emit) -> None:
    visit_maps(requests, stage1, stage2, cleaner_by_name(cleaner), data_root, emit)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage1_path, stage2_path = model_paths(args.model_dir)
    stage1 = load_stage1(stage1_path)
    stage2 = load_stage2(stage2_path)
    runtime = {
        "cleaner": CLEANER,
        "threshold": THRESHOLD,
        "model_dir": str(Path(args.model_dir).resolve()),
        "stage1": str(stage1_path),
        "stage2": str(stage2_path),
        "drawings": [],
    }
    for image, requests in _group(load_requests(args.inputs)).items():
        _one_drawing(image, requests, stage1, stage2, args, runtime)
    runtime["elapsed_seconds"] = time.perf_counter() - started
    runtime["peak_gb"] = peak_gb()
    (args.output_dir / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")


def _group(requests: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for request in requests:
        grouped[request["image"]].append(request)
    return grouped


def _one_drawing(image: str, requests: list[dict], stage1, stage2, args: argparse.Namespace, runtime: dict) -> None:
    started = time.perf_counter()
    predict_drawing(requests, stage1, stage2, CLEANER, args.data_root, _emitter(args.output_dir))
    runtime["drawings"].append({"image": image, "queries": len(requests), "seconds": time.perf_counter() - started})
    log(f"wrote {len(requests)} masks for {image}")


def _emitter(output_dir: Path):
    def emit(request: dict, gated) -> None:
        write_mask(output_dir / f"{request['id']}.png", gated >= THRESHOLD)

    return emit


if __name__ == "__main__":
    main()
