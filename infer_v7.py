"""Frozen v7 entry point: cleaned crop descriptor, structure stage 2, color gate.

Reads a label-free request manifest and writes one binary mask per request.
The cleaner is guarded rule g2 and the score threshold is 0.25. Model files
come from --model-dir (default: models/v7 next to this script).
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

import constants
import evaluate_dev
from bootstrap import log, peak_gb
from inference_v7 import load_requests, load_stage2, predict_drawing, write_mask

CLEANER = "g2"
THRESHOLD = 0.25
STAGE1_NAME = "texture-hgb-crop-v1.joblib"
STAGE2_NAME = "stage2-structure-hgb-v7.joblib"
MODEL_FILES = (
    STAGE1_NAME,
    "texture-hgb-crop-v1.json",
    STAGE2_NAME,
    "stage2-structure-hgb-v7.json",
)
DEFAULT_MODEL_DIR = ROOT / "models" / "v7"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path, help="Label-free request manifest")
    parser.add_argument("--data-root", required=True, type=Path, help="Root for relative drawing paths")
    parser.add_argument("--output-dir", required=True, type=Path, help="Destination of <id>.png predictions")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="Directory of frozen v7 weights")
    return parser.parse_args(argv)


def model_paths(model_dir: Path) -> tuple[Path, Path]:
    directory = Path(model_dir).resolve()
    _require_model_files(directory)
    stage1 = directory / STAGE1_NAME
    _bind_stage1(stage1)
    return stage1, directory / STAGE2_NAME


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _stage1_path, stage2_path = model_paths(args.model_dir)
    stage1 = _load_stage1()
    stage2 = load_stage2(stage2_path)
    runtime = _runtime_header(args, stage2_path)
    for image, requests in _group(load_requests(args.inputs)).items():
        _one_drawing(image, requests, stage1, stage2, args, runtime)
    runtime["elapsed_seconds"] = time.perf_counter() - started
    runtime["peak_gb"] = peak_gb()
    _write_runtime(args.output_dir, runtime)


def _require_model_files(directory: Path) -> None:
    missing = [name for name in MODEL_FILES if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError("missing model files in " + str(directory) + ": " + ", ".join(missing))


def _bind_stage1(stage1: Path) -> None:
    if Path(constants.__file__).resolve().parent.name != "stacker":
        raise RuntimeError("stage-1 path constant is not stacker.constants")
    constants.MODEL_PATH = stage1
    evaluate_dev.MODEL_PATH = stage1


def _load_stage1():
    model, _metadata = evaluate_dev.load_original()
    return model


def _group(requests: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for request in requests:
        grouped[request["image"]].append(request)
    return grouped


def _runtime_header(args: argparse.Namespace, stage2_path: Path) -> dict:
    return {
        "cleaner": CLEANER,
        "threshold": THRESHOLD,
        "model_dir": str(Path(args.model_dir).resolve()),
        "stage1": str(evaluate_dev.MODEL_PATH),
        "stage2": str(stage2_path),
        "drawings": [],
    }


def _one_drawing(image: str, requests: list[dict], stage1, stage2, args: argparse.Namespace, runtime: dict) -> None:
    started = time.perf_counter()
    predict_drawing(requests, stage1, stage2, CLEANER, args.data_root, _emitter(args.output_dir))
    runtime["drawings"].append(
        {"image": image, "queries": len(requests), "seconds": time.perf_counter() - started}
    )
    log(f"wrote {len(requests)} masks for {image}")


def _emitter(output_dir: Path):
    def emit(request: dict, gated) -> None:
        write_mask(output_dir / f"{request['id']}.png", gated >= THRESHOLD)

    return emit


def _write_runtime(output_dir: Path, runtime: dict) -> None:
    (output_dir / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
