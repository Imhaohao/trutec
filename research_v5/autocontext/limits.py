"""Thread caps and paths. Import this before numpy consumers."""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "6"
os.environ["OPENBLAS_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["NUMEXPR_NUM_THREADS"] = "6"
os.environ["VECLIB_MAXIMUM_THREADS"] = "6"

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESEARCH = HERE.parent
STACKER = RESEARCH / "stacker"
STACKER_CACHE = STACKER / "cache"
FOLDS_PATH = STACKER / "folds.json"
STACKER_CV_PATH = STACKER / "cv_results.json"
FROZEN_STAGE2 = STACKER_CACHE / "stage2_model.joblib"
V6_COLOR_REPORT = STACKER / "dev_report_color.json"
CACHE = HERE / "cache"
CELL_CACHE = CACHE / "cells"
STAGE2_FOLD_CACHE = CACHE / "stage2_folds"
DEV_RAW = CACHE / "dev_raw"
DEV_GATED = CACHE / "dev_gated"
IMAGE_LOG = CACHE / "image_times.jsonl"
STAGE3_MODEL = CACHE / "stage3_model.joblib"
CV_RESULTS = HERE / "cv_results.json"
SETTINGS_PATH = HERE / "settings.json"
PROGRESS_PATH = CACHE / "progress.json"

_THREAD_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _install_paths() -> None:
    for entry in (str(ROOT), str(RESEARCH), str(HERE), str(STACKER)):
        if entry not in sys.path:
            sys.path.insert(0, entry)


def _cap_threads() -> None:
    for key in _THREAD_KEYS:
        os.environ[key] = "6"


_install_paths()

import cv2  # noqa: E402
import numpy as np  # noqa: E402

cv2.setNumThreads(6)

import bootstrap  # noqa: E402

_cap_threads()

import threadpoolctl  # noqa: E402

_THREAD_LIMIT = threadpoolctl.threadpool_limits(limits=6)
_THREAD_LIMIT.__enter__()

from bootstrap import atomic_save_npy, atomic_savez  # noqa: E402


def log(message: str) -> None:
    print(message, flush=True)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_ready(payload)) + "\n")


def read_json(path: Path):
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def peak_gb() -> float:
    import resource

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / 1e9
    return rss / 1e6


def current_gb() -> float:
    import subprocess

    raw = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())], text=True)
    return int(raw.strip()) / 1e6


def dump_joblib(path: Path, payload) -> None:
    import joblib

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        joblib.dump(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.is_file():
            temporary.unlink()


def free_gb() -> float:
    return shutil.disk_usage(HERE).free / 1e9


def guard_memory() -> None:
    import gc

    if current_gb() >= 12.0:
        gc.collect()
    require(peak_gb() < 14.0, f"peak rss {peak_gb():.2f} GB exceeds 14 GB")


def thread_report() -> list[dict]:
    return threadpoolctl.threadpool_info()


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value
