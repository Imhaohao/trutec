"""Thread caps and paths. Import this before numpy."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESEARCH = HERE.parent
STACKER = RESEARCH / "stacker"
CACHE = HERE / "cache"
BANK_CACHE = CACHE / "banks"
CELL_CACHE = CACHE / "cells"
MODEL_CACHE = CACHE / "models"
DEV_RAW = CACHE / "dev_raw"
DEV_GATED = CACHE / "dev_gated"
CV_PATH = HERE / "cv_results.json"
SETTINGS_PATH = HERE / "settings.json"
FINDINGS_PATH = HERE / "findings.txt"

_MEMORY_GB = 9.5


def _install_paths() -> None:
    for entry in (str(HERE), str(STACKER), str(RESEARCH), str(ROOT)):
        if entry in sys.path:
            sys.path.remove(entry)
        sys.path.append(entry)


_install_paths()

import cv2  # noqa: E402
import numpy as np  # noqa: E402

cv2.setNumThreads(4)

import bootstrap  # noqa: E402
import threadpoolctl  # noqa: E402

_THREAD_LIMIT = threadpoolctl.threadpool_limits(limits=4)
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


def read_json(path: Path):
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def current_gb() -> float:
    import subprocess

    raw = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())], text=True)
    return int(raw.strip()) / 1e6


def peak_gb() -> float:
    import resource

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / 1e9
    return rss / 1e6


def guard_memory() -> None:
    import gc

    gc.collect()
    require(current_gb() < _MEMORY_GB, f"rss {current_gb():.2f} GB exceeds {_MEMORY_GB} GB")


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
