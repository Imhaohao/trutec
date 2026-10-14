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
FIT_CACHE = CACHE / "fit"
DEV_CACHE = CACHE / "dev"
MODEL_CACHE = CACHE / "models"
FOLDS_PATH = STACKER / "folds.json"
STAGE2_FOLD_DIR = RESEARCH / "autocontext" / "cache" / "stage2_folds"
FROZEN_STAGE2 = STACKER / "cache" / "stage2_model.joblib"
STAGE1_FOLD_DIR = STACKER / "cache" / "folds"
PUBLISHED_V6 = STACKER / "dev_report_color.json"
SETTINGS_PATH = HERE / "settings.json"
CHOSEN_PATH = HERE / "chosen.json"
DEV_REPORT_PATH = HERE / "dev_report.json"
FINDINGS_PATH = HERE / "findings.txt"
IMAGE_LOG = CACHE / "images.jsonl"
VERSION = 1
MEMORY_LIMIT_GB = 10.0

for _entry in (str(ROOT), str(RESEARCH), str(STACKER), str(HERE)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)


def log(message: str) -> None:
    print(message, flush=True)


def peak_gb() -> float:
    import resource

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / 1e9
    return rss / 1e6


def current_gb() -> float:
    import subprocess

    raw = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())], text=True)
    return int(raw.strip() or "0") / 1e6


def guard_memory() -> None:
    import gc

    gc.collect()
    used = peak_gb()
    log(f"memory current_gb={current_gb():.2f} peak_gb={used:.2f}")
    if used >= MEMORY_LIMIT_GB:
        raise RuntimeError(f"peak rss {used:.2f} GB exceeds {MEMORY_LIMIT_GB:.0f} GB")


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")
    os.replace(temporary, path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_ready(payload)) + "\n")


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


def _json_ready(value):
    import numpy as np

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


def _cap_libraries() -> None:
    import cv2
    import threadpoolctl

    cv2.setNumThreads(4)
    global _THREAD_LIMIT
    _THREAD_LIMIT = threadpoolctl.threadpool_limits(limits=4)
    _THREAD_LIMIT.__enter__()


_THREAD_LIMIT = None
_cap_libraries()
