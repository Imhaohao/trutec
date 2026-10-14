"""Thread caps, paths, and cache writes. Import before numpy."""
from __future__ import annotations

import json
import os
import subprocess
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
COMBO = RESEARCH / "combo"
RETRAIN = RESEARCH / "retrain7"
STACKER = RESEARCH / "stacker"
CACHE = HERE / "cache"
WINDOW_CACHE = CACHE / "windows"
CELL_CACHE = CACHE / "cells"
MODEL_CACHE = CACHE / "models"
DEV_CACHE = CACHE / "dev"
CV_PATH = HERE / "cv_results.json"
CONTROL_PATH = HERE / "cv_control.json"
DEV_REPORT = HERE / "dev_report.json"
SETTINGS_PATH = HERE / "settings.json"
SUMMARY_PATH = CACHE / "cell_summary.json"
FINDINGS_PATH = HERE / "findings.txt"
V7_REPORT = COMBO / "dev_report_g2.json"
V7_DEV_CACHE = COMBO / "cache" / "dev" / "g2"
RETRAIN_CV = RETRAIN / "cv_results.json"
RETRAIN_CELLS = RETRAIN / "cache" / "cells"
FOLDS_PATH = STACKER / "folds.json"
FIT_REAL = ROOT / "artifacts" / "splits" / "fit-real.json"
TARGET_DOCUMENT = "doc-cfd973b494d531ad"
CELL_VERSION = 1
WINDOW_VERSION = 1
SCORE_VERSION = 1
BASE_WIDTH = 144
MEMORY_LIMIT_GB = 9.6
CLEANER = "g2"
BUDGET_SECONDS = 110 * 60

_ACTIVE = False


def activate() -> None:
    """Pin frozen import paths, then keep this directory first."""
    global _ACTIVE
    if _ACTIVE:
        return
    combo = str(COMBO)
    if combo not in sys.path:
        sys.path.insert(0, combo)
    import bootstrap

    bootstrap.pin_paths()
    _front(str(RETRAIN))
    _front(str(HERE))
    _ACTIVE = True


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


def atomic_savez(path: Path, **arrays) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with temporary.open("wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.is_file():
            temporary.unlink()


def current_gb() -> float:
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
    used = current_gb()
    require(used < MEMORY_LIMIT_GB, f"rss {used:.2f} GB exceeds {MEMORY_LIMIT_GB:.1f} GB")


def safe_id(query_id: str) -> str:
    require(bool(query_id) and "/" not in query_id and "\\" not in query_id, "unsafe id")
    require(not query_id.startswith("."), "unsafe id")
    return query_id


def _front(entry: str) -> None:
    if entry in sys.path:
        sys.path.remove(entry)
    sys.path.insert(0, entry)


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return _json_scalar(value)


def _json_scalar(value):
    import numpy as np

    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value
