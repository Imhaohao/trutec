"""Bootstrap and environment configuration for v7 false positive diagnosis.

Pins thread limits, configures search paths, and provides memory guards.
"""
from __future__ import annotations

import os
import resource
import subprocess
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
ROOT = RESEARCH.parent

STACKER = RESEARCH / "stacker"
STRUCTURE = RESEARCH / "structure"
QUERYCROP = RESEARCH / "querycrop"
QUERYCROP2 = RESEARCH / "querycrop2"
FACES = RESEARCH / "faces"
COMBO = RESEARCH / "combo"

DATA_ROOT = ROOT / "dataset"
V7_DEV_CACHE = COMBO / "cache" / "dev" / "g2"
STRUCTURE_MODEL = STRUCTURE / "cache" / "models" / "hardneg_0.75.joblib"
FIGURES_DIR = HERE / "figures"
FINDINGS_PATH = HERE / "findings.txt"
SUMMARY_JSON = HERE / "diagnose_summary.json"

MEMORY_LIMIT_GB = 9.5
_ORDER = (ROOT, RESEARCH, HERE, COMBO, STRUCTURE, FACES, QUERYCROP2, QUERYCROP, STACKER)


def pin_paths() -> None:
    for entry in _ORDER:
        text = str(entry)
        if text in sys.path:
            sys.path.remove(text)
        sys.path.insert(0, text)


pin_paths()

import cv2  # noqa: E402
import threadpoolctl  # noqa: E402

cv2.setNumThreads(4)
_THREAD_LIMIT = threadpoolctl.threadpool_limits(limits=4)
_THREAD_LIMIT.__enter__()


def log(message: str) -> None:
    print(message, flush=True)


def current_gb() -> float:
    try:
        raw = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())], text=True)
        return int(raw.strip()) / 1e6
    except Exception:
        return 0.0


def peak_gb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / 1e9
    return rss / 1e6


def guard_memory() -> None:
    import gc

    gc.collect()
    used = current_gb()
    if used >= MEMORY_LIMIT_GB:
        raise RuntimeError(f"rss {used:.2f} GB exceeds limit {MEMORY_LIMIT_GB:.1f} GB")
