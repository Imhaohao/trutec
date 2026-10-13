"""Thread caps and import order. Import this before numpy."""
from __future__ import annotations

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
STRUCTURE = RESEARCH / "structure"
CELL_DIR = HERE / "cache" / "cells"
BANK_DIR = HERE / "cache" / "banks"
MODEL_PATH = HERE / "stage2_alldata_hardneg_0.75.joblib"
FIT_CELL_DIR = STRUCTURE / "cache" / "cells"
DEVELOPMENT_MANIFEST = ROOT / "artifacts" / "splits" / "development-real.json"
FROZEN_MANIFEST = ROOT.parent / "hatch-matching-challenge" / "artifacts" / "frozen-pipeline-v7.json"
ORIGINAL_STAGE2 = STRUCTURE / "cache" / "models" / "hardneg_0.75.joblib"
RSS_LIMIT_GB = 11.5


def pin() -> None:
    """Keep stacker ahead of this directory so `import bootstrap` stays the stacker's."""
    for entry in (str(ROOT), str(RESEARCH), str(STRUCTURE), str(STACKER)):
        if entry in sys.path:
            sys.path.remove(entry)
        sys.path.insert(0, entry)


pin()


def log(message: str) -> None:
    print(message, flush=True)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


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


def guard_rss() -> None:
    import gc

    gc.collect()
    used = current_gb()
    log(f"rss_gb={used:.2f} peak_gb={peak_gb():.2f}")
    require(used < RSS_LIMIT_GB, f"rss {used:.2f} GB exceeds {RSS_LIMIT_GB} GB")
