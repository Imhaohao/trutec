"""Thread caps, import paths, and atomic cache writes. Import before numpy."""
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
QUERYCROP = RESEARCH / "querycrop"
FACES = RESEARCH / "faces"
CACHE = HERE / "cache"
FIT_CACHE = CACHE / "fit"
DEV_CACHE = CACHE / "dev"
FOLDS_PATH = STACKER / "folds.json"
STAGE1_FOLD_DIR = STACKER / "cache" / "folds"
STAGE2_FOLD_DIR = RESEARCH / "autocontext" / "cache" / "stage2_folds"
FROZEN_STAGE2 = STACKER / "cache" / "stage2_model.joblib"
V6_DEV_SCORES = STACKER / "cache" / "dev_scores"
FIT_MANIFEST = ROOT / "artifacts" / "splits" / "fit.json"
FIT_REAL_MANIFEST = ROOT / "artifacts" / "splits" / "fit-real.json"
FINDINGS_PATH = HERE / "findings.txt"
CHOSEN_PATH = HERE / "chosen.json"
IMAGE_LOG = CACHE / "images.jsonl"
MEMORY_LIMIT_GB = 10.0

for _entry in (str(ROOT), str(RESEARCH), str(FACES), str(STACKER), str(QUERYCROP), str(HERE)):
    if _entry in sys.path:
        sys.path.remove(_entry)
    sys.path.insert(0, _entry)

import evaluate  # noqa: E402


def _logical_data_path(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise evaluate.EvaluationError("Dataset paths must be nonempty relative paths")
    parts = Path(relative).parts
    if ".." in parts:
        raise evaluate.EvaluationError("Dataset path escapes --data-root: " + str(relative))
    base = Path(root)
    path = base.joinpath(*parts)
    if not path.is_relative_to(base):
        raise evaluate.EvaluationError("Dataset path escapes --data-root: " + str(relative))
    return path


evaluate.data_path = _logical_data_path


def log(message: str) -> None:
    print(message, flush=True)


def peak_gb() -> float:
    import resource

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / 1e9
    return rss / 1e6


def guard_memory() -> None:
    import gc

    gc.collect()
    used = peak_gb()
    log(f"memory peak_gb={used:.2f}")
    if used >= MEMORY_LIMIT_GB:
        raise RuntimeError(f"peak rss {used:.2f} GB exceeds {MEMORY_LIMIT_GB:.0f} GB")


def atomic_savez(path, **arrays) -> None:
    import numpy as np

    path = Path(path)
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


def atomic_save_npy(path, values) -> None:
    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, values)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.is_file():
            temporary.unlink()


def _cap_libraries() -> None:
    import cv2
    import threadpoolctl

    cv2.setNumThreads(4)
    global _THREAD_LIMIT
    _THREAD_LIMIT = threadpoolctl.threadpool_limits(limits=4)
    _THREAD_LIMIT.__enter__()


_THREAD_LIMIT = None
_cap_libraries()
