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
STRUCTURE = RESEARCH / "structure"
QUERYCROP = RESEARCH / "querycrop"
QUERYCROP2 = RESEARCH / "querycrop2"
FACES = RESEARCH / "faces"
CACHE = HERE / "cache"
DEV_CACHE = CACHE / "dev"
FINDINGS_PATH = HERE / "findings.txt"
QUERYCROP2_FINDINGS = QUERYCROP2 / "findings.txt"
STRUCTURE_MODEL = STRUCTURE / "cache" / "models" / "hardneg_0.75.joblib"
V6_STAGE2 = STACKER / "cache" / "stage2_model.joblib"
STRUCTURE_GATED = STRUCTURE / "cache" / "dev_gated" / "hardneg_0.75"
V6_REPORT = STACKER / "dev_report_color.json"
STRUCTURE_REPORT = STRUCTURE / "dev_hardneg_0.75.json"
MEMORY_LIMIT_GB = 9.5
_ORDER = (ROOT, RESEARCH, STRUCTURE, FACES, QUERYCROP2, QUERYCROP, STACKER, HERE)


def pin_paths() -> None:
    for entry in _ORDER:
        text = str(entry)
        if text in sys.path:
            sys.path.remove(text)
        sys.path.insert(0, text)


pin_paths()

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

import cv2  # noqa: E402
import threadpoolctl  # noqa: E402

cv2.setNumThreads(4)
_THREAD_LIMIT = threadpoolctl.threadpool_limits(limits=4)
_THREAD_LIMIT.__enter__()


def log(message: str) -> None:
    print(message, flush=True)


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
    used = current_gb()
    log(f"memory current_gb={used:.2f} peak_gb={peak_gb():.2f}")
    if used >= MEMORY_LIMIT_GB:
        raise RuntimeError(f"rss {used:.2f} GB exceeds {MEMORY_LIMIT_GB:.1f} GB")


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


def atomic_save_npy(path: Path, values) -> None:
    import numpy as np

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
