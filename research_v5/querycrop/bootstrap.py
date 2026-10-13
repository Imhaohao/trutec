"""Thread caps and path setup for query cleaning experiments."""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Thread caps per research brief
for var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[var] = "4"

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STACKER = ROOT / "research_v5" / "stacker"
FACES = ROOT / "research_v5" / "faces"
DATA_ROOT = ROOT / "dataset"
CACHE = HERE / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

for path in (str(HERE), str(ROOT / "research_v5"), str(STACKER), str(FACES), str(ROOT)):
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)

import evaluate  # noqa: E402


def _logical_data_path(root: str | Path, relative: str | Path) -> Path:
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



def log(msg: str) -> None:
    print(msg, flush=True)


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def atomic_save_npy(path: Path, values: object) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with temp.open("wb") as handle:
            np.save(handle, values)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if temp.is_file():
            temp.unlink()


def atomic_savez(path: Path, **arrays: object) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with temp.open("wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if temp.is_file():
            temp.unlink()

