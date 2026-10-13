"""Thread caps and import order. Import this before numpy."""
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
STACKER = RESEARCH / "stacker"
STRUCTURE = RESEARCH / "structure"
QUERYCROP2 = RESEARCH / "querycrop2"
QUERYCROP = RESEARCH / "querycrop"
FACES = RESEARCH / "faces"
CACHE = HERE / "cache"
STACKER_CACHE = CACHE / "stacker"
CELL_CACHE = CACHE / "cells"
BANK_CACHE = CACHE / "banks"
MODEL_CACHE = CACHE / "models"
DEV_CACHE = CACHE / "dev"
CV_PATH = HERE / "cv_results.json"
DEV_REPORT = HERE / "dev_report.json"
SETTINGS_PATH = HERE / "settings.json"
SUMMARY_PATH = CACHE / "cell_summary.json"
FINDINGS_PATH = HERE / "findings.txt"
FOLDS_PATH = STACKER / "folds.json"
STRUCTURE_CELLS = STRUCTURE / "cache" / "cells"
V7_REPORT = COMBO / "dev_report_g2.json"
FIT_REAL = ROOT / "artifacts" / "splits" / "fit-real.json"
CELL_VERSION = 1
STACKER_VERSION = 1
MEMORY_LIMIT_GB = 9.5
CLEANER = "g2"

_ACTIVE = False


def activate() -> None:
    """Put frozen modules on the path and import them once, in an order that resolves."""
    global _ACTIVE
    if _ACTIVE:
        hold()
        return
    _load_combo_bootstrap()
    _import_frozen()
    hold()
    _ACTIVE = True


def hold() -> None:
    here = str(HERE)
    if here in sys.path:
        sys.path.remove(here)
    sys.path.insert(0, here)


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
    if used >= MEMORY_LIMIT_GB:
        raise RuntimeError(f"rss {used:.2f} GB exceeds {MEMORY_LIMIT_GB:.1f} GB")


def _load_combo_bootstrap() -> None:
    combo = str(COMBO)
    if combo in sys.path:
        sys.path.remove(combo)
    sys.path.insert(0, combo)
    import bootstrap

    bootstrap.pin_paths()


def _import_frozen() -> None:
    import drawing  # noqa: F401
    import dataset  # noqa: F401
    import folds  # noqa: F401
    import stage1  # noqa: F401
    import model  # noqa: F401
    import guards  # noqa: F401
    import structure_features  # noqa: F401
    import banks  # noqa: F401


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
