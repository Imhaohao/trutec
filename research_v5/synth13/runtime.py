"""Thread caps and import order. Import this before numpy."""
from __future__ import annotations

import hashlib
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
STRUCTURE = RESEARCH / "structure"
ALLDATA12 = RESEARCH / "alldata12"
CACHE = HERE / "cache"
SAMPLE_DIR = CACHE / "stage1"
FOLD_DIR = CACHE / "folds"
CELL_FIT = CACHE / "cells" / "fit"
CELL_DEV = CACHE / "cells" / "development"
CELL_SYN = CACHE / "cells" / "synthetic"
BANK_DIR = CACHE / "banks"
FITS = ROOT / "artifacts" / "splits" / "fit.json"
DEVELOPMENT = ROOT / "artifacts" / "splits" / "development-real.json"
SYNTHETIC_ROOT = RESEARCH / "synthetic" / "data"
SYNTHETIC_MANIFEST = SYNTHETIC_ROOT / "manifest.json"
DATASET_ROOT = ROOT / "dataset"
ALLDATA12_FOLDS = ALLDATA12 / "folds.json"
ALLDATA12_SAMPLES = ALLDATA12 / "cache" / "stage1"
STACKER_SAMPLES = STACKER / "cache" / "stage1"
STRUCTURE_CELLS = STRUCTURE / "cache" / "cells"
FROZEN_MANIFEST = ROOT.parent / "hatch-matching-challenge" / "artifacts" / "frozen-pipeline-v12.json"
ORIGINAL_STAGE1 = ROOT / "models" / "models" / "texture-hgb-crop-v1.joblib"
ORIGINAL_STAGE2 = ALLDATA12 / "stage2_alldata_hardneg_0.75.joblib"
FROZEN_STAGE1 = ALLDATA12 / "stage1_alldata.joblib"
STAGE1_PATH = HERE / "stage1_alldata.joblib"
STAGE2_PATH = HERE / "stage2_alldata_hardneg_0.75.joblib"
FOLDS_PATH = HERE / "folds.json"
RSS_LIMIT_GB = 11.5


def pin() -> None:
    """Keep stacker ahead of this directory so `import bootstrap` stays the stacker's."""
    for entry in (str(ROOT), str(RESEARCH), str(STRUCTURE), str(STACKER)):
        if entry in sys.path:
            sys.path.remove(entry)
        sys.path.insert(0, entry)


pin()

import bootstrap  # noqa: E402,F401


def log(message: str) -> None:
    print(message, flush=True)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload) -> None:
    inside(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


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


def inside(path: Path) -> None:
    require(path.resolve().is_relative_to(HERE.resolve()), f"refusing write outside synth13: {path}")
