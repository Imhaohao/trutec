"""Per-drawing structure banks. One drawing is cached so a restart resumes."""
from __future__ import annotations

import hashlib
import time

import limits  # noqa: F401

import numpy as np
from PIL import Image

from constants import DATA_ROOT, GRID_STRIDE
from limits import BANK_CACHE, atomic_savez, current_gb, log, require
from structure_features import FEATURE_VERSION, bank_from_rgb
from texture_learning import grid_centers

Image.MAX_IMAGE_PIXELS = None


def ensure_bank(image: str, rgb: np.ndarray) -> dict:
    path = bank_path(image)
    cached = _load_bank(path, image, rgb.shape)
    if cached is not None:
        return cached
    started = time.perf_counter()
    bank = bank_from_rgb(rgb)
    _check_grid(bank, rgb.shape)
    _save_bank(path, image, rgb.shape, bank)
    log(f"bank {image} seconds={time.perf_counter() - started:.1f} rss_gb={current_gb():.2f}")
    return bank


def load_rgb(image: str) -> np.ndarray:
    with Image.open(DATA_ROOT / image) as source:
        return np.asarray(source.convert("RGB"))


def bank_path(image: str):
    digest = hashlib.sha256(image.encode()).hexdigest()[:20]
    return BANK_CACHE / f"{digest}.npz"


def _load_bank(path, image: str, shape) -> dict | None:
    if not path.is_file():
        return None
    try:
        with np.load(path) as stored:
            if not _matches(stored, image, shape):
                return None
            return _as_dict(stored)
    except (OSError, ValueError, KeyError):
        return None


def _matches(stored, image: str, shape) -> bool:
    if int(stored["version"]) != FEATURE_VERSION:
        return False
    if str(stored["image"].item()) != image:
        return False
    return int(stored["height"]) == int(shape[0]) and int(stored["width"]) == int(shape[1])


def _as_dict(stored) -> dict:
    return {
        "energy": np.array(stored["energy"], dtype=np.float32),
        "period": np.array(stored["period"], dtype=np.float32),
        "periodicity": np.array(stored["periodicity"], dtype=np.float32),
        "mark_area": np.array(stored["mark_area"], dtype=np.float32),
        "mark_side": np.array(stored["mark_side"], dtype=np.float32),
    }


def _save_bank(path, image: str, shape, bank: dict) -> None:
    atomic_savez(
        path,
        energy=bank["energy"],
        period=bank["period"],
        periodicity=bank["periodicity"],
        mark_area=bank["mark_area"],
        mark_side=bank["mark_side"],
        image=np.array(image),
        height=np.int32(shape[0]),
        width=np.int32(shape[1]),
        version=np.int32(FEATURE_VERSION),
    )


def _check_grid(bank: dict, shape) -> None:
    centers_y, centers_x = grid_centers(int(shape[0]), int(shape[1]), GRID_STRIDE)
    require(bank["energy"].shape[:2] == (len(centers_y), len(centers_x)), "bank grid")
