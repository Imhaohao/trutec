"""CPU tests for guarded cleaners. No drawings and no labels."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bootstrap  # noqa: E402,F401

import numpy as np

from bootstrap import CHOSEN_PATH
from cleaners import clean_crop_components
from guards import GUARDED, PAPER_LEVEL, crop_for

def test_crosshatch_keeps_ink() -> None:
    crop = _crosshatch()
    _require_rule_c_drops(crop)
    for name in _names():
        assert _kept(crop, crop_for(name, crop)) >= 0.80


def test_parallel_lines_keep_ink() -> None:
    crop = _parallel_lines()
    _require_rule_c_drops(crop)
    for name in _names():
        assert _kept(crop, crop_for(name, crop)) >= 0.80


def test_glyphs_are_removed() -> None:
    crop = _hatch_with_glyph()
    for name in _names():
        cleaned = crop_for(name, crop)
        if not np.all(cleaned[10:26, 8:24] >= PAPER_LEVEL):
            raise AssertionError(f"{name} kept the glyph")
        if not np.any(cleaned[8:11, 40:43] < PAPER_LEVEL):
            raise AssertionError(f"{name} deleted the hatch dots")


def run_all() -> None:
    test_crosshatch_keeps_ink()
    test_parallel_lines_keep_ink()
    test_glyphs_are_removed()


def _names() -> tuple[str, ...]:
    if not CHOSEN_PATH.is_file():
        return GUARDED
    chosen = json.loads(CHOSEN_PATH.read_text(encoding="utf-8"))["variant"]
    if chosen not in GUARDED:
        raise AssertionError(f"selected variant {chosen} is not a guarded cleaner")
    return (chosen,)


def _require_rule_c_drops(crop: np.ndarray) -> None:
    if _kept(crop, clean_crop_components(crop, paper_level=PAPER_LEVEL)) >= 0.80:
        raise AssertionError("rule c did not drop the pattern, so the guard is untested")


def _kept(original: np.ndarray, cleaned: np.ndarray) -> float:
    before = int(np.count_nonzero(original < PAPER_LEVEL))
    after = int(np.count_nonzero(cleaned < PAPER_LEVEL))
    return after / before


def _crosshatch() -> np.ndarray:
    crop = np.full((40, 40), 255, dtype=np.uint8)
    crop[::5, :] = 0
    crop[:, ::5] = 0
    for y, x in (
        (2, 2), (2, 7), (2, 12), (2, 17), (2, 22), (2, 27),
        (7, 2), (7, 7), (7, 12), (12, 2), (17, 2), (22, 2),
    ):
        crop[y, x] = 0
    return crop


def _parallel_lines() -> np.ndarray:
    crop = np.full((80, 80), 255, dtype=np.uint8)
    for y in (8, 16, 24, 32, 40, 48, 56):
        crop[y, 4:76] = 0
    for y in (12, 20, 28, 36):
        for x in (10, 30, 50):
            crop[y : y + 3, x : x + 3] = 0
    return crop


def _hatch_with_glyph() -> np.ndarray:
    crop = np.full((80, 80), 255, dtype=np.uint8)
    for y in range(8, 72, 8):
        for x in range(40, 76, 8):
            crop[y : y + 3, x : x + 3] = 0
    crop[10:26, 8:24] = 0
    return crop


if __name__ == "__main__":
    run_all()
    print("ok")
