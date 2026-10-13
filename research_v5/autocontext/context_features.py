"""Label-free auto-context features on the stride-8 lattice.

A cell sees the stage-2 score at itself, window statistics of that score and of
the stage-1 probability, distance to the query box, and one color-mismatch value.
Nothing in this module reads a label, a split file, or a reviewed mask.
"""
from __future__ import annotations

import limits  # noqa: F401

import cv2
import numpy as np

from constants import GRID_STRIDE, HISTOGRAM_BINS
from features import gather, integral_image
from limits import require

FEATURE_VERSION = 1
WINDOWS_PX = (24, 48, 96, 192, 384)
SOURCES = ("stage2", "stage1")
STATS = ("mean", "max", "p25", "p75", "p90", "frac_gt_0_3", "frac_gt_0_6")
QUANTILES = (0.25, 0.75, 0.90)


def feature_names() -> list[str]:
    names = ["stage2_score"]
    for source in SOURCES:
        for window in WINDOWS_PX:
            _require_window(window)
            for stat in STATS:
                names.append(f"{source}_w{window}_{stat}")
    names.append("signed_distance_over_256")
    names.append("color_mismatch_w32")
    require(len(names) == len(set(names)), "duplicate feature name")
    return names


def half_cells(window_px: int) -> int:
    """Chebyshev half-width in cells for a nominal square of `window_px` pixels.

    The stacker's window_sum uses this half-width. The effective side is
    (2 * half + 1) * 8 pixels, the odd cell count whose half-width is
    (window_px // 2) // 8.
    """
    return (int(window_px) // 2) // GRID_STRIDE


def effective_side(window_px: int) -> int:
    return (2 * half_cells(window_px) + 1) * GRID_STRIDE


def context_volume(
    stage2: np.ndarray,
    stage1: np.ndarray,
    distance: np.ndarray,
    mismatch: np.ndarray,
) -> np.ndarray:
    _same_shape(stage2, stage1, distance, mismatch)
    names = feature_names()
    volume = np.empty((stage2.shape[0], stage2.shape[1], len(names)), dtype=np.float32)
    index = {name: position for position, name in enumerate(names)}
    volume[:, :, index["stage2_score"]] = stage2
    _write_windows(volume, index, "stage2", stage2)
    _write_windows(volume, index, "stage1", stage1)
    volume[:, :, index["signed_distance_over_256"]] = distance
    volume[:, :, index["color_mismatch_w32"]] = mismatch
    return volume


def lookup(grid: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.asarray(gather(grid, y, x), dtype=np.float32)


def plane_window_stats(plane: np.ndarray, half: int) -> dict[str, np.ndarray]:
    return stats_from_pack(integral_pack(plane), half)


def window_max(plane: np.ndarray, half: int) -> np.ndarray:
    if half <= 0:
        return np.array(plane, dtype=np.float32, copy=True)
    kernel = np.ones((half * 2 + 1, half * 2 + 1), dtype=np.uint8)
    return cv2.dilate(plane, kernel)


def _write_windows(volume, index, source: str, plane: np.ndarray) -> None:
    pack = integral_pack(plane)
    for window in WINDOWS_PX:
        _write_one_window(volume, index, source, window, stats_from_pack(pack, half_cells(window)))


def _write_one_window(volume, index, source: str, window: int, stats: dict) -> None:
    for name, values in stats.items():
        volume[:, :, index[f"{source}_w{window}_{name}"]] = values


def integral_pack(plane: np.ndarray) -> dict:
    """One integral image per bin, reused across window sizes."""
    work = np.ascontiguousarray(plane, dtype=np.float32)
    bins = HISTOGRAM_BINS
    scaled = np.clip(work, 0.0, 1.0)
    bin_index = np.minimum((scaled * bins).astype(np.int32), bins - 1)
    return {
        "work": work,
        "ones": integral_image(np.ones(work.shape, dtype=np.float64)),
        "values": integral_image(work),
        "gt03": integral_image(work > 0.3),
        "gt06": integral_image(work > 0.6),
        "bins": [integral_image(bin_index == bin_id) for bin_id in range(bins)],
    }


def stats_from_pack(pack: dict, half: int) -> dict[str, np.ndarray]:
    total = sat_window(pack["ones"], half)
    denom = np.maximum(total, 1.0)
    p25, p75, p90 = _quantiles_from_bins(pack["bins"], half, total)
    return {
        "mean": (sat_window(pack["values"], half) / denom).astype(np.float32),
        "max": window_max(pack["work"], half),
        "p25": p25,
        "p75": p75,
        "p90": p90,
        "frac_gt_0_3": (sat_window(pack["gt03"], half) / denom).astype(np.float32),
        "frac_gt_0_6": (sat_window(pack["gt06"], half) / denom).astype(np.float32),
    }


def sat_window(sat: np.ndarray, half: int) -> np.ndarray:
    height = int(sat.shape[0]) - 1
    width = int(sat.shape[1]) - 1
    y0, y1 = _axis_bounds(height, half)
    x0, x1 = _axis_bounds(width, half)
    return sat[y1][:, x1] - sat[y0][:, x1] - sat[y1][:, x0] + sat[y0][:, x0]


def _quantiles_from_bins(bin_sats, half: int, total: np.ndarray):
    bins = len(bin_sats)
    cdf = np.zeros(total.shape, dtype=np.float64)
    outputs = [np.zeros(total.shape, dtype=np.float32) for _ in range(3)]
    seen = [np.zeros(total.shape, dtype=bool) for _ in range(3)]
    targets = [level * total for level in QUANTILES]
    for bin_id, sat in enumerate(bin_sats):
        cdf += sat_window(sat, half)
        _mark_all(outputs, seen, cdf, targets, total, np.float32((bin_id + 0.5) / bins))
    return outputs[0], outputs[1], outputs[2]


def _axis_bounds(length: int, half: int):
    index = np.arange(length)
    start = np.clip(index - half, 0, length)
    stop = np.clip(index + half + 1, 0, length)
    return start, stop


def _mark_all(outputs, seen, cdf, targets, total, value) -> None:
    for out, done, target in zip(outputs, seen, targets):
        _mark_one(out, done, cdf, target, total, value)


def _mark_one(out, seen, cdf, target, total, value) -> None:
    hit = (~seen) & (total > 0) & (cdf >= target)
    out[hit] = value
    seen[hit] = True


def _same_shape(stage2, stage1, distance, mismatch) -> None:
    require(stage2.shape == stage1.shape, "stage maps differ")
    require(stage2.shape == distance.shape, "distance shape")
    require(stage2.shape == mismatch.shape, "mismatch shape")


def _require_window(window: int) -> None:
    require(half_cells(window) > 0, f"window {window} collapses on the stride-8 grid")
