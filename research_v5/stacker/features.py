"""Window statistics on the stride-8 lattice and native ink fractions."""
from __future__ import annotations

import bootstrap  # noqa: F401  # sets thread caps before numpy

import cv2
import numpy as np

from constants import (
    COLOR_WINDOWS,
    DISTANCE_SCALE,
    GRID_STRIDE,
    HISTOGRAM_BINS,
    RADII,
    SCORE_STATS,
    require,
)
from color.color_features import compute_sample_mismatch
from texture_learning import grid_centers


def feature_names() -> list[str]:
    names = [
        "cell_score",
        "query_mean",
        "query_max",
        "query_p75",
        "query_p90",
        "query_frac_gt_0_5",
        "query_std",
    ]
    for radius in RADII:
        for stat in SCORE_STATS:
            names.append(f"r{radius}_{stat}")
        for stat in SCORE_STATS:
            names.append(f"r{radius}_{stat}_minus_query")
        names.append(f"r{radius}_frac_above_query_median")
    for radius in RADII:
        names.append(f"r{radius}_ink_fraction")
        names.append(f"r{radius}_chroma_fraction")
    for window in COLOR_WINDOWS:
        names.append(f"color_mismatch_w{window}")
    names.append("signed_distance_over_256")
    require(len(names) == len(set(names)), "duplicate feature name")
    return names


def integral_image(values: np.ndarray) -> np.ndarray:
    height, width = int(values.shape[0]), int(values.shape[1])
    work = np.array(values, dtype=np.float64, copy=True)
    np.cumsum(work, axis=0, out=work)
    np.cumsum(work, axis=1, out=work)
    sat = np.zeros((height + 1, width + 1), dtype=np.float64)
    sat[1:, 1:] = work
    return sat


def fraction_maps(
    mask: np.ndarray, centers_y: np.ndarray, centers_x: np.ndarray
) -> dict[int, np.ndarray]:
    sat = integral_image(mask)
    height, width = int(mask.shape[0]), int(mask.shape[1])
    return {
        radius: _fraction_from_sat(sat, centers_y, centers_x, height, width, radius)
        for radius in RADII
    }


def mismatch_grid(
    signature, maps, centers_y: np.ndarray, centers_x: np.ndarray
) -> np.ndarray:
    width = int(len(centers_x))
    yy = np.repeat(centers_y, width)
    xx = np.tile(centers_x, len(centers_y))
    flat = compute_sample_mismatch(signature, maps, yy, xx)
    return np.asarray(flat, dtype=np.float32).reshape(len(centers_y), width)


def assemble(
    prob: np.ndarray,
    query_box: tuple[int, int, int, int],
    height: int,
    width: int,
    ink_by_radius: dict[int, np.ndarray],
    chroma_by_radius: dict[int, np.ndarray],
    mismatch_by_window: dict[int, np.ndarray],
) -> np.ndarray:
    names = feature_names()
    volume = np.empty((prob.shape[0], prob.shape[1], len(names)), dtype=np.float32)
    index = {name: position for position, name in enumerate(names)}
    centers_y, centers_x = grid_centers(height, width, GRID_STRIDE)
    require(prob.shape == (len(centers_y), len(centers_x)), "probability grid shape")
    stats = query_distribution(prob, centers_y, centers_x, query_box)
    _fill_constants(volume, index, prob, stats)
    _fill_radii(volume, index, prob, stats, ink_by_radius, chroma_by_radius)
    for window in COLOR_WINDOWS:
        volume[:, :, index[f"color_mismatch_w{window}"]] = mismatch_by_window[window]
    volume[:, :, index["signed_distance_over_256"]] = signed_distance(
        centers_y, centers_x, query_box
    )
    require(bool(np.isfinite(volume).all()), "non-finite context features")
    return volume


def query_distribution(
    prob: np.ndarray,
    centers_y: np.ndarray,
    centers_x: np.ndarray,
    query_box: tuple[int, int, int, int],
) -> dict[str, float]:
    values = _query_values(prob, centers_y, centers_x, query_box)
    return {
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
        "frac_gt_0_5": float(np.mean(values > 0.5)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
    }


def window_statistics(
    prob: np.ndarray, half: int, query_median: float, bins: int = HISTOGRAM_BINS
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    plane = np.ascontiguousarray(prob, dtype=np.float32)
    total = window_sum(np.ones(plane.shape, dtype=np.float64), half)
    denom = np.maximum(total, 1.0)
    mean = (window_sum(plane, half) / denom).astype(np.float32)
    frac = (window_sum(plane > 0.5, half) / denom).astype(np.float32)
    above = (window_sum(plane > query_median, half) / denom).astype(np.float32)
    p75, p90 = _percentiles(plane, half, total, bins)
    return mean, _maximum(plane, half), p75, p90, frac, above


def window_sum(values: np.ndarray, half: int) -> np.ndarray:
    sat = integral_image(values)
    y0, y1 = _bounds(int(values.shape[0]), half)
    x0, x1 = _bounds(int(values.shape[1]), half)
    return _rect(sat, y0, y1, x0, x1)


def signed_distance(
    centers_y: np.ndarray, centers_x: np.ndarray, query_box: tuple[int, int, int, int]
) -> np.ndarray:
    x0, y0, x1, y1 = [int(value) for value in query_box]
    cy = centers_y.astype(np.float32)[:, None]
    cx = centers_x.astype(np.float32)[None, :]
    inside = (cy >= y0) & (cy < y1) & (cx >= x0) & (cx < x1)
    outside = np.hypot(_outside_gap(x0, x1, cx), _outside_gap(y0, y1, cy))
    inside_dist = np.minimum(_inside_gap(x0, x1, cx), _inside_gap(y0, y1, cy))
    signed = np.where(inside, -inside_dist, outside)
    return (signed / np.float32(DISTANCE_SCALE)).astype(np.float32)


def cell_index(y: np.ndarray, x: np.ndarray, grid_h: int, grid_w: int) -> tuple[np.ndarray, np.ndarray]:
    iy = np.minimum(np.asarray(y, dtype=np.int64) // GRID_STRIDE, grid_h - 1)
    ix = np.minimum(np.asarray(x, dtype=np.int64) // GRID_STRIDE, grid_w - 1)
    return iy, ix


def gather(grid: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    iy, ix = cell_index(y, x, int(grid.shape[0]), int(grid.shape[1]))
    return grid[iy, ix]


def _fill_constants(volume, index, prob, stats) -> None:
    volume[:, :, index["cell_score"]] = prob
    for name in ("mean", "max", "p75", "p90", "frac_gt_0_5", "std"):
        volume[:, :, index[f"query_{name}"]] = np.float32(stats[name])


def _fill_radii(volume, index, prob, stats, ink_by_radius, chroma_by_radius) -> None:
    for radius in RADII:
        _fill_one_radius(volume, index, prob, radius, stats, ink_by_radius, chroma_by_radius)


def _fill_one_radius(volume, index, prob, radius, stats, ink_by_radius, chroma_by_radius) -> None:
    half = radius // GRID_STRIDE
    mean, maximum, p75, p90, frac, above = window_statistics(prob, half, stats["median"])
    local = {"mean": mean, "max": maximum, "p75": p75, "p90": p90, "frac_gt_0_5": frac}
    for stat, values in local.items():
        volume[:, :, index[f"r{radius}_{stat}"]] = values
        volume[:, :, index[f"r{radius}_{stat}_minus_query"]] = values - np.float32(stats[stat])
    volume[:, :, index[f"r{radius}_frac_above_query_median"]] = above
    volume[:, :, index[f"r{radius}_ink_fraction"]] = ink_by_radius[radius]
    volume[:, :, index[f"r{radius}_chroma_fraction"]] = chroma_by_radius[radius]


def _percentiles(plane, half, total, bins: int):
    scaled = np.clip(plane, 0.0, 1.0)
    bin_index = np.minimum((scaled * bins).astype(np.int32), bins - 1)
    cdf = np.zeros(plane.shape, dtype=np.float64)
    p75 = np.zeros(plane.shape, dtype=np.float32)
    p90 = np.zeros(plane.shape, dtype=np.float32)
    seen75 = np.zeros(plane.shape, dtype=bool)
    seen90 = np.zeros(plane.shape, dtype=bool)
    _accumulate_bins(
        bin_index, half, bins, cdf, total, p75, seen75, p90, seen90
    )
    return p75, p90


def _accumulate_bins(bin_index, half, bins, cdf, total, p75, seen75, p90, seen90) -> None:
    target75 = 0.75 * total
    target90 = 0.90 * total
    for bin_id in range(bins):
        cdf += window_sum(bin_index == bin_id, half)
        value = np.float32((bin_id + 0.5) / bins)
        _mark(p75, seen75, cdf, target75, total, value)
        _mark(p90, seen90, cdf, target90, total, value)


def _mark(out, seen, cdf, target, total, value) -> None:
    hit = (~seen) & (total > 0) & (cdf >= target)
    out[hit] = value
    seen[hit] = True


def _maximum(plane: np.ndarray, half: int) -> np.ndarray:
    if half <= 0:
        return plane.copy()
    kernel = np.ones((half * 2 + 1, half * 2 + 1), dtype=np.uint8)
    return cv2.dilate(plane, kernel)


def _bounds(length: int, half: int) -> tuple[np.ndarray, np.ndarray]:
    index = np.arange(length)
    start = np.clip(index - half, 0, length)
    stop = np.clip(index + half + 1, 0, length)
    return start, stop


def _rect(sat, y0, y1, x0, x1) -> np.ndarray:
    return sat[y1][:, x1] - sat[y0][:, x1] - sat[y1][:, x0] + sat[y0][:, x0]


def _fraction_from_sat(sat, centers_y, centers_x, height, width, radius) -> np.ndarray:
    y0 = np.clip(centers_y - radius, 0, height)
    y1 = np.clip(centers_y + radius + 1, 0, height)
    x0 = np.clip(centers_x - radius, 0, width)
    x1 = np.clip(centers_x + radius + 1, 0, width)
    sums = _rect(sat, y0, y1, x0, x1)
    areas = (y1 - y0).astype(np.float64)[:, None] * (x1 - x0).astype(np.float64)[None, :]
    return (sums / np.maximum(areas, 1.0)).astype(np.float32)


def _query_values(prob, centers_y, centers_x, query_box) -> np.ndarray:
    x0, y0, x1, y1 = [int(value) for value in query_box]
    rows = np.flatnonzero((centers_y >= y0) & (centers_y < y1))
    cols = np.flatnonzero((centers_x >= x0) & (centers_x < x1))
    if len(rows) == 0 or len(cols) == 0:
        return _nearest_value(prob, centers_y, centers_x, x0, y0, x1, y1)
    return np.asarray(prob[np.ix_(rows, cols)], dtype=np.float32).ravel()


def _nearest_value(prob, centers_y, centers_x, x0, y0, x1, y1) -> np.ndarray:
    cy = (y0 + y1 - 1) / 2.0
    cx = (x0 + x1 - 1) / 2.0
    iy = int(np.argmin(np.abs(centers_y.astype(np.float64) - cy)))
    ix = int(np.argmin(np.abs(centers_x.astype(np.float64) - cx)))
    return np.asarray([prob[iy, ix]], dtype=np.float32)


def _outside_gap(start: int, stop: int, coord: np.ndarray) -> np.ndarray:
    return np.maximum(np.maximum(start - coord, coord - (stop - 1)), 0.0)


def _inside_gap(start: int, stop: int, coord: np.ndarray) -> np.ndarray:
    return np.minimum(coord - start, (stop - 1) - coord)
