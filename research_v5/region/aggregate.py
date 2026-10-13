"""Aggregate a full-image probability grid inside label-free regions.

Statistics are computed on every stride-grid cell. Reviewed coordinates are
used only afterwards, to read the finished region score at pixels the harness
asks for.
"""

from __future__ import annotations

import numpy as np

from inference_texture import native_grid_coordinates, predict_probability_grid
from evaluate_texture_sweep import remap_known_coordinates


def center_labels(labels: np.ndarray, stride: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = labels.shape
    rows, cols = _centers(height, width, stride)
    return labels[np.ix_(rows, cols)], rows, cols


def sample_pixels(
    probability: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    stride: int,
    height: int,
    width: int,
) -> np.ndarray:
    grid_y, grid_x = native_grid_coordinates(height, width, stride)
    return remap_known_coordinates(probability, grid_x[x], grid_y[y])


def probability_columns(region_ids: np.ndarray, probability: np.ndarray) -> dict[str, np.ndarray]:
    """Mean, quantiles, and high-probability fractions for every region id."""
    ids = region_ids.reshape(-1).astype(np.int32, copy=False)
    values = probability.reshape(-1).astype(np.float64, copy=False)
    counts = np.bincount(ids)
    safe = np.maximum(counts, 1)
    columns = {
        "prob_mean": _bincount_mean(ids, values, safe),
        "frac_ge_0.25": _bincount_mean(ids, (values >= 0.25).astype(np.float64), safe),
        "frac_ge_0.50": _bincount_mean(ids, (values >= 0.50).astype(np.float64), safe),
    }
    columns.update(_quantiles(ids, values, counts))
    return columns


def feature_columns(features: np.ndarray, region_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-region mean and median of grid descriptors. Empty ids stay zero."""
    ids = region_ids.reshape(-1).astype(np.int32, copy=False)
    vectors = features.reshape(-1, features.shape[-1])
    order = np.argsort(ids, kind="mergesort")
    ordered_ids = ids[order]
    ordered = vectors[order]
    starts, ends, unique = _group_bounds(ordered_ids)
    width = vectors.shape[1]
    max_id = int(ordered_ids[-1]) + 1 if ordered_ids.size else 1
    means = np.zeros((max_id, width), dtype=np.float32)
    medians = np.zeros((max_id, width), dtype=np.float32)
    counts = np.zeros(max_id, dtype=np.int32)
    _write_feature_stats(means, medians, counts, ordered, unique, starts, ends)
    return means, medians, counts


def score_descriptors(model: object, vectors: np.ndarray, query: np.ndarray, counts: np.ndarray) -> np.ndarray:
    scores = np.zeros(vectors.shape[0], dtype=np.float32)
    present = counts > 0
    if not np.any(present):
        return scores
    grid = vectors[present][:, None, :]
    scores[present] = predict_probability_grid(model, grid, query)[:, 0]
    return scores


def gabor_similarity(
    medians: np.ndarray, query: np.ndarray, gabor_index: np.ndarray, counts: np.ndarray
) -> np.ndarray:
    region = medians[:, gabor_index]
    target = query[gabor_index]
    gap = np.abs(region - target[np.newaxis, :])
    scale = np.abs(region) + np.abs(target[np.newaxis, :]) + np.float32(1e-4)
    distance = (gap / scale).mean(axis=1)
    similarity = np.clip(1.0 - distance, 0.0, 1.0).astype(np.float32)
    similarity[counts == 0] = 0.0
    return similarity


def spread_present(values: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Min-max of regions that contain grid cells, computed on the full image."""
    present = counts > 0
    if not np.any(present):
        return np.zeros_like(values)
    low = float(values[present].min())
    high = float(values[present].max())
    span = high - low
    if span < 1e-8:
        return present.astype(np.float32)
    scaled = (values - np.float32(low)) / np.float32(span)
    scaled = np.clip(scaled, 0.0, 1.0).astype(np.float32)
    scaled[~present] = 0.0
    return scaled


def lookup(column: np.ndarray, region_ids: np.ndarray) -> np.ndarray:
    ids = region_ids.astype(np.int32, copy=False)
    if int(ids.max()) >= column.shape[0]:
        raise ValueError("region id is outside the aggregated table")
    return column[ids]


def mix_pixel_and_region(pixel: np.ndarray, region: np.ndarray) -> dict[str, np.ndarray]:
    safe_region = np.maximum(region, 0.0)
    return {
        "min_pixel_region": np.minimum(pixel, region),
        "avg_pixel_region": np.float32(0.5) * (pixel + region),
        "geom_pixel_region": np.sqrt(np.maximum(pixel, 0.0) * safe_region).astype(np.float32),
    }


def _centers(height: int, width: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    from texture_learning import grid_centers

    return grid_centers(height, width, stride)


def weighted_mean(region_ids: np.ndarray, probability: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Probability mean weighted by a non-negative full-grid map, for example ink density."""
    ids = region_ids.reshape(-1).astype(np.int32, copy=False)
    values = probability.reshape(-1).astype(np.float64, copy=False)
    safe_weights = np.maximum(weights.reshape(-1).astype(np.float64, copy=False), 0.0)
    sums = np.bincount(ids, weights=values * safe_weights)
    denom = np.bincount(ids, weights=safe_weights)
    return (sums / np.maximum(denom, 1e-8)).astype(np.float32)


def _bincount_mean(ids: np.ndarray, values: np.ndarray, safe: np.ndarray) -> np.ndarray:
    sums = np.bincount(ids, weights=values)
    return (sums / safe).astype(np.float32)


def _quantiles(ids: np.ndarray, values: np.ndarray, counts: np.ndarray) -> dict[str, np.ndarray]:
    order = np.argsort(ids, kind="mergesort")
    ordered_ids = ids[order]
    ordered = values[order]
    starts, ends, unique = _group_bounds(ordered_ids)
    max_id = int(counts.shape[0])
    q25 = np.zeros(max_id, dtype=np.float32)
    q50 = np.zeros(max_id, dtype=np.float32)
    q75 = np.zeros(max_id, dtype=np.float32)
    _write_quantiles(q25, q50, q75, ordered, unique, starts, ends)
    return {"prob_q25": q25, "prob_q50": q50, "prob_q75": q75}


def _group_bounds(ordered_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if ordered_ids.size == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, empty
    cuts = np.flatnonzero(np.diff(ordered_ids)) + 1
    starts = np.concatenate((np.zeros(1, dtype=np.int64), cuts))
    ends = np.concatenate((cuts, np.array([ordered_ids.size], dtype=np.int64)))
    return starts, ends, ordered_ids[starts]


def _write_quantiles(
    q25: np.ndarray,
    q50: np.ndarray,
    q75: np.ndarray,
    ordered: np.ndarray,
    unique: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> None:
    for region_id, start, end in zip(unique, starts, ends):
        chunk = ordered[int(start):int(end)]
        low, mid, high = np.quantile(chunk, (0.25, 0.50, 0.75))
        q25[int(region_id)] = low
        q50[int(region_id)] = mid
        q75[int(region_id)] = high


def _write_feature_stats(
    means: np.ndarray,
    medians: np.ndarray,
    counts: np.ndarray,
    ordered: np.ndarray,
    unique: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> None:
    for region_id, start, end in zip(unique, starts, ends):
        chunk = ordered[int(start):int(end)]
        slot = int(region_id)
        means[slot] = chunk.mean(axis=0)
        medians[slot] = np.median(chunk, axis=0)
        counts[slot] = chunk.shape[0]
