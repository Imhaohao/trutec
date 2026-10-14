"""Label-free features of one full gated score map and its query box.

Nothing here reads a label, a reviewed coordinate, or a split file.
"""
from __future__ import annotations

import limits  # noqa: F401

import numpy as np

from experiment import LEVELS, OTSU_BINS, QUANTILES


def feature_names() -> list[str]:
    names = [f"quantile_{int(round(level * 100)):02d}" for level in QUANTILES]
    names += [f"fraction_above_{level:.1f}" for level in LEVELS]
    names += ["query_median", "otsu", "fraction_above_otsu", "box_width", "box_height", "box_area"]
    if len(names) != len(set(names)):
        raise RuntimeError("duplicate feature name")
    return names


def feature_index(name: str) -> int:
    return feature_names().index(name)


def map_features(score: np.ndarray, box) -> np.ndarray:
    flat = _flat(score)
    _check_box(score, box)
    otsu = otsu_threshold(flat)
    width, height, area = box_geometry(box)
    values = np.concatenate(
        [
            np.quantile(flat, QUANTILES),
            _fractions(flat),
            np.array(
                [box_median(score, box), otsu, float(np.mean(flat >= otsu)), width, height, area],
                dtype=np.float64,
            ),
        ]
    )
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (len(feature_names()),) or not np.isfinite(result).all():
        raise RuntimeError("feature vector is the wrong shape or non-finite")
    return result


def otsu_threshold(values: np.ndarray, bins: int = OTSU_BINS) -> float:
    flat = np.asarray(values, dtype=np.float64).ravel()
    counts, edges = np.histogram(np.clip(flat, 0.0, 1.0), bins=bins, range=(0.0, 1.0))
    return _threshold_from_histogram(counts, edges)


def box_geometry(box) -> tuple[float, float, float]:
    x0, y0, x1, y1 = _box_parts(box)
    width = float(x1 - x0)
    height = float(y1 - y0)
    return width, height, width * height


def box_median(score: np.ndarray, box) -> float:
    x0, y0, x1, y1 = _box_parts(box)
    patch = np.asarray(score[y0:y1, x0:x1], dtype=np.float64)
    if patch.size == 0:
        raise RuntimeError("empty query box")
    return float(np.median(patch))


def _flat(score: np.ndarray) -> np.ndarray:
    array = np.asarray(score)
    if array.ndim != 2 or array.size == 0 or not np.isfinite(array).all():
        raise RuntimeError("score map must be a finite 2d array")
    return np.asarray(array, dtype=np.float64).ravel()


def _fractions(flat: np.ndarray) -> np.ndarray:
    return np.array([float(np.mean(flat > level)) for level in LEVELS], dtype=np.float64)


def _check_box(score: np.ndarray, box) -> None:
    x0, y0, x1, y1 = _box_parts(box)
    height, width = score.shape
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise RuntimeError("query box outside the score map")


def _box_parts(box) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = (int(value) for value in box)
    return x0, y0, x1, y1


def _threshold_from_histogram(counts: np.ndarray, edges: np.ndarray) -> float:
    if int(counts.sum()) == 0:
        return 0.0
    split = _otsu_bin(counts)
    if split < 0:
        return float(edges[int(np.argmax(counts))])
    return float(edges[split + 1])


def _otsu_bin(counts: np.ndarray) -> int:
    total = float(counts.sum())
    cumulative = np.cumsum(counts)
    cumulative_value = np.cumsum(counts * np.arange(len(counts), dtype=np.float64))
    total_value = float(cumulative_value[-1])
    best_index = -1
    best_score = 0.0
    for index, weight0 in enumerate(cumulative[:-1]):
        score = _between_variance(float(weight0), total, float(cumulative_value[index]), total_value)
        if score > best_score:
            best_score = score
            best_index = index
    return best_index


def _between_variance(weight0: float, total: float, sum0: float, total_value: float) -> float:
    weight1 = total - weight0
    if weight0 <= 0.0 or weight1 <= 0.0:
        return 0.0
    mean0 = sum0 / weight0
    mean1 = (total_value - sum0) / weight1
    return weight0 * weight1 * (mean0 - mean1) ** 2
