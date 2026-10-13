"""Label-free cutoff rules read from the feature vector. No labels."""
from __future__ import annotations

import limits  # noqa: F401

import numpy as np

from experiment import CLIP_HIGH, CLIP_LOW, MEDIAN_FRACTIONS
from scoremap import feature_index


def fraction_name(fraction: float) -> str:
    return f"median_x{fraction:.2f}"


def clipped_name(name: str) -> str:
    return f"{name}__clip_{CLIP_LOW:.2f}_{CLIP_HIGH:.2f}"


def otsu_column(features: np.ndarray) -> np.ndarray:
    return np.asarray(features[:, feature_index("otsu")], dtype=np.float64)


def median_column(features: np.ndarray, fraction: float) -> np.ndarray:
    median = features[:, feature_index("query_median")]
    return np.asarray(fraction * median, dtype=np.float64)


def clip_column(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=np.float64), CLIP_LOW, CLIP_HIGH)


def rule_columns(features: np.ndarray) -> dict[str, list[float]]:
    columns = {"otsu": otsu_column(features).tolist()}
    for fraction in MEDIAN_FRACTIONS:
        columns[fraction_name(fraction)] = median_column(features, fraction).tolist()
    return columns
