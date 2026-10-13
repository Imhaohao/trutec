"""Score a stage-3 feature volume. Label-free."""
from __future__ import annotations

import limits  # noqa: F401

import numpy as np

from features import cell_index
from limits import require
from model import positive_probability


def score_grid(model, features: np.ndarray) -> np.ndarray:
    height, width, channels = (int(features.shape[0]), int(features.shape[1]), int(features.shape[2]))
    require(channels == int(model.n_features_in_), "stage3 width")
    flat = np.ascontiguousarray(features).reshape(-1, channels)
    return positive_probability(model, flat).reshape(height, width)


def score_coordinates(model, features: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Per-coordinate scoring: each pixel is the stage-3 probability of its cell."""
    require(int(features.shape[-1]) == int(model.n_features_in_), "stage3 width")
    grid_w = int(features.shape[1])
    iy, ix = cell_index(y, x, int(features.shape[0]), grid_w)
    keys = iy * grid_w + ix
    unique, inverse = np.unique(keys, return_inverse=True)
    rows = unique // grid_w
    cols = unique % grid_w
    predicted = positive_probability(model, features[rows, cols])
    return predicted[inverse]
