"""Query-conditioned integration for shared texture and cadence features."""
from __future__ import annotations

import math

import numpy as np

import hatch_cadence
import hatch_texture
import texture_learning

STRIDE = 4
BASE_CHANNELS = len(hatch_texture.texture_feature_names("base"))
CADENCE_CHANNELS = len(hatch_cadence.FEATURE_NAMES)
AUGMENTED_CHANNELS = BASE_CHANNELS + CADENCE_CHANNELS


def query_descriptor(
    gray: np.ndarray,
    shared_features: np.ndarray,
    query_box: tuple[int, int, int, int],
) -> tuple[np.ndarray, hatch_cadence.LagState]:
    """Return the 51-channel exact-crop query descriptor and its estimated lag."""
    _validate_image_and_grid(gray, shared_features)
    base = hatch_texture.descriptor_for_query(
        gray, shared_features, query_box, stride=STRIDE, source="crop", feature_set="base"
    )
    x0, y0, x1, y1 = query_box
    crop = gray[y0:y1, x0:x1]
    lag = hatch_cadence.estimate_query_lag(crop)
    cadence = hatch_cadence.query_cadence_descriptor(crop, lag)
    result = np.concatenate((base, cadence)).astype(np.float32)
    if result.shape != (AUGMENTED_CHANNELS,) or not np.isfinite(result).all():
        raise ValueError("query descriptor must contain 51 finite channels")
    return result, lag


def pair_features_for_cells(
    gray: np.ndarray,
    shared_features: np.ndarray,
    grid_coordinates: np.ndarray,
    query: np.ndarray,
    lag: hatch_cadence.LagState,
) -> np.ndarray:
    """Build 204 pair features for requested global stride-four grid cells."""
    _validate_image_and_grid(gray, shared_features)
    coordinates = np.asarray(grid_coordinates)
    if (coordinates.ndim != 2 or coordinates.shape[1] != 2
            or not np.issubdtype(coordinates.dtype, np.integer)):
        raise ValueError("grid_coordinates must be an integer array of shape (N, 2)")
    grid_h, grid_w = shared_features.shape[:2]
    _validate_coordinates(coordinates, grid_h, grid_w)
    query_vector = np.asarray(query)
    if query_vector.shape != (AUGMENTED_CHANNELS,) or not np.isfinite(query_vector).all():
        raise ValueError("query must have shape (51,) and contain only finite values")
    local_base = shared_features[coordinates[:, 0], coordinates[:, 1], :]
    cadence = hatch_cadence.cadence_features_tiled(gray, lag, coordinates)
    local = np.concatenate((local_base, cadence), axis=1).astype(np.float32)
    result = texture_learning.pair_features(local, query_vector.astype(np.float32))
    if result.shape != (len(coordinates), AUGMENTED_CHANNELS * 4):
        raise ValueError("pair feature output has an unexpected shape")
    return result


def _validate_image_and_grid(gray: np.ndarray, features: np.ndarray) -> None:
    if not isinstance(gray, np.ndarray) or gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError("gray must be a two-dimensional uint8 array")
    if min(gray.shape) <= 0:
        raise ValueError("gray dimensions must be positive")
    if not isinstance(features, np.ndarray) or features.ndim != 3:
        raise ValueError("shared_features must have shape (grid_height, grid_width, channels)")
    expected = (math.ceil(gray.shape[0] / STRIDE), math.ceil(gray.shape[1] / STRIDE), BASE_CHANNELS)
    if features.shape != expected:
        raise ValueError(f"shared_features shape {features.shape} does not match {expected}")
    if not np.isfinite(features).all():
        raise ValueError("shared_features contains non-finite values")


def _validate_coordinates(coords: np.ndarray, grid_h: int, grid_w: int) -> None:
    if (np.any(coords < 0) or np.any(coords[:, 0] >= grid_h)
            or np.any(coords[:, 1] >= grid_w)):
        raise ValueError("grid coordinate lies outside the global stride-four grid")
