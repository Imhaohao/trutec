"""Anchor-and-grow and query expansion algorithms for hatch matching.

Implements label-free test-time query expansion, anchor selection, local density
growth, morphological gap-filling, and connected-component growing.
"""

from __future__ import annotations

import cv2
import numpy as np

import hatch_texture
from inference_texture import native_grid_coordinates, predict_probability_grid
from texture_learning import grid_centers


def extract_query_crop_cells(
    prob_grid: np.ndarray,
    query_box: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    stride: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (y_indices, x_indices, probabilities) for cells inside query box."""
    height, width = image_shape
    centers_y, centers_x = grid_centers(height, width, stride)
    x0, y0, x1, y1 = query_box
    y_indices = np.flatnonzero((centers_y >= y0) & (centers_y < y1))
    x_indices = np.flatnonzero((centers_x >= x0) & (centers_x < x1))

    if len(y_indices) == 0 or len(x_indices) == 0:
        cy = (y0 + y1 - 1) / 2.0
        cx = (x0 + x1 - 1) / 2.0
        y_indices = np.array([int(np.argmin(np.abs(centers_y - cy)))])
        x_indices = np.array([int(np.argmin(np.abs(centers_x - cx)))])

    crop_probs = prob_grid[np.ix_(y_indices, x_indices)]
    return y_indices, x_indices, crop_probs


def expand_query_descriptor(
    features: np.ndarray,
    prob_grid: np.ndarray,
    query_box: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    stride: int = 4,
    anchor_quantile: float = 0.50,
) -> np.ndarray:
    """Re-estimate query descriptor from high-confidence anchor cells in query."""
    y_idx, x_idx, crop_probs = extract_query_crop_cells(
        prob_grid, query_box, image_shape, stride
    )
    flat_probs = crop_probs.ravel()
    tau = float(np.quantile(flat_probs, anchor_quantile))
    high_mask = crop_probs >= tau

    if not np.any(high_mask):
        high_mask = crop_probs >= np.min(crop_probs)

    crop_features = features[np.ix_(y_idx, x_idx)]
    selected_features = crop_features[high_mask]
    if selected_features.ndim == 1:
        selected_features = selected_features[np.newaxis, :]

    new_desc = np.median(selected_features, axis=0).astype(np.float32)
    return new_desc


def iterate_query_expansion(
    model: object,
    features: np.ndarray,
    initial_desc: np.ndarray,
    query_box: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    stride: int = 4,
    iterations: int = 1,
    anchor_quantile: float = 0.50,
) -> tuple[np.ndarray, np.ndarray]:
    """Run iterative query expansion and return (final_prob_grid, final_desc)."""
    current_desc = initial_desc
    current_grid = predict_probability_grid(model, features, current_desc)

    for _ in range(iterations):
        current_desc = expand_query_descriptor(
            features,
            current_grid,
            query_box,
            image_shape,
            stride=stride,
            anchor_quantile=anchor_quantile,
        )
        current_grid = predict_probability_grid(model, features, current_desc)

    return current_grid, current_desc


def compute_anchor_density(
    prob_grid: np.ndarray,
    anchor_threshold: float,
    window_pixels: int,
    stride: int = 4,
) -> np.ndarray:
    """Compute local density of anchors in a box window."""
    k_side = max(3, int(round(window_pixels / stride)))
    if k_side % 2 == 0:
        k_side += 1

    anchors = (prob_grid >= anchor_threshold).astype(np.float32)
    density = cv2.boxFilter(
        anchors,
        ddepth=cv2.CV_32F,
        ksize=(k_side, k_side),
        normalize=True,
        borderType=cv2.BORDER_REFLECT_101,
    )
    return np.asarray(density, dtype=np.float32)


def morphological_closing_grid(
    prob_grid: np.ndarray,
    window_pixels: int,
    stride: int = 4,
) -> np.ndarray:
    """Grayscale closing to bridge text and dimension line gaps inside hatch."""
    k_side = max(3, int(round(window_pixels / stride)))
    if k_side % 2 == 0:
        k_side += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_side, k_side))
    closed = cv2.morphologyEx(prob_grid, cv2.MORPH_CLOSE, kernel)
    return np.asarray(closed, dtype=np.float32)


def morphological_dilation_grid(
    prob_grid: np.ndarray,
    window_pixels: int,
    stride: int = 4,
) -> np.ndarray:
    """Grayscale dilation (max filter) across neighboring cells."""
    k_side = max(3, int(round(window_pixels / stride)))
    if k_side % 2 == 0:
        k_side += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_side, k_side))
    dilated = cv2.dilate(prob_grid, kernel)
    return np.asarray(dilated, dtype=np.float32)


def grow_connected_components(
    prob_grid: np.ndarray,
    density_map: np.ndarray,
    density_threshold: float = 0.20,
    min_area_cells: int = 20,
) -> np.ndarray:
    """Grow scores inside connected components of anchor density with hole filling."""
    binary_mask = (density_map >= density_threshold).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )

    result = np.zeros_like(prob_grid)
    for label_idx in range(1, num_labels):
        area = stats[label_idx, cv2.CC_STAT_AREA]
        if area < min_area_cells:
            continue

        comp_mask = (labels == label_idx).astype(np.uint8)
        contours, _ = cv2.findContours(
            comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        filled_mask = np.zeros_like(comp_mask)
        cv2.drawContours(filled_mask, contours, -1, color=1, thickness=-1)

        mask_bool = filled_mask > 0
        comp_scores = np.maximum(prob_grid[mask_bool], density_map[mask_bool])
        result[mask_bool] = comp_scores

    return np.asarray(result, dtype=np.float32)


def fuse_prob_and_density(
    prob_grid: np.ndarray,
    density_map: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Linear blend of expanded probability grid and anchor density map."""
    fused = alpha * prob_grid + (1.0 - alpha) * density_map
    return np.clip(fused, 0.0, 1.0).astype(np.float32)


def sample_grid_to_coordinates(
    grid: np.ndarray,
    height: int,
    width: int,
    stride: int,
    sample_y: np.ndarray,
    sample_x: np.ndarray,
) -> np.ndarray:
    """Bilinear sample 2D grid at given native (y, x) coordinates."""
    gy_native, gx_native = native_grid_coordinates(height, width, stride)
    gy = gy_native[sample_y]
    gx = gx_native[sample_x]

    gh, gw = grid.shape
    x0 = np.clip(np.floor(gx).astype(np.int32), 0, gw - 1)
    x1 = np.clip(x0 + 1, 0, gw - 1)
    y0 = np.clip(np.floor(gy).astype(np.int32), 0, gh - 1)
    y1 = np.clip(y0 + 1, 0, gh - 1)

    wx = (gx - x0).astype(np.float32)
    wy = (gy - y0).astype(np.float32)

    val00 = grid[y0, x0]
    val01 = grid[y0, x1]
    val10 = grid[y1, x0]
    val11 = grid[y1, x1]

    top = val00 * (1.0 - wx) + val01 * wx
    bottom = val10 * (1.0 - wx) + val11 * wx
    return (top * (1.0 - wy) + bottom * wy).astype(np.float32)
