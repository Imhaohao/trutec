"""Label-free partitions of one drawing into regions.

Partitions never read labels, document ids, or reviewed coordinates.
"""

from __future__ import annotations

import cv2
import numpy as np


def ink_mask(rgb: np.ndarray, deficit_threshold: int) -> np.ndarray:
    """Pixels whose darkest channel is at least `deficit_threshold` below white."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("rgb must be uint8 with shape (H, W, 3)")
    if deficit_threshold <= 0 or deficit_threshold > 255:
        raise ValueError("deficit_threshold must lie in 1..255")
    darkest = rgb.min(axis=2)
    return np.uint8(darkest <= np.uint8(255 - deficit_threshold))


def paper_after_thin_removal(
    rgb: np.ndarray, open_radius: int, deficit_threshold: int
) -> np.ndarray:
    """Connected paper after an opening deletes thin strokes. Thick ink joins the nearest paper."""
    thick = _opened(ink_mask(rgb, deficit_threshold), open_radius)
    return absorb_barriers(np.uint8(thick == 0))


def boundary_enclosures(
    rgb: np.ndarray,
    open_radius: int,
    close_radius: int,
    deficit_threshold: int,
) -> np.ndarray:
    """Seal thick walls, then let thin strokes fall into the enclosed paper component."""
    thick = _opened(ink_mask(rgb, deficit_threshold), open_radius)
    walls = _closed(thick, close_radius)
    return absorb_barriers(np.uint8(walls == 0))


def filled_closing(
    rgb: np.ndarray, close_radius: int, deficit_threshold: int
) -> np.ndarray:
    """Close ink so nearby strokes become one blob. Each blob and each remaining hole is a region."""
    closed = _closed(ink_mask(rgb, deficit_threshold), close_radius)
    return fill_and_holes(closed)


def texture_superpixels(rgb: np.ndarray, cell: int, blur_radius: int) -> np.ndarray:
    """Watershed superpixels on the gradient of local color and luminance variance."""
    if cell < 4:
        raise ValueError("cell must be at least 4")
    gradient = texture_gradient(rgb, blur_radius)
    markers = _place_seeds(gradient, cell)
    _run_watershed(gradient, markers)
    return _fill_missing(markers)


def texture_gradient(rgb: np.ndarray, blur_radius: int) -> np.ndarray:
    if blur_radius < 3:
        raise ValueError("blur_radius must be at least 3")
    image = rgb.astype(np.float32)
    window = (blur_radius, blur_radius)
    color_edge = _channel_gradient(cv2.blur(image, window))
    gray = image.mean(axis=2)
    variance_edge = _gradient_magnitude(_local_variance(gray, window))
    return color_edge + variance_edge


def absorb_barriers(free: np.ndarray) -> np.ndarray:
    """4-connected components of free pixels. Barrier pixels take the nearest component id."""
    _count, labels = cv2.connectedComponents(free, connectivity=4)
    if not np.any(free == 0):
        return labels
    if int(labels.max()) == 0:
        return np.ones(free.shape, dtype=np.int32)
    return _nearest_known_labels(labels, free > 0)


def fill_and_holes(closed: np.ndarray) -> np.ndarray:
    """8-connected filled blobs plus 4-connected holes, in one label image."""
    _fill_count, fill_labels = cv2.connectedComponents(closed, connectivity=8)
    holes = np.uint8(closed == 0)
    _hole_count, hole_labels = cv2.connectedComponents(holes, connectivity=4)
    offset = int(fill_labels.max())
    combined = np.where(closed > 0, fill_labels, hole_labels + offset)
    return combined.astype(np.int32, copy=False)


def area_stats(labels: np.ndarray) -> dict[str, float]:
    counts = np.bincount(labels.ravel())
    present = counts[counts > 0]
    return {
        "region_count": float(present.size),
        "median_area": float(np.median(present)),
        "p10_area": float(np.percentile(present, 10)),
    }


def _opened(ink: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return ink
    return cv2.morphologyEx(ink, cv2.MORPH_OPEN, _disk(radius))


def _closed(ink: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return ink
    return cv2.morphologyEx(ink, cv2.MORPH_CLOSE, _disk(radius))


def _disk(radius: int) -> np.ndarray:
    size = 2 * radius + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _nearest_known_labels(labels: np.ndarray, known: np.ndarray) -> np.ndarray:
    source = np.where(known, np.uint8(0), np.uint8(255))
    _distance, nearest = cv2.distanceTransformWithLabels(
        source, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL,
    )
    rows, cols = np.nonzero(source == 0)
    return labels[rows[nearest - 1], cols[nearest - 1]]


def _channel_gradient(image: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt((gx * gx + gy * gy).sum(axis=2))


def _gradient_magnitude(image: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def _local_variance(gray: np.ndarray, window: tuple[int, int]) -> np.ndarray:
    mean = cv2.blur(gray, window)
    square_mean = cv2.blur(gray * gray, window)
    return np.maximum(square_mean - mean * mean, 0.0)


def _place_seeds(gradient: np.ndarray, cell: int) -> np.ndarray:
    height, width = gradient.shape
    markers = np.zeros((height, width), dtype=np.int32)
    label = 1
    for y in range(cell // 2, height, cell):
        label = _seed_row(gradient, markers, y, cell, label)
    if label == 1:
        markers[0, 0] = 1
    return markers


def _seed_row(
    gradient: np.ndarray, markers: np.ndarray, y: int, cell: int, label: int
) -> int:
    width = gradient.shape[1]
    for x in range(cell // 2, width, cell):
        label = _write_seed(gradient, markers, y, x, cell, label)
    return label


def _write_seed(
    gradient: np.ndarray,
    markers: np.ndarray,
    y: int,
    x: int,
    cell: int,
    label: int,
) -> int:
    half = max(1, cell // 4)
    y0 = max(0, y - half)
    x0 = max(0, x - half)
    patch = gradient[y0:y + half + 1, x0:x + half + 1]
    local_y, local_x = np.unravel_index(int(np.argmin(patch)), patch.shape)
    seed_y = y0 + int(local_y)
    seed_x = x0 + int(local_x)
    if markers[seed_y, seed_x] != 0:
        return label
    markers[seed_y, seed_x] = label
    return label + 1


def _run_watershed(gradient: np.ndarray, markers: np.ndarray) -> None:
    scale = float(np.percentile(gradient, 99)) + 1e-6
    scaled = np.clip(gradient / scale * 255.0, 0, 255).astype(np.uint8)
    color = cv2.cvtColor(scaled, cv2.COLOR_GRAY2BGR)
    cv2.watershed(color, markers)


def _fill_missing(markers: np.ndarray) -> np.ndarray:
    if not np.any(markers < 1):
        return markers
    known = markers >= 1
    filled = _nearest_known_labels(markers, known)
    return filled.astype(np.int32, copy=False)
