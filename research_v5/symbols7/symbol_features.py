"""Label-free overlay-symbol channels on the stride-8 lattice.

Each cell sees square windows of 64, 128 and 256 px. Three numbers are measured
in every window, the same three numbers are measured on the g2-cleaned query
crop, and the absolute difference of each pair is kept.

Closed ink is the share of ink pixels whose connected component has an outer
contour that encloses a hole and whose bounding-box diameter is 20 to 120 px.
Circle count is the number of outer contours with circularity above 0.7 whose
bounding-box center falls in the window. A contour whose box is at least four
times as long as it is wide is skipped: that shape cannot reach circularity 0.7.
Large ink is the share of ink pixels in components with area above 400 px.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from constants import GRID_STRIDE
from features import integral_image
from structure_features import window_sums
from texture_learning import grid_centers

WINDOWS = (64, 128, 256)
STATS = ("closed_ink", "circle_count", "large_ink")
DIAMETER_MIN = 20
DIAMETER_MAX = 120
CIRCULARITY_MIN = 0.7
LARGE_AREA = 400
PAPER_LEVEL = 210
ASPECT_LIMIT = 4.0
CHANNEL_COUNT = len(WINDOWS) * len(STATS) * 2 + len(STATS)


def feature_names() -> list[str]:
    names = [f"{stat}_w{side}" for side in WINDOWS for stat in STATS]
    names.extend(f"{stat}_abs_w{side}" for side in WINDOWS for stat in STATS)
    names.extend(f"query_{stat}" for stat in STATS)
    if len(names) != CHANNEL_COUNT or len(names) != len(set(names)):
        raise RuntimeError("symbol feature names")
    return names


def drawing_ink(rgb: np.ndarray) -> np.ndarray:
    """Same ink test as color.extract_ink_and_chroma, without the float copies."""
    value = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[:, :, 2]
    spread = np.max(rgb, axis=2).astype(np.int16) - np.min(rgb, axis=2).astype(np.int16)
    return (value < 235) | (spread > 20)


def gray_ink(gray: np.ndarray) -> np.ndarray:
    return np.asarray(gray) < PAPER_LEVEL


def summarize_mask(ink: np.ndarray) -> np.ndarray:
    """Closed-ink fraction, outer-circle count, and large-ink fraction."""
    ink_u8 = _binary(ink)
    labels, closed_of, large_of, circles = _marks(ink_u8)
    ink_count = float(np.count_nonzero(ink_u8))
    closed_frac = _share(closed_of[labels], ink_count)
    large_frac = _share(large_of[labels], ink_count)
    return np.array([closed_frac, float(len(circles)), large_frac], np.float32)


def window_cube(rgb: np.ndarray) -> np.ndarray:
    """(grid_y, grid_x, window, stat) for the three window statistics."""
    ink = drawing_ink(rgb)
    centers_y, centers_x = grid_centers(int(rgb.shape[0]), int(rgb.shape[1]), GRID_STRIDE)
    return cube_from_ink(ink, centers_y, centers_x)


def cube_from_ink(ink: np.ndarray, centers_y, centers_x) -> np.ndarray:
    ink_u8 = _binary(ink)
    labels, closed_of, large_of, circles = _marks(ink_u8)
    closed = closed_of[labels]
    large = large_of[labels]
    del labels, closed_of, large_of
    plane = _circle_plane(ink_u8.shape, circles)
    return _stack_stats(ink_u8, closed, large, plane, centers_y, centers_x)


def query_stats(gray: np.ndarray, box) -> np.ndarray:
    from guards import crop_for

    x0, y0, x1, y1 = (int(value) for value in box)
    crop = np.ascontiguousarray(gray[y0:y1, x0:x1])
    cleaned = crop_for("g2", crop)
    return summarize_mask(gray_ink(cleaned))


def channels_from_cube(cube: np.ndarray, query: np.ndarray) -> np.ndarray:
    stats = np.asarray(query, dtype=np.float32).reshape(1, 1, 1, len(STATS))
    window = cube.reshape(cube.shape[0], cube.shape[1], -1)
    absolute = np.abs(cube - stats).reshape(cube.shape[0], cube.shape[1], -1)
    broadcast = np.empty((cube.shape[0], cube.shape[1], len(STATS)), np.float32)
    broadcast[:] = stats.reshape(1, 1, len(STATS))
    channels = np.concatenate([window, absolute, broadcast], axis=-1)
    if channels.shape[-1] != CHANNEL_COUNT:
        raise RuntimeError("symbol channel width")
    return channels


def _stack_stats(ink_u8, closed, large, plane, centers_y, centers_x) -> np.ndarray:
    ink_w = _window_stack(ink_u8.astype(np.float32), centers_y, centers_x)
    closed_w = _window_stack(closed.astype(np.float32), centers_y, centers_x)
    large_w = _window_stack(large.astype(np.float32), centers_y, centers_x)
    circle_w = _window_stack(plane, centers_y, centers_x)
    stacked = np.stack([_ratio(closed_w, ink_w), circle_w, _ratio(large_w, ink_w)], axis=-1)
    return np.ascontiguousarray(stacked, dtype=np.float32)


def _window_stack(values, centers_y, centers_x) -> np.ndarray:
    sat = integral_image(values)
    layers = [window_sums(sat, centers_y, centers_x, side) for side in WINDOWS]
    return np.stack(layers, axis=-1)


def _marks(ink_u8: np.ndarray):
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(ink_u8, 8)
    large_of = np.zeros(count, dtype=bool)
    if count > 1:
        large_of[1:] = stats[1:, cv2.CC_STAT_AREA] > LARGE_AREA
    closed_of = np.zeros(count, dtype=bool)
    circles: list[tuple[float, float]] = []
    _scan_contours(ink_u8, labels, closed_of, circles)
    return labels, closed_of, large_of, circles


def _scan_contours(ink_u8, labels, closed_of, circles) -> None:
    contours, hierarchy = cv2.findContours(ink_u8.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return
    nodes = hierarchy[0]
    for index, contour in enumerate(contours):
        _one_contour(contour, nodes[index], labels, closed_of, circles)


def _one_contour(contour, node, labels, closed_of, circles) -> None:
    if int(node[2]) >= 0:
        _mark_closed(contour, labels, closed_of)
    if int(node[3]) < 0:
        _maybe_circle(contour, circles)


def _mark_closed(contour, labels, closed_of) -> None:
    _x, _y, width, height = cv2.boundingRect(contour)
    diameter = max(int(width), int(height))
    if diameter < DIAMETER_MIN or diameter > DIAMETER_MAX:
        return
    label = _contour_label(contour, labels)
    if label > 0:
        closed_of[label] = True


def _maybe_circle(contour, circles) -> None:
    _x, _y, width, height = cv2.boundingRect(contour)
    if width <= 0 or height <= 0:
        return
    if max(width, height) >= ASPECT_LIMIT * min(width, height):
        return
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    if area <= 0.0 or perimeter <= 0.0:
        return
    if (4.0 * math.pi * area) / (perimeter * perimeter) <= CIRCULARITY_MIN:
        return
    circles.append((_y + (height - 1) / 2.0, _x + (width - 1) / 2.0))


def _contour_label(contour, labels) -> int:
    point = contour[0, 0]
    x = int(point[0])
    y = int(point[1])
    height, width = labels.shape
    if y < 0 or x < 0 or y >= height or x >= width:
        return 0
    return int(labels[y, x])


def _circle_plane(shape, circles) -> np.ndarray:
    plane = np.zeros(shape, np.float32)
    height, width = shape
    for y_value, x_value in circles:
        y_index = int(round(y_value))
        x_index = int(round(x_value))
        if 0 <= y_index < height and 0 <= x_index < width:
            plane[y_index, x_index] += 1.0
    return plane


def _binary(ink: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray((np.asarray(ink) > 0).astype(np.uint8))


def _share(mask, ink_count: float) -> float:
    if ink_count <= 0.0:
        return 0.0
    return float(np.count_nonzero(mask)) / ink_count


def _ratio(part, whole) -> np.ndarray:
    out = np.zeros(part.shape, np.float32)
    np.divide(part, whole, out=out, where=whole > 0)
    return out
