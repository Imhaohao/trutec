"""Label-free symbol-circle mask: closed, roughly circular ink outlines 30-140 px across."""
from __future__ import annotations

import cv2
import numpy as np

MIN_DIAMETER = 30
MAX_DIAMETER = 140
MIN_CIRCULARITY = 0.75
INK_LEVEL = 160


def circle_mask(gray: np.ndarray) -> np.ndarray:
    """Return a bool mask covering the filled interiors of circular closed outlines."""
    ink = (gray < INK_LEVEL).astype(np.uint8)
    contours, hierarchy = cv2.findContours(ink, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(gray.shape, dtype=np.uint8)
    if hierarchy is None:
        return mask.astype(bool)
    for index, contour in enumerate(contours):
        if hierarchy[0][index][3] < 0:
            continue  # holes only: the paper inside a closed outline
        if _is_symbol_circle(contour):
            cv2.drawContours(mask, contours, index, 1, thickness=-1)
            cv2.drawContours(mask, contours, index, 1, thickness=6)
    return mask.astype(bool)


def _is_symbol_circle(contour: np.ndarray) -> bool:
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0 or area <= 0:
        return False
    (_x, _y), radius = cv2.minEnclosingCircle(contour)
    diameter = 2.0 * radius
    if not MIN_DIAMETER <= diameter <= MAX_DIAMETER:
        return False
    circularity = 4.0 * np.pi * area / (perimeter * perimeter)
    fill = area / (np.pi * radius * radius)
    return circularity >= MIN_CIRCULARITY and fill >= 0.6


def crop_has_circles(gray: np.ndarray, box) -> bool:
    x0, y0, x1, y1 = (int(v) for v in box)
    pad = MAX_DIAMETER // 2
    window = gray[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad]
    local = circle_mask(window)
    inner = local[min(pad, y0):min(pad, y0) + (y1 - y0), min(pad, x0):min(pad, x0) + (x1 - x0)]
    return bool(inner.mean() > 0.05)
