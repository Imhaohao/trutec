"""Label-free symbol-circle detection (large labeled symbol bubbles) with a fixed Hough setting."""
from __future__ import annotations

import cv2
import numpy as np

SCALE = 0.5
PARAMS = {"dp": 1.2, "minDist": 20, "param1": 150, "param2": 60, "minRadius": 14, "maxRadius": 40}


def detect_circles(gray: np.ndarray) -> np.ndarray:
    """Return an (n, 3) array of native-resolution (x, y, r) symbol circles."""
    small = cv2.medianBlur(cv2.resize(gray, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA), 3)
    found = cv2.HoughCircles(small, cv2.HOUGH_GRADIENT, **PARAMS)
    if found is None:
        return np.zeros((0, 3), dtype=np.float32)
    return (found[0] / SCALE).astype(np.float32)


def circle_mask(shape, circles: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    for x, y, r in circles:
        cv2.circle(mask, (int(round(x)), int(round(y))), int(round(r)), 1, -1)
    return mask.astype(bool)


def query_shows_circles(circles: np.ndarray, box) -> bool:
    x0, y0, x1, y1 = (float(v) for v in box)
    centers = circles[:, :2] if len(circles) else np.zeros((0, 2))
    return bool(np.any((centers[:, 0] >= x0) & (centers[:, 0] < x1) & (centers[:, 1] >= y0) & (centers[:, 1] < y1)))
