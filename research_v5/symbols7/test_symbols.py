"""CPU checks for overlay-symbol channels."""
from __future__ import annotations

import unittest

import guard

guard.activate()

import cv2
import numpy as np

from constants import GRID_STRIDE
from symbol_features import cube_from_ink, feature_names, gray_ink, summarize_mask
from texture_learning import grid_centers


class SymbolTests(unittest.TestCase):
    def test_circles_over_a_hatch_raise_the_closed_contour_fraction(self) -> None:
        plain = summarize_mask(gray_ink(_hatch()))
        covered = summarize_mask(gray_ink(_circles_over_hatch()))
        self.assertGreater(float(covered[0]), float(plain[0]) + 0.01)
        self.assertGreater(float(covered[1]), float(plain[1]))
        self.assertGreater(_window_closed(_circles_over_hatch(), 100), _window_closed(_circles_over_hatch(), 16))

    def test_plain_hatch_does_not_raise_the_closed_contour_fraction(self) -> None:
        plain = summarize_mask(gray_ink(_hatch()))
        self.assertEqual(float(plain[0]), 0.0)
        self.assertEqual(float(plain[1]), 0.0)
        self.assertEqual(len(feature_names()), 21)


def _hatch(size: int = 400) -> np.ndarray:
    image = np.full((size, size), 255, np.uint8)
    for y_index in range(4, size, 12):
        image[y_index, 4 : size - 4] = 0
    return image


def _circles_over_hatch() -> np.ndarray:
    image = _hatch()
    for center, radius in (((100, 100), 30), ((300, 280), 36)):
        cv2.circle(image, center, radius + 4, 255, -1)
        cv2.circle(image, center, radius, 0, 2)
    return image


def _window_closed(image: np.ndarray, target: int) -> float:
    ink = gray_ink(image)
    centers_y, _centers_x = grid_centers(int(image.shape[0]), int(image.shape[1]), GRID_STRIDE)
    cube = cube_from_ink(ink, centers_y, centers_y)
    index = int(np.argmin(np.abs(centers_y - target)))
    return float(cube[index, index, 0, 0])


if __name__ == "__main__":
    unittest.main()
