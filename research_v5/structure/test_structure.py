"""Unit tests for structure features, window sums, and hard-negative weights."""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

import limits  # noqa: F401

import numpy as np

from features import integral_image
from structure_features import (
    PERIODS,
    _fundamental,
    bank_from_rgb,
    feature_names,
    orientation_bin,
    reference_from_rgb,
    structure_volume,
    window_sums,
)
from weights import apply_hard_negative

HERE = Path(__file__).resolve().parent
LABEL_FREE = ("structure_features.py", "banks.py")
BANNED = ("reviewed_domains", "devharness", "fit.json", "development-real", "labels")


class StructureTests(unittest.TestCase):
    def test_window_sum_matches_the_crop(self) -> None:
        values = np.arange(100, dtype=np.float64).reshape(10, 10)
        got = window_sums(integral_image(values), np.array([5]), np.array([5]), 4)
        self.assertEqual(float(got[0, 0]), float(values[3:7, 3:7].sum()))

    def test_shortest_strong_period_wins(self) -> None:
        stack = np.zeros((1, 1, 1, len(PERIODS)), np.float32)
        stack[..., PERIODS.index(64)] = 1.0
        stack[..., PERIODS.index(32)] = 0.95
        period, strength = _fundamental(stack)
        self.assertEqual(float(period[0, 0, 0]), 32.0)
        self.assertAlmostEqual(float(strength[0, 0, 0]), 0.95)

    def test_vertical_lines_have_period_32(self) -> None:
        rgb = _vertical_lines()
        reference = reference_from_rgb(rgb, (16, 16, 240, 240))
        self.assertEqual(int(np.argmax(reference.histogram)), 0)
        self.assertEqual(reference.period, 32.0)
        bank = bank_from_rgb(rgb)
        center = bank["period"].shape[0] // 2
        self.assertEqual(float(bank["period"][center, center, 2]), 32.0)
        volume = structure_volume(bank, reference)
        self.assertEqual(volume.shape[-1], len(feature_names()))
        self.assertTrue(np.isfinite(volume).all())
        self.assertGreater(float(volume[center, center, feature_names().index("orient_cos_w128")]), 0.9)

    def test_hard_negative_is_three_times_an_easy_negative(self) -> None:
        labels = np.array([1, 1, 0, 0, 0], dtype=np.int8)
        documents = np.zeros(5, dtype=np.int32)
        stage1 = np.array([0.2, 0.2, 0.9, 0.1, 0.8], dtype=np.float32)
        base = np.array([0.25, 0.25, 0.5 / 3, 0.5 / 3, 0.5 / 3], dtype=np.float64)
        adjusted = apply_hard_negative(base, labels, stage1, documents, 0.5)
        self.assertAlmostEqual(float(adjusted[2] / adjusted[3]), 3.0)
        self.assertAlmostEqual(float(adjusted[labels == 0].sum()), 0.5)
        self.assertAlmostEqual(float(adjusted[labels == 1].sum()), 0.5)
        untouched = apply_hard_negative(base, labels, stage1, documents, None)
        self.assertTrue(np.allclose(untouched, base))

    def test_orientation_bins_are_centered(self) -> None:
        bins = orientation_bin(np.array([0.0, np.pi / 2, np.pi / 4, np.pi - 1e-6]))
        self.assertTrue(np.array_equal(bins, np.array([0, 4, 2, 0])))

    def test_feature_modules_do_not_read_labels(self) -> None:
        for name in LABEL_FREE:
            source = (HERE / name).read_text(encoding="utf-8")
            for banned in BANNED:
                self.assertNotIn(banned, source, name)
            self._assert_imports(ast.parse(source), name)

    def _assert_imports(self, tree: ast.AST, name: str) -> None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            else:
                continue
            for module in modules:
                self.assertNotIn("devharness", module, name)
                self.assertNotIn("evaluate", module, name)


def _vertical_lines() -> np.ndarray:
    gray = np.full((256, 256), 240, np.uint8)
    for left in range(0, 256, 32):
        gray[:, left : left + 4] = 20
    return np.repeat(gray[:, :, None], 3, axis=2)


if __name__ == "__main__":
    unittest.main()
