"""Unit tests for stage-3 features and stride-8 lookup."""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

import limits  # noqa: F401

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from cells import sample_indices
from context_features import (
    WINDOWS_PX,
    context_volume,
    feature_names,
    half_cells,
    lookup,
    plane_window_stats,
    sat_window,
)
from features import integral_image, window_sum
from model import scores_at_coordinates
from stage3_score import score_coordinates, score_grid

HERE = Path(__file__).resolve().parent
LABEL_FREE = ("context_features.py", "grids.py", "stage3_score.py")
BANNED = ("reviewed_domains", "devharness", "fit.json", "development-real", "labels")


class ContextTests(unittest.TestCase):
    def test_features_do_not_read_labels(self) -> None:
        for name in LABEL_FREE:
            _assert_label_free(HERE / name)
        _assert_volume_ignores_an_unused_array()

    def test_stride8_lookup_equals_coordinate_scoring(self) -> None:
        volume = _tiny_volume()
        model = _tiny_model(volume)
        grid = score_grid(model, volume)
        y = np.array([0, 7, 8, 15, 100])
        x = np.array([0, 7, 9, 31, 2])
        looked = lookup(grid, y, x)
        direct = score_coordinates(model, volume, y, x)
        stacker = scores_at_coordinates(model, volume, y, x)
        self.assertTrue(np.allclose(looked, direct))
        self.assertTrue(np.allclose(looked, stacker))

    def test_unknown_centers_are_not_negatives(self) -> None:
        positive, known, centers_y, centers_x = _unknown_fixture()
        pos_idx, neg_idx, n_pos, n_neg = sample_indices(
            positive, known, centers_y, centers_x, 10, 0
        )
        self.assertEqual((n_pos, n_neg), (1, 1))
        self.assertTrue(np.array_equal(pos_idx, np.array([[0, 0]])))
        self.assertTrue(np.array_equal(neg_idx, np.array([[0, 1]])))

    def test_window_matches_stacker_sum_and_constant_plane(self) -> None:
        _assert_half_widths()
        values = np.arange(12, dtype=np.float64).reshape(3, 4)
        sat = integral_image(values)
        self.assertTrue(np.allclose(sat_window(sat, 1), window_sum(values, 1)))
        stats = plane_window_stats(np.full((8, 8), 0.2, dtype=np.float32), 1)
        self.assertTrue(np.allclose(stats["mean"], 0.2))
        self.assertTrue(np.allclose(stats["max"], 0.2))
        self.assertTrue(np.allclose(stats["frac_gt_0_3"], 0.0))
        self.assertTrue(np.allclose(stats["p25"], 6.5 / 32.0))


def _assert_label_free(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    for banned in BANNED:
        if banned in source:
            raise AssertionError(f"{path.name} contains {banned}")
    _assert_imports(ast.parse(source), path.name)


def _assert_imports(tree: ast.AST, name: str) -> None:
    for node in ast.walk(tree):
        _reject_import(node, name)


def _reject_import(node: ast.AST, name: str) -> None:
    if not isinstance(node, ast.ImportFrom):
        return
    module = node.module or ""
    if module in {"evaluate", "devharness", "cells", "dataset", "develop"}:
        raise AssertionError(f"{name} imports {module}")


def _assert_volume_ignores_an_unused_array() -> None:
    plane = np.full((8, 8), 0.2, dtype=np.float32)
    zeros = np.zeros((8, 8), dtype=np.float32)
    unused = np.array([1, 0, 1], dtype=np.int8)
    first = context_volume(plane, plane, zeros, zeros)
    unused[0] = 0
    second = context_volume(plane, plane, zeros, zeros)
    if first.shape != (8, 8, len(feature_names())):
        raise AssertionError(first.shape)
    if not np.allclose(first[:, :, 0], 0.2):
        raise AssertionError("stage2 channel")
    if not np.allclose(first, second):
        raise AssertionError("feature output changed without a map changing")
    if not np.isfinite(first).all():
        raise AssertionError("non-finite features")


def _assert_half_widths() -> None:
    expected = {24: 1, 48: 3, 96: 6, 192: 12, 384: 24}
    for window in WINDOWS_PX:
        if half_cells(window) != expected[window]:
            raise AssertionError((window, half_cells(window)))


def _tiny_volume() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.random((5, 4, 3), dtype=np.float32)


def _tiny_model(volume: np.ndarray):
    target = np.zeros(volume.shape[0] * volume.shape[1], dtype=np.int32)
    target[:7] = 1
    model = HistGradientBoostingClassifier(
        max_iter=8, max_leaf_nodes=4, early_stopping=False, random_state=0
    )
    model.fit(volume.reshape(-1, 3), target)
    return model


def _unknown_fixture():
    positive = np.zeros((16, 16), dtype=bool)
    known = np.zeros((16, 16), dtype=bool)
    positive[4, 4] = True
    known[4, 4] = True
    known[4, 12] = True
    return positive, known, np.array([4, 8]), np.array([4, 12])


if __name__ == "__main__":
    unittest.main()
