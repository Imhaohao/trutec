"""Unit tests for label-free score-map features."""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

import limits  # noqa: F401

import numpy as np

from scoremap import feature_index, feature_names, map_features, otsu_threshold

HERE = Path(__file__).resolve().parent
LABEL_FREE = ("scoremap.py", "gating.py", "rules.py")
BANNED = ("reviewed_domains", "devharness", "fit.json", "development-real", "positive_mask")


class ScoreMapTests(unittest.TestCase):
    def test_query_median_ignores_the_rest_of_the_map(self) -> None:
        score = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        features = map_features(score, (0, 0, 2, 1))
        self.assertEqual(features.shape, (len(feature_names()),))
        self.assertAlmostEqual(float(features[feature_index("query_median")]), 0.0)
        self.assertAlmostEqual(float(features[feature_index("quantile_50")]), 0.5)
        self.assertAlmostEqual(float(features[feature_index("fraction_above_0.1")]), 0.5)
        self.assertAlmostEqual(float(features[feature_index("box_width")]), 2.0)
        self.assertAlmostEqual(float(features[feature_index("box_height")]), 1.0)
        self.assertAlmostEqual(float(features[feature_index("box_area")]), 2.0)

    def test_otsu_splits_two_spikes_and_keeps_a_constant_map(self) -> None:
        spikes = np.concatenate([np.full(1000, 0.2), np.full(1000, 0.8)])
        threshold = otsu_threshold(spikes)
        self.assertGreater(threshold, 0.2)
        self.assertLess(threshold, 0.8)
        self.assertAlmostEqual(float(np.mean(spikes >= threshold)), 0.5)
        constant = np.full(500, 0.2)
        constant_threshold = otsu_threshold(constant)
        self.assertLessEqual(constant_threshold, 0.2)
        self.assertAlmostEqual(float(np.mean(constant >= constant_threshold)), 1.0)

    def test_feature_modules_do_not_name_labels(self) -> None:
        for name in LABEL_FREE:
            source = (HERE / name).read_text(encoding="utf-8")
            tree = ast.parse(source)
            for banned in BANNED:
                self.assertNotIn(banned, source, name)
            self.assertIsNotNone(tree)


if __name__ == "__main__":
    unittest.main()
