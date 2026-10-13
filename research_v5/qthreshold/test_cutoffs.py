"""Unit tests for cutoff targets and the small regressors."""
from __future__ import annotations

import math
import unittest
from pathlib import Path

import limits  # noqa: F401

import numpy as np

from cutoffs import best_cutoffs, confusion_counts
from evaluate import metrics_from_counts
from regress import ScaledRidge, TreeCutoff

HERE = Path(__file__).resolve().parent


class CutoffTests(unittest.TestCase):
    def test_iou_cutoff_and_recall_cutoff_disagree(self) -> None:
        positive = np.array([0.2, 0.9])
        negative = np.array([0.3, 0.3, 0.3, 0.3, 0.3])
        iou_cutoff, recall_cutoff = best_cutoffs(positive, negative)
        self.assertAlmostEqual(iou_cutoff, 0.9)
        self.assertAlmostEqual(recall_cutoff, 0.2)

    def test_missing_positives_have_no_target_and_ties_take_the_largest(self) -> None:
        iou_cutoff, recall_cutoff = best_cutoffs(np.array([]), np.array([0.2]))
        self.assertTrue(math.isnan(iou_cutoff))
        self.assertTrue(math.isnan(recall_cutoff))
        iou_cutoff, recall_cutoff = best_cutoffs(np.array([0.5]), np.array([]))
        self.assertAlmostEqual(iou_cutoff, 0.5)
        self.assertAlmostEqual(recall_cutoff, 0.5)

    def test_confusion_counts_match_the_evaluator(self) -> None:
        counts = confusion_counts(np.array([0.8, 0.2]), np.array([0.9, 0.1]), 0.5)
        self.assertEqual(counts, (1, 1, 1, 1))
        self.assertIs(type(counts[0]), int)
        metrics = metrics_from_counts(*counts)
        self.assertAlmostEqual(metrics["iou"], 1.0 / 3.0)

    def test_regressors_fit_a_line(self) -> None:
        features = np.arange(20, dtype=np.float64).reshape(10, 2)
        target = features[:, 0] * 0.01
        weight = np.ones(10)
        for model in (ScaledRidge(1.0), TreeCutoff(3, 20)):
            model.fit(features, target, weight)
            predicted = model.predict(features)
            self.assertEqual(predicted.shape, (10,))
            self.assertTrue(np.isfinite(predicted).all())

    def test_training_modules_do_not_read_development_scores(self) -> None:
        regress_source = (HERE / "regress.py").read_text(encoding="utf-8")
        develop_source = (HERE / "develop.py").read_text(encoding="utf-8")
        self.assertNotIn("development_queries", regress_source)
        self.assertNotIn("target_iou", develop_source)
        self.assertNotIn("fit_rows", develop_source)


if __name__ == "__main__":
    unittest.main()
