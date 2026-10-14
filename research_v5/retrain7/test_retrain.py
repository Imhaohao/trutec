"""CPU checks for weights, the fit-CV operating point, and the label-free request."""
from __future__ import annotations

import unittest

import runtime

runtime.activate()

import numpy as np

from cell_metrics import operating_point, prefer
from hardneg import apply_hard_negative


class RetrainTests(unittest.TestCase):
    def test_weight_three_matches_structure_and_weight_five_is_five(self) -> None:
        from weights import apply_hard_negative as structure_apply

        labels = np.array([1, 1, 0, 0, 0], dtype=np.int8)
        documents = np.zeros(5, dtype=np.int32)
        stage1 = np.array([0.2, 0.2, 0.9, 0.1, 0.8], dtype=np.float32)
        base = np.array([0.25, 0.25, 0.5 / 3, 0.5 / 3, 0.5 / 3], dtype=np.float64)
        frozen = structure_apply(base, labels, stage1, documents, 0.5)
        ours = apply_hard_negative(base, labels, stage1, documents, 0.5, 3.0)
        self.assertTrue(np.allclose(ours, frozen))
        boosted = apply_hard_negative(base, labels, stage1, documents, 0.5, 5.0)
        self.assertAlmostEqual(float(boosted[2] / boosted[3]), 5.0)
        self.assertAlmostEqual(float(boosted[labels == 0].sum()), 0.5)
        self.assertAlmostEqual(float(boosted[labels == 1].sum()), 0.5)

    def test_recall_floor_keeps_the_higher_macro_and_precision_breaks_ties(self) -> None:
        queries = [_query([0.9, 0.4, 0.7, 0.2], [1, 1, 0, 0], "d1")]
        point = operating_point(queries, 0.95, [0.3, 0.8])
        self.assertEqual(point["threshold"], 0.3)
        self.assertAlmostEqual(point["macro_iou"], 2.0 / 3.0)
        self.assertAlmostEqual(point["recall"], 1.0)
        low = {"macro_iou": 0.5, "precision": 0.4}
        high = {"macro_iou": 0.5, "precision": 0.9}
        better_macro = {"macro_iou": 0.8, "precision": 0.1}
        self.assertTrue(prefer(high, low))
        self.assertFalse(prefer(low, high))
        self.assertTrue(prefer(better_macro, high))

    def test_double_sample_keeps_domain_counts(self) -> None:
        from dataset import _sample_cells
        from texture_learning import grid_centers

        positive = np.zeros((80, 80), dtype=bool)
        known = np.zeros((80, 80), dtype=bool)
        positive[8:70, 8:70] = True
        known[4:76, 4:76] = True
        centers_y, centers_x = grid_centers(80, 80, 8)
        base = _sample_cells(positive, known, centers_y, centers_x, 5, 3)
        double = _sample_cells(positive, known, centers_y, centers_x, 10, 3)
        self.assertEqual(base[2], double[2])
        self.assertEqual(base[3], double[3])
        self.assertEqual(len(double[0]), 10)
        self.assertEqual(len(base[0]), 5)
        self.assertGreaterEqual(len(double[1]), len(base[1]))

    def test_g2_descriptor_width_and_request_drops_labels(self) -> None:
        from types import SimpleNamespace

        from features import feature_names
        from guards import descriptor_for
        from hatch_texture import texture_feature_names
        from score_dev import request_from

        gray = np.full((48, 48), 240, np.uint8)
        gray[10:30, 10:14] = 20
        gray[10:30, 24:28] = 20
        descriptor = descriptor_for("g2", gray, (8, 8, 40, 40))
        self.assertEqual(descriptor.shape, (len(texture_feature_names()),))
        self.assertEqual(descriptor.dtype, np.float32)
        self.assertTrue(np.isfinite(descriptor).all())
        self.assertEqual(feature_names()[0], "cell_score")
        self.assertEqual(len(feature_names()), 75)
        query = SimpleNamespace(
            id="q-1",
            image="a.png",
            width=4,
            height=5,
            query_box=(0, 0, 1, 1),
            context_boxes=((1, 1, 2, 2),),
            document_id="doc",
            positive=np.array([1]),
        )
        request = request_from(query)
        self.assertEqual(set(request), {"id", "image", "width", "height", "query_box", "context_boxes"})


def _query(scores, labels, document_id: str) -> dict:
    values = np.asarray(scores, dtype=np.float32)
    target = np.asarray(labels, dtype=np.int8)
    return {"document_id": document_id, "labels": target, "scores": values, "pop": np.ones(len(target))}


if __name__ == "__main__":
    unittest.main()
