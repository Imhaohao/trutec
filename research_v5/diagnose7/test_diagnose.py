"""Unit tests for v7 false positive diagnosis module."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bootstrap  # noqa: E402,F401
import unittest  # noqa: E402

import numpy as np  # noqa: E402

from analysis import (
    QueryStat,
    _aggregate_doc_stats,
    _compute_fp_distances,
    _is_glyph_component,
    _single_query_stat,
    analyze_spatial_distances,
    compute_concentration,
)
from visualize import _compose_side_panel, _draw_legend, _resize_crop


class TestDiagnosis(unittest.TestCase):
    def test_confusion_matrix_and_metrics(self):
        # 10 reviewed pixels: 4 positives, 6 negatives
        # Predictions: 3 TP, 1 FN, 2 FP, 4 TN
        scores = np.array([0.9, 0.8, 0.7, 0.1, 0.6, 0.5, 0.2, 0.1, 0.0, 0.1], dtype=np.float32)
        positives = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=bool)

        class DummyQuery:
            id = "q-test"
            document_id = "doc-test"
            image = "test.png"
            width = 100
            height = 100
            query_box = (10, 10, 20, 20)
            positive = positives
            y = np.arange(10)
            x = np.arange(10)

        stat = _single_query_stat(DummyQuery(), scores, threshold=0.25)
        self.assertEqual(stat.tp, 3)
        self.assertEqual(stat.fn, 1)
        self.assertEqual(stat.fp, 2)
        self.assertEqual(stat.tn, 4)

        expected_iou = 3 / (3 + 2 + 1)
        self.assertAlmostEqual(stat.iou, expected_iou)
        self.assertAlmostEqual(stat.recall, 3 / 4)
        self.assertAlmostEqual(stat.precision, 3 / 5)
        self.assertAlmostEqual(stat.fpr, 2 / 6)

    def test_spatial_distances(self):
        # Create a grid where TP is at (50, 50)
        # FP1 is at (50, 60) -> distance 10 px (boundary bleed <= 24px)
        # FP2 is at (50, 90) -> distance 40 px (far FP > 24px)
        y = np.array([50, 50, 50], dtype=np.int32)
        x = np.array([50, 60, 90], dtype=np.int32)
        scores = np.array([0.9, 0.8, 0.8], dtype=np.float32)
        positive = np.array([True, False, False], dtype=bool)

        stat = QueryStat(
            id="q-dist",
            document_id="doc-dist",
            image="test.png",
            width=200,
            height=200,
            query_box=(0, 0, 10, 10),
            tp=1,
            fp=2,
            fn=0,
            tn=0,
            iou=1 / 3,
            recall=1.0,
            precision=1 / 3,
            fpr=1.0,
            scores=scores,
            decision=scores >= 0.25,
            positive=positive,
            y=y,
            x=x,
        )

        dists = _compute_fp_distances(stat, stat.decision & positive, stat.decision & ~positive)
        self.assertEqual(len(dists), 2)
        self.assertAlmostEqual(dists[0], 10.0, places=4)
        self.assertAlmostEqual(dists[1], 40.0, places=4)

        recs = analyze_spatial_distances([stat], top_k=1, distance_px=24.0)
        self.assertEqual(recs[0]["boundary_bleed_near_24px"], 1)
        self.assertEqual(recs[0]["far_fp_over_24px"], 1)
        self.assertAlmostEqual(recs[0]["boundary_bleed_fraction"], 0.5)

    def test_concentration_math(self):
        # 3 queries with FP counts 500, 300, 200 (total = 1000)
        query_stats = [
            QueryStat(
                id=f"q-{i}",
                document_id="doc-1" if i < 2 else "doc-2",
                image="test.png",
                width=10,
                height=10,
                query_box=(0, 0, 5, 5),
                tp=10,
                fp=fp,
                fn=0,
                tn=100,
                iou=0.5,
                recall=1.0,
                precision=0.5,
                fpr=0.1,
                scores=np.array([0.5]),
                decision=np.array([True]),
                positive=np.array([True]),
                y=np.array([0]),
                x=np.array([0]),
            )
            for i, fp in enumerate([500, 300, 200])
        ]
        doc_stats = [
            {"document_id": "doc-1", "fp": 800, "tp": 20},
            {"document_id": "doc-2", "fp": 200, "tp": 10},
        ]
        conc = compute_concentration(query_stats, doc_stats)
        self.assertEqual(conc["total_fp"], 1000)
        self.assertAlmostEqual(conc["top_queries_fp"][1]["share"], 0.5)
        self.assertAlmostEqual(conc["top_queries_fp"][3]["share"], 1.0)
        self.assertAlmostEqual(conc["top_docs_fp"][1]["share"], 0.8)

    def test_glyph_component_filter(self):
        # cv2 stats row format: [x, y, width, height, area]
        # Text glyph: w<=45, h<=45, area<=700, aspect<=2.5
        glyph_stat = np.array([10, 10, 20, 25, 300], dtype=np.int32)
        line_stat = np.array([10, 10, 500, 2, 800], dtype=np.int32)

        self.assertTrue(_is_glyph_component(glyph_stat))
        self.assertFalse(_is_glyph_component(line_stat))

    def test_visual_side_panel_rendering(self):
        stat = QueryStat(
            id="q-viz",
            document_id="doc-viz",
            image="test.png",
            width=50,
            height=50,
            query_box=(0, 0, 20, 20),
            tp=100,
            fp=200,
            fn=10,
            tn=500,
            iou=0.32,
            recall=0.91,
            precision=0.33,
            fpr=0.28,
            scores=np.array([0.5]),
            decision=np.array([True]),
            positive=np.array([True]),
            y=np.array([0]),
            x=np.array([0]),
        )
        orig = np.zeros((30, 30, 3), dtype=np.uint8)
        cleaned = np.ones((30, 30, 3), dtype=np.uint8) * 255
        panel = _compose_side_panel(stat, orig, cleaned)
        self.assertEqual(panel.size, (400, 700))


if __name__ == "__main__":
    unittest.main()
