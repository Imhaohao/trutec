"""Tests for learned texture representation, model persistence, and inference boundary."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from typing import Any
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import hatch_texture
import numpy as np
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier
import sklearn

from evaluate import EvaluationError
from inference import read_requests
from inference_texture import (
    interpolate_and_threshold_mask,
    process_query_group,
    verify_model_metadata,
)
from texture_learning import (
    grid_centers,
    load_texture_model,
    pair_feature_names,
    pair_features,
    sample_grid_domain_centers,
    save_texture_model,
)
from train_texture import (
    train_classifier,
    validate_split_safety,
)


class TestLearningRepresentation(unittest.TestCase):
    def test_pair_similarity_finite_and_symmetric_difference(self) -> None:
        c = 4
        vectors = np.array([
            [0.0, 0.0, 0.0, 0.0],
            [1e-5, 0.5, 2.0, 10.0],
            [0.2, 0.4, 0.6, 0.8],
        ], dtype=np.float32)
        query = np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float32)

        pairs = pair_features(vectors, query)
        self.assertEqual(pairs.shape, (3, 4 * c))
        self.assertEqual(pairs.dtype, np.float32)
        self.assertTrue(np.isfinite(pairs).all())

        # Exact match vector (row 2) has zero difference
        abs_diff_match = pairs[2, 2 * c : 3 * c]
        sym_diff_match = pairs[2, 3 * c : 4 * c]
        np.testing.assert_allclose(abs_diff_match, 0.0, atol=1e-6)
        np.testing.assert_allclose(sym_diff_match, 0.0, atol=1e-6)

        # Symmetry: relative difference of (v, q) must equal relative difference of (q, v)
        single_v = vectors[1:2]
        pair_vq = pair_features(single_v, query)
        pair_qv = pair_features(query[np.newaxis, :], single_v[0])
        sym_diff_vq = pair_vq[0, 3 * c : 4 * c]
        sym_diff_qv = pair_qv[0, 3 * c : 4 * c]
        np.testing.assert_allclose(sym_diff_vq, sym_diff_qv, rtol=1e-5, atol=1e-6)

        # Rejection of invalid inputs
        with self.assertRaises(ValueError):
            pair_features(np.array([[1.0, np.nan]]), np.array([1.0, 1.0]))
        with self.assertRaises(ValueError):
            pair_features(np.array([[1.0, 1.0]]), np.array([1.0, np.inf]))
        with self.assertRaises(ValueError):
            pair_features(vectors, np.array([1.0, 2.0]))  # dimension mismatch

    def test_deterministic_grid_sampling_never_picks_unknown_or_query_domains(self) -> None:
        h, w, stride = 80, 80, 4
        pos_domain = np.zeros((h, w), dtype=bool)
        pos_domain[40:60, 10:30] = True

        known_domain = np.zeros((h, w), dtype=bool)
        known_domain[40:60, 10:30] = True
        known_domain[40:60, 40:60] = True
        neg_domain = known_domain & ~pos_domain

        # Query box inside known negative area (mimicking reviewer exclusion)
        query_box = (44, 44, 52, 52)
        neg_domain[query_box[1]:query_box[3], query_box[0]:query_box[2]] = False

        pos_sampled, neg_sampled = sample_grid_domain_centers(
            h, w, pos_domain, neg_domain, stride=stride, samples_per_class=30, seed=42
        )

        self.assertGreater(len(pos_sampled), 0)
        self.assertGreater(len(neg_sampled), 0)

        centers_y, centers_x = grid_centers(h, w, stride=stride)

        for gy, gx in pos_sampled:
            cy, cx = centers_y[gy], centers_x[gx]
            self.assertTrue(pos_domain[cy, cx], "Sampled positive center outside positive domain")
            self.assertTrue(known_domain[cy, cx], "Sampled positive center in unknown domain")

        for gy, gx in neg_sampled:
            cy, cx = centers_y[gy], centers_x[gx]
            self.assertTrue(neg_domain[cy, cx], "Sampled negative center outside negative domain")
            self.assertFalse(pos_domain[cy, cx], "Sampled negative center inside positive domain")
            self.assertTrue(known_domain[cy, cx], "Sampled negative center in unknown domain")
            in_query = (query_box[0] <= cx < query_box[2]) and (query_box[1] <= cy < query_box[3])
            self.assertFalse(in_query, "Sampled negative center inside query box")

        # Determinism check
        pos2, neg2 = sample_grid_domain_centers(
            h, w, pos_domain, neg_domain, stride=stride, samples_per_class=30, seed=42
        )
        np.testing.assert_array_equal(pos_sampled, pos2)
        np.testing.assert_array_equal(neg_sampled, neg2)

    def test_injected_annotation_fields_rejected_at_inference_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_req = {
                "id": "q_test",
                "image": "synthetic.png",
                "width": 32,
                "height": 32,
                "query_box": [4, 4, 12, 12],
                "context_boxes": [],
            }
            # Injected labels
            injected_labels = dict(base_req)
            injected_labels["labels"] = {"positive_boxes": [[16, 16, 24, 24]]}
            path1 = Path(tmpdir) / "req_labels.json"
            path1.write_text(json.dumps({"schema": "hatch-matching-inputs/v1", "examples": [injected_labels]}))
            with self.assertRaises(EvaluationError):
                read_requests(path1)

            # Injected document_id
            injected_doc = dict(base_req)
            injected_doc["document_id"] = "secret-doc-123"
            path2 = Path(tmpdir) / "req_doc.json"
            path2.write_text(json.dumps({"schema": "hatch-matching-inputs/v1", "examples": [injected_doc]}))
            with self.assertRaises(EvaluationError):
                read_requests(path2)

            # Invalid schema
            path3 = Path(tmpdir) / "req_schema.json"
            path3.write_text(json.dumps({"schema": "wrong-schema/v1", "examples": [base_req]}))
            with self.assertRaises(EvaluationError):
                read_requests(path3)

    def test_tiny_end_to_end_fitted_model_writes_native_binary_png_and_rejects_feature_mismatch(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            base_names = hatch_texture.texture_feature_names()
            c = len(base_names)

            # Train a tiny model
            rng = np.random.default_rng(123)
            x_syn = rng.random((20, 4 * c), dtype=np.float32)
            y_syn = np.array([0, 1] * 10, dtype=np.int32)
            clf = HistGradientBoostingClassifier(max_iter=10, random_state=42)
            clf.fit(x_syn, y_syn)

            model_path = tmp_path / "model.joblib"
            metadata = {
                "stride": 4,
                "base_feature_names": base_names,
                "pair_feature_names": pair_feature_names(base_names),
                "source_manifest_sha256": "dummy_sha256",
                "training_ids": ["q_syn"],
                "training_documents": ["doc_syn"],
                "sklearn_version": sklearn.__version__,
                "numpy_version": np.__version__,
            }
            save_texture_model(clf, metadata, model_path)

            loaded_clf, loaded_meta = load_texture_model(model_path)
            verify_model_metadata(loaded_meta)

            # Feature signature mismatch check
            corrupted_meta = dict(loaded_meta)
            corrupted_meta["base_feature_names"] = ["incompatible_feat_0", "incompatible_feat_1"]
            with self.assertRaises(EvaluationError):
                verify_model_metadata(corrupted_meta)

            # End-to-end inference execution writing native binary PNG
            img_shape = (48, 48)
            gray = np.full(img_shape, 200, dtype=np.uint8)
            query_group = [{
                "id": "query_e2e",
                "image": "unused.png",
                "width": 48,
                "height": 48,
                "query_box": [4, 4, 16, 16],
                "context_boxes": [],
            }]
            records = process_query_group(
                loaded_clf, gray, query_group, tmp_path, stride=4, threshold=0.5
            )
            self.assertEqual(len(records), 1)

            out_png = tmp_path / "query_e2e.png"
            self.assertTrue(out_png.is_file())
            with Image.open(out_png) as img:
                self.assertEqual(img.size, (48, 48))
                self.assertEqual(img.mode, "L")
                vals = set(int(v) for v in np.unique(np.asarray(img)))
                self.assertTrue(vals <= {0, 255})

    def test_odd_dimensions_preserve_correct_alignment(self) -> None:
        h, w, stride = 17, 13, 4
        centers_y, centers_x = grid_centers(h, w, stride=stride)

        # Expected grid sizes: ceil(17 / 4) = 5, ceil(13 / 4) = 4
        self.assertEqual(len(centers_y), 5)
        self.assertEqual(len(centers_x), 4)

        # Centers must never exceed image boundary
        self.assertLess(int(centers_y[-1]), h)
        self.assertLess(int(centers_x[-1]), w)
        self.assertEqual(int(centers_y[-1]), 16)
        self.assertEqual(int(centers_x[-1]), 12)

        # Test interpolation with odd dimensions and non-divisible stripe height
        prob_grid = np.zeros((5, 4), dtype=np.float32)
        prob_grid[2, 1] = 1.0  # cell (gy=2, gx=1), center cy=10, cx=6

        mask_stripe6 = interpolate_and_threshold_mask(
            prob_grid, h, w, stride=stride, threshold=0.5, stripe_height=6
        )
        mask_stripe3 = interpolate_and_threshold_mask(
            prob_grid, h, w, stride=stride, threshold=0.5, stripe_height=3
        )

        self.assertEqual(mask_stripe6.shape, (17, 13))
        self.assertEqual(mask_stripe6.dtype, bool)
        self.assertTrue(mask_stripe6[10, 6])
        self.assertFalse(mask_stripe6[0, 0])
        # Stripe size invariance
        np.testing.assert_array_equal(mask_stripe6, mask_stripe3)

        edge_grid = np.zeros((5, 4), dtype=np.float32)
        edge_grid[-1, -1] = 1.0
        edge_mask = interpolate_and_threshold_mask(edge_grid, h, w, stride, threshold=0.99)
        self.assertTrue(edge_mask[-1, -1], 'Last clamped grid center must retain its own probability')


class TestTrainingSafetyAndWorkflow(unittest.TestCase):
    def test_train_texture_split_safety_rejects_val_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            val_path = tmp / "val.json"
            val_path.write_text(json.dumps({"examples": []}))
            with self.assertRaises(EvaluationError):
                validate_split_safety(val_path, tmp, [])

    def test_train_texture_split_safety_rejects_shared_doc_groups(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            val_path = tmp / "val.json"
            val_data = {
                "examples": [
                    {"document_id": "doc-shared-val-001"},
                    {"document_id": "doc-shared-val-002"},
                ]
            }
            val_path.write_text(json.dumps(val_data))

            train_rows_bad = [
                {"id": "q1", "document_id": "doc-shared-val-001"},
                {"id": "q2", "document_id": "doc-train-clean"},
            ]
            train_path = tmp / "train.json"
            train_path.write_text(json.dumps({"examples": train_rows_bad}))

            with self.assertRaises(EvaluationError):
                validate_split_safety(train_path, tmp, train_rows_bad)

            train_rows_clean = [
                {"id": "q3", "document_id": "doc-train-clean-1"},
                {"id": "q4", "document_id": "doc-train-clean-2"},
            ]
            # Should not raise
            validate_split_safety(train_path, tmp, train_rows_clean)

    def test_train_classifier_parameters(self) -> None:
        x = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]], dtype=np.float32)
        y = np.array([0, 0, 1, 1], dtype=np.int32)
        weights = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)

        clf = train_classifier(x, y, weights, max_iter=50, seed=42)
        self.assertEqual(clf.learning_rate, 0.08)
        self.assertEqual(clf.max_leaf_nodes, 31)
        self.assertEqual(clf.l2_regularization, 1.0)
        self.assertEqual(clf.max_iter, 50)
        self.assertFalse(clf.early_stopping)
        self.assertEqual(clf.random_state, 42)


if __name__ == "__main__":
    unittest.main()
