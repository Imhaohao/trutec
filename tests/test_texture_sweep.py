"""Tests for fast exact training-development threshold sweep."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from typing import Any
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier

from evaluate import EvaluationError, evaluate, load_manifest, reviewed_domains
from evaluate_texture_sweep import (
    DEFAULT_THRESHOLDS,
    build_sweep_report,
    find_required_grid_indices,
    predict_unique_grid_probabilities,
    remap_known_coordinates,
    run_sweep,
    score_known_probabilities,
    score_query_thresholds,
    validate_split_safety,
)
import hatch_texture
from inference import save_prediction
from inference_texture import (
    interpolate_and_threshold_mask,
    native_grid_coordinates,
    predict_probability_grid,
)
from texture_learning import pair_features, save_texture_model


class DeterministicPredictor:
    """Predictor returning deterministic smooth probabilities without training."""

    def __init__(self, num_features: int, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        self.weights = rng.standard_normal(num_features).astype(np.float32) * 0.1

    def predict_proba(self, pairs: np.ndarray) -> np.ndarray:
        raw = np.tanh(pairs @ self.weights)
        prob_pos = 0.5 * (raw + 1.0)
        return np.column_stack([1.0 - prob_pos, prob_pos])


class TrackingPredictor:
    """Wrapper recording all vector batches queried from predict_proba."""

    def __init__(self, base_model: Any) -> None:
        self.base_model = base_model
        self.total_vectors: int = 0
        self.call_count: int = 0

    def predict_proba(self, pairs: np.ndarray) -> np.ndarray:
        self.call_count += 1
        self.total_vectors += len(pairs)
        return self.base_model.predict_proba(pairs)


def create_tiny_classifier(num_channels: int) -> HistGradientBoostingClassifier:
    """Train a tiny HistGradientBoostingClassifier on synthetic pairs."""
    pair_dim = 4 * num_channels
    rng = np.random.default_rng(123)
    x_train = rng.uniform(0.0, 1.0, (120, pair_dim)).astype(np.float32)
    y_train = (x_train[:, 0] > 0.5).astype(int)
    clf = HistGradientBoostingClassifier(max_iter=5, random_state=123)
    clf.fit(x_train, y_train)
    return clf


def compute_full_reference_probabilities(
    model: Any,
    features: np.ndarray,
    query_desc: np.ndarray,
    height: int,
    width: int,
    stride: int,
) -> np.ndarray:
    """Compute native-resolution probabilities via full-grid inference."""
    prob_grid = predict_probability_grid(model, features, query_desc)
    gy_native, gx_native = native_grid_coordinates(height, width, stride)
    map_x, map_y = np.meshgrid(gx_native, gy_native)
    full_remap = cv2.remap(
        prob_grid,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return full_remap.astype(np.float32)


class TestTextureSweepEquivalence(unittest.TestCase):
    def test_numerical_equivalence_on_odd_dimensions_and_stripe_boundaries(self) -> None:
        """Prove exact numerical equivalence between sparse known scoring and full-grid remap."""
        # Test odd dimensions non-divisible by stride (67x103 with stride 4)
        # and tall images (520x73) crossing the 256-row stripe boundary.
        cases = [
            (67, 103, 4),
            (520, 73, 4),
        ]
        num_channels = len(hatch_texture.texture_feature_names())
        model = DeterministicPredictor(4 * num_channels)

        for height, width, stride in cases:
            grid_h = (height + stride - 1) // stride
            grid_w = (width + stride - 1) // stride
            rng = np.random.default_rng(height * width)

            features = rng.uniform(0.05, 0.95, (grid_h, grid_w, num_channels)).astype(np.float32)
            query_desc = rng.uniform(0.05, 0.95, num_channels).astype(np.float32)

            full_probs = compute_full_reference_probabilities(
                model, features, query_desc, height, width, stride
            )

            # 1. Sparse random known mask
            sparse_known = rng.uniform(0.0, 1.0, (height, width)) < 0.10
            # Ensure some edge and boundary pixels are marked known
            sparse_known[0, 0] = True
            sparse_known[height - 1, width - 1] = True
            sparse_known[min(256, height - 1), :] = True

            selective_probs = score_known_probabilities(
                model, features, query_desc, sparse_known, height, width, stride
            )
            expected_probs = full_probs[sparse_known]

            self.assertEqual(len(selective_probs), len(expected_probs))
            np.testing.assert_allclose(selective_probs, expected_probs, atol=1e-6)
            max_diff = float(np.max(np.abs(selective_probs - expected_probs)))
            self.assertLessEqual(max_diff, 1e-6)

    def test_all_known_and_single_edge_pixels(self) -> None:
        """Verify equivalence when all pixels are known and for solitary edge pixels."""
        height, width, stride = 35, 45, 4
        num_channels = len(hatch_texture.texture_feature_names())
        clf = create_tiny_classifier(num_channels)

        grid_h = (height + stride - 1) // stride
        grid_w = (width + stride - 1) // stride
        rng = np.random.default_rng(999)

        features = rng.uniform(0.0, 1.0, (grid_h, grid_w, num_channels)).astype(np.float32)
        query_desc = rng.uniform(0.0, 1.0, num_channels).astype(np.float32)

        full_probs = compute_full_reference_probabilities(
            clf, features, query_desc, height, width, stride
        )

        # All-known mask
        all_known = np.ones((height, width), dtype=bool)
        selective_all = score_known_probabilities(
            clf, features, query_desc, all_known, height, width, stride
        )
        np.testing.assert_allclose(selective_all, full_probs.ravel(), atol=1e-6)

        # Four extreme corner edge pixels
        corners = [(0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1)]
        for y, x in corners:
            single_known = np.zeros((height, width), dtype=bool)
            single_known[y, x] = True
            single_prob = score_known_probabilities(
                clf, features, query_desc, single_known, height, width, stride
            )
            self.assertEqual(len(single_prob), 1)
            self.assertAlmostEqual(float(single_prob[0]), float(full_probs[y, x]), places=6)

    def test_model_called_only_for_necessary_grid_neighbors(self) -> None:
        """Prove that unneeded grid points are never evaluated during sparse scoring."""
        height, width, stride = 100, 100, 4
        num_channels = len(hatch_texture.texture_feature_names())
        base_model = DeterministicPredictor(4 * num_channels)
        tracking = TrackingPredictor(base_model)

        grid_h = (height + stride - 1) // stride
        grid_w = (width + stride - 1) // stride
        features = np.ones((grid_h, grid_w, num_channels), dtype=np.float32) * 0.5
        query_desc = np.ones(num_channels, dtype=np.float32) * 0.5

        # Single pixel at (0, 0) maps to clamped center (stride//2, stride//2).
        # In grid coordinates, gy=0.0 and gx=0.0. Floor and ceil are both 0.
        # Only 1 unique grid cell (0, 0) must be evaluated.
        single_edge = np.zeros((height, width), dtype=bool)
        single_edge[0, 0] = True

        score_known_probabilities(
            tracking, features, query_desc, single_edge, height, width, stride
        )
        self.assertEqual(tracking.total_vectors, 1)

        # Interior pixel at (stride, stride) has non-integer gy and gx strictly in (0, 1).
        # Bounding corners are (0,0), (0,1), (1,0), (1,1) -> exactly 4 unique cells.
        tracking = TrackingPredictor(base_model)
        interior = np.zeros((height, width), dtype=bool)
        interior[stride, stride] = True

        score_known_probabilities(
            tracking, features, query_desc, interior, height, width, stride
        )
        self.assertEqual(tracking.total_vectors, 4)

    def test_chunked_remap_across_chunk_boundary(self) -> None:
        """Verify chunked remap matches full remap across the 16384-vector boundary."""
        grid_h, grid_w = 40, 40
        prob_grid = np.random.uniform(0.0, 1.0, (grid_h, grid_w)).astype(np.float32)

        n_points = 25000
        sample_gx = np.random.uniform(0, grid_w - 1, n_points).astype(np.float32)
        sample_gy = np.random.uniform(0, grid_h - 1, n_points).astype(np.float32)

        chunked = remap_known_coordinates(prob_grid, sample_gx, sample_gy, chunk_size=8192)

        # Compare directly against OpenCV remap on sub-chunks
        full_ref = np.empty(n_points, dtype=np.float32)
        for i in range(0, n_points, 10000):
            end = min(i + 10000, n_points)
            cx = sample_gx[i:end].reshape(-1, 1)
            cy = sample_gy[i:end].reshape(-1, 1)
            sub = cv2.remap(
                prob_grid, cx, cy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
            )
            full_ref[i:end] = sub.ravel()

        np.testing.assert_allclose(chunked, full_ref, atol=1e-6)


class TestTextureSweepOfficialAlignment(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_selected_threshold_counts_match_upstream_evaluate_on_written_png(self) -> None:
        """Prove exact count and metric equivalence between sweep and upstream evaluate.py."""
        height, width, stride = 48, 64, 4
        threshold = 0.50

        # Create synthetic grayscale drawing
        drawing = np.full((height, width), 200, dtype=np.uint8)
        drawing[10:30, 10:30] = 50  # darker pattern
        img_path = self.root / "drawing.png"
        cv2.imwrite(str(img_path), drawing)

        # Labeled positive mask
        pos_mask = np.zeros((height, width), dtype=np.uint8)
        pos_mask[12:28, 12:28] = 255
        pos_path = self.root / "pos.png"
        cv2.imwrite(str(pos_path), pos_mask)

        # Labeled blank mask (reviewed negative region)
        blank_mask = np.zeros((height, width), dtype=np.uint8)
        blank_mask[35:45, 35:45] = 255
        blank_path = self.root / "blank.png"
        cv2.imwrite(str(blank_path), blank_mask)

        # Reviewed known mask
        known_mask = np.zeros((height, width), dtype=np.uint8)
        known_mask[10:46, 10:46] = 255
        known_path = self.root / "known.png"
        cv2.imwrite(str(known_path), known_mask)

        row = {
            "id": "query_test_01",
            "document_id": "doc_dev_01",
            "image": "drawing.png",
            "kind": "hatch",
            "width": width,
            "height": height,
            "query_box": [14, 14, 22, 22],
            "labels": {
                "positive_mask": "pos.png",
                "blank_mask": "blank.png",
                "known_mask": "known.png",
            },
        }
        manifest_data = {"examples": [row]}
        manifest_path = self.root / "train_dev.json"
        manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")

        # Train a tiny classifier
        num_channels = len(hatch_texture.texture_feature_names())
        clf = create_tiny_classifier(num_channels)
        metadata = {
            "stride": stride,
            "base_feature_names": hatch_texture.texture_feature_names(),
            "pair_feature_names": [f"f{i}" for i in range(4 * num_channels)],
            "source_manifest_sha256": "0" * 64,
            "training_ids": ["q1"],
            "training_documents": ["doc1"],
            "sklearn_version": "1.0",
            "numpy_version": "1.0",
        }
        model_path = self.root / "model.joblib"
        save_texture_model(clf, metadata, model_path)

        # 1. Full inference writing native PNG
        features = hatch_texture.texture_features(drawing, stride=stride)
        qb = row["query_box"]
        assert isinstance(qb, list)
        query_box = (int(qb[0]), int(qb[1]), int(qb[2]), int(qb[3]))
        query_desc = hatch_texture.query_descriptor(
            features, query_box, drawing.shape, stride=stride
        )
        prob_grid = predict_probability_grid(clf, features, query_desc)
        full_mask = interpolate_and_threshold_mask(prob_grid, height, width, stride, threshold)

        pred_dir = self.root / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        save_prediction(pred_dir / "query_test_01.png", full_mask)

        # 2. Evaluate with upstream evaluate.py
        eval_result = evaluate(manifest_path, self.root, pred_dir)
        upstream_metrics = eval_result["examples"][0]["metrics"]

        # 3. Score with evaluate_texture_sweep logic
        positive, known, blank, meta = reviewed_domains(row, self.root)
        known_probs = score_known_probabilities(
            clf, features, query_desc, known, height, width, stride
        )
        sweep_scores = score_query_thresholds(
            known_probs, positive, known, blank, row, meta, [threshold]
        )
        sweep_metrics = sweep_scores[str(threshold)]["metrics"]

        # Assert exact equality of all counts and metrics
        for key in (
            "tp",
            "fp",
            "fn",
            "tn",
            "blank_false_positive_pixels",
            "reviewed_blank_pixels",
            "iou",
            "precision",
            "recall",
            "blank_false_positive_rate",
        ):
            self.assertEqual(
                sweep_metrics[key],
                upstream_metrics[key],
                f"Mismatch in metric {key}: sweep={sweep_metrics[key]} vs upstream={upstream_metrics[key]}",
            )

    def test_split_safety_blocks_validation_data(self) -> None:
        """Verify that split safety strictly rejects val.json and validation documents."""
        val_path = self.root / "val.json"
        val_path.write_text(json.dumps({"examples": [{"document_id": "val_doc_secret"}]}))

        # Rejecting manifest named val.json
        with self.assertRaises(EvaluationError):
            validate_split_safety(val_path, self.root, [])

        # Rejecting development manifest containing validation document ID
        dev_manifest = self.root / "dev.json"
        dev_manifest.write_text(json.dumps({"examples": []}))
        leaking_rows = [{"id": "bad_query", "document_id": "val_doc_secret"}]
        with self.assertRaises(EvaluationError):
            validate_split_safety(dev_manifest, self.root, leaking_rows)

    def test_run_sweep_and_report_metadata(self) -> None:
        """Verify incremental logging and sweep.json structure."""
        height, width, stride = 40, 40, 4
        drawing = np.full((height, width), 220, dtype=np.uint8)
        cv2.imwrite(str(self.root / "img.png"), drawing)

        pos_mask = np.zeros((height, width), dtype=np.uint8)
        pos_mask[10:20, 10:20] = 255
        cv2.imwrite(str(self.root / "pos.png"), pos_mask)

        row = {
            "id": "query_sweep_01",
            "document_id": "doc_dev_99",
            "image": "img.png",
            "kind": "hatch",
            "width": width,
            "height": height,
            "query_box": [5, 5, 15, 15],
            "labels": {"positive_mask": "pos.png"},
        }
        manifest_path = self.root / "train.json"
        manifest_path.write_text(json.dumps([row]), encoding="utf-8")

        num_channels = len(hatch_texture.texture_feature_names())
        clf = create_tiny_classifier(num_channels)
        metadata = {"stride": stride, "base_feature_names": hatch_texture.texture_feature_names()}
        model_path = self.root / "model.joblib"
        save_texture_model(clf, {
            **metadata,
            "pair_feature_names": [f"f{i}" for i in range(4 * num_channels)],
            "source_manifest_sha256": "0" * 64,
            "training_ids": ["q1"],
            "training_documents": ["doc1"],
            "sklearn_version": "1.0",
            "numpy_version": "1.0",
        }, model_path)

        output_dir = self.root / "sweep_out"
        output_dir.mkdir(parents=True, exist_ok=True)

        records = run_sweep(
            [row], self.root, clf, metadata, output_dir, thresholds=[0.10, 0.50, 0.90]
        )
        self.assertEqual(len(records), 1)

        # Check incremental JSONL
        jsonl_path = output_dir / "query-records.jsonl"
        self.assertTrue(jsonl_path.is_file())
        logged = [json.loads(line) for line in jsonl_path.read_text().splitlines()]
        self.assertEqual(len(logged), 1)
        self.assertEqual(logged[0]["id"], "query_sweep_01")

        # Check report generation
        summaries = {"0.5": {"nonempty_target_examples": 1, "empty_target_examples": 0, "positive_only_examples": 1}}
        report = build_sweep_report([row], manifest_path, model_path, summaries, 1.23, [0.5])
        self.assertEqual(report["method"], "learned-texture-sweep-v1")
        self.assertEqual(report["queries"], 1)
        self.assertEqual(report["real_document_count"], 1)
        self.assertFalse(report["whole_sheet_accuracy_certified"])
        self.assertTrue(report["exact_known_scoring_only"])
        self.assertIn("manifest_sha256", report)
        self.assertIn("model_sha256", report)
        self.assertIn("implementation_sha256", report)


if __name__ == "__main__":
    unittest.main()
