"""Synthetic checks for deterministic cadence fitting data collection."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_cadence_matching as trainer
import train_texture
import texture_learning


class CadenceTrainerTests(unittest.TestCase):
    def test_grouping_seed_schedule_and_balanced_weights_match_texture_semantics(self):
        gray_a = np.where(np.indices((24, 28))[1] % 8 < 4, 0, 255).astype(np.uint8)
        gray_b = np.where(np.indices((24, 28))[0] % 8 < 4, 0, 255).astype(np.uint8)
        images = {"a.png": gray_a, "b.png": gray_b}
        rows = [self._row("a1", "a.png"), self._row("b1", "b.png"),
                self._row("a2", "a.png")]

        def image_reader(_root, row):
            return images[row["image"]]

        def domain_reader(row, _root):
            positive = np.zeros((24, 28), dtype=bool)
            known = np.zeros_like(positive)
            if row["id"] == "a1":
                positive[4:12, 4:12] = True
                known[:] = positive
            elif row["id"] == "a2":
                positive[4:12, 4:12] = True
                known[:] = True
            else:
                known[:] = True
            return positive, known, None, None

        x, labels, weights, audit = trainer.collect_training_dataset(
            rows, Path("synthetic"), image_reader, domain_reader
        )
        self.assertEqual([entry["id"] for entry in audit], ["a1", "a2", "b1"])
        self.assertEqual([entry["query_seed"] for entry in audit], [42, 43, 44])
        self.assertEqual(x.shape[1], 204)
        self.assertEqual(len(x), len(labels))
        self.assertEqual(len(labels), len(weights))
        self.assertAlmostEqual(sum(audit[0]["sample_weights"]), 1.0)
        self.assertAlmostEqual(sum(audit[1]["sample_weights"][:len(audit[1]["positive_grid_coordinates"])]), 0.5)
        self.assertAlmostEqual(sum(audit[1]["sample_weights"][len(audit[1]["positive_grid_coordinates"]):]), 0.5)
        self.assertAlmostEqual(sum(audit[2]["sample_weights"]), 1.0)
        self.assertTrue(all(entry["pair_features_shape"][1] == 204 for entry in audit))
        self.assertTrue(all(len(entry["drawing_pixel_sha256"]) == 64 for entry in audit))

    def test_classifier_delegates_locked_estimator_parameters(self):
        x = np.zeros((4, 204), dtype=np.float32)
        y = np.array([0, 1, 0, 1], dtype=np.int32)
        weights = np.full(4, 0.25, dtype=np.float32)
        sentinel = object()
        with patch.object(train_texture, "train_classifier", return_value=sentinel) as fit:
            result = trainer.train_classifier(x, y, weights)
        self.assertIs(result, sentinel)
        fit.assert_called_once_with(x, y, weights, max_iter=200, seed=42)

    def test_metadata_records_frozen_dimensions_hashes_and_query_audit(self):
        rows = [self._row("q1", "drawing.png")]
        labels = np.array([1, 0], dtype=np.int32)
        audit = [{"id": "q1", "lag_available": True, "query_seed": 42}]
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            support, resource, assets = (temp / name for name in ("s.json", "r.json", "a.json"))
            for path in (support, resource, assets):
                path.write_text("{}", encoding="utf-8")
            metadata = trainer.build_metadata("fit-hash", rows, labels, audit,
                                              support, resource, assets, "asset-map",
                                              {"drawing.png": {"sha256": "image-hash"}}, 2.0)
        self.assertEqual(metadata["schema"], "cadence-pair-classifier/v1")
        self.assertEqual(metadata["source_manifest_sha256"], "fit-hash")
        self.assertEqual(metadata["fit_manifest_sha256"], "fit-hash")
        self.assertEqual(len(metadata["base_feature_names"]), 51)
        self.assertEqual(len(metadata["pair_feature_names"]), 204)
        self.assertEqual(metadata["query_descriptor_source"], "crop")
        self.assertEqual(metadata["stride"], 4)
        self.assertEqual(metadata["query_audit"], audit)
        self.assertEqual(metadata["fit_asset_mapping"], {"drawing.png": {"sha256": "image-hash"}})
        self.assertEqual(set(metadata["source_sha256"]), {
            "hatch_texture.py", "texture_learning.py", "train_texture.py",
            "hatch_cadence.py", "cadence_matching.py", "train_cadence_matching.py",
            str(trainer.PROTOCOL_PATH),
        })

    def test_model_save_writes_matching_sidecar_digest(self):
        required = {
            "stride": 4, "base_feature_names": [], "pair_feature_names": [],
            "source_manifest_sha256": "fit", "training_ids": [], "training_documents": [],
            "sklearn_version": "test", "numpy_version": "test",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "cadence.joblib"
            digest = trainer.save_model({"kind": "synthetic"}, required, output)
            self.assertEqual(digest, hashlib.sha256(output.read_bytes()).hexdigest())
            self.assertEqual(output.with_suffix(".json").read_text(),
                             json.dumps(required, indent=2))
            self.assertEqual(output.with_suffix(".joblib.sha256").read_text(),
                             f"{digest}  {output.name}\n")
            loaded = texture_learning.joblib.load(output)
            self.assertEqual(loaded["metadata"], required)

    def test_resource_gate_accepts_frozen_producer_schema_and_references(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pilot_path = root / "cadence-resource-pilot.json"
            log_path = root / "cadence-resource-pilot.log"
            proof_path = root / "cadence-resource-pilot-proof.json"
            pilot = {
                "schema": "cadence-resource-pilot/v1", "manifest_sha256": "fit",
                "protocol_sha256": trainer.file_sha256(trainer.PROTOCOL_PATH),
                "support_sha256": "support", "all_features_finite": True,
                "source_sha256": {"hatch_cadence.py": trainer.file_sha256(Path("hatch_cadence.py"))},
                "unavailable_fallback_verified": True,
                "direct_max_abs_diff": 1e-8, "sparse_dense_max_abs_diff": 0.0,
                "feature_min": [0.1] * 5, "feature_max": [0.9] * 5,
            }
            pilot_path.write_text(json.dumps(pilot), encoding="utf-8")
            log_path.write_text("synthetic pilot log", encoding="utf-8")
            proof = {
                "schema": "cadence-resource-pilot-proof/v1", "status": "passed",
                "manifest_sha256": "fit", "protocol_sha256": trainer.file_sha256(trainer.PROTOCOL_PATH),
                "support_sha256": "support", "source_sha256": {
                    "hatch_cadence.py": trainer.file_sha256(Path("hatch_cadence.py"))},
                "peak_process_footprint_bytes": 1024, "process_swap_count": 0,
                "system_swapout_delta_pages": 0, "process_resource_gate_passed": True,
                "feature_bounds_finite_fallback_verified": True,
                "input_pixels_independently_reconciled": True,
                "crop_lag_hash_matches_support": True,
                "direct_max_abs_diff": 1e-8, "sparse_dense_max_abs_diff": 0.0,
                "pilot_sha256": trainer.file_sha256(pilot_path),
                "log_sha256": trainer.file_sha256(log_path),
            }
            proof_path.write_text(json.dumps(proof), encoding="utf-8")
            verified = trainer.verify_resource_proof(proof_path, "fit", "support")
            self.assertEqual(verified["status"], "passed")
            proof["input_pixels_independently_reconciled"] = False
            proof_path.write_text(json.dumps(proof), encoding="utf-8")
            with self.assertRaises(ValueError):
                trainer.verify_resource_proof(proof_path, "fit", "support")

    @staticmethod
    def _row(ident, image):
        return {"id": ident, "document_id": f"doc-{image}", "image": image,
                "height": 24, "width": 28, "query_box": [4, 4, 20, 20]}


if __name__ == "__main__":
    unittest.main()
