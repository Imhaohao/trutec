from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier

import hatch_cadence
import hatch_texture
import texture_learning
import train_cadence_matching as trainer
import tools.verify_cadence_fit as verifier
from evaluate import EvaluationError


def _lag_record(gray: np.ndarray, row: dict[str, Any], seed: int,
                positive: np.ndarray, negative: np.ndarray) -> dict[str, object]:
    x0, y0, x1, y1 = row["query_box"]
    crop = gray[y0:y1, x0:x1]
    lag = hatch_cadence.estimate_query_lag(crop)
    labels = [1] * len(positive) + [0] * len(negative)
    positive_weight = (0.5 if len(negative) else 1.0) / len(positive)
    negative_weight = ((0.5 if len(positive) else 1.0) / len(negative)
                       if len(negative) else 0.0)
    weights = ([positive_weight] * len(positive) + [negative_weight] * len(negative))
    return {
        "document_id": row["document_id"], "image": row["image"], "query_seed": seed,
        "query_box": row["query_box"], "drawing_pixel_sha256": hashlib.sha256(gray.tobytes()).hexdigest(),
        "crop_shape": list(crop.shape), "crop_pixel_sha256": hashlib.sha256(crop.tobytes()).hexdigest(),
        "lag_available": lag.available, "lag": verifier._serialized_lag(lag),
        "positive_grid_coordinates": positive.tolist(), "negative_grid_coordinates": negative.tolist(),
        "labels": labels, "sample_weights": weights,
        "pair_features_shape": [len(labels), 204],
    }


def test_query_audit_recomputes_fit_coordinates_labels_and_weights(monkeypatch, tmp_path) -> None:
    gray = np.full((12, 12), 255, dtype=np.uint8)
    row = {"id": "q1", "document_id": "d1", "image": "drawing.png", "height": 12,
           "width": 12, "query_box": [0, 0, 4, 4]}
    positive = np.array([[0, 0]], dtype=np.int32)
    negative = np.array([[1, 1]], dtype=np.int32)
    monkeypatch.setattr(verifier, "reviewed_domains",
                        lambda _row, _root: (np.zeros_like(gray, dtype=bool),
                                             np.ones_like(gray, dtype=bool), None, None))
    monkeypatch.setattr(verifier.texture_learning, "sample_grid_domain_centers",
                        lambda *args, **kwargs: (positive, negative))
    record = _lag_record(gray, row, 42, positive, negative)
    totals = {"positive": 0, "negative": 0, "stable_lag_queries": 0}
    verifier._verify_query_record(row, record, gray, hashlib.sha256(gray.tobytes()).hexdigest(),
                                  tmp_path, 42, totals)
    assert totals == {"positive": 1, "negative": 1, "stable_lag_queries": 0}

    record["sample_weights"] = [0.25, 0.75]
    with pytest.raises(EvaluationError, match="labels or weights"):
        verifier._verify_query_record(row, record, gray, hashlib.sha256(gray.tobytes()).hexdigest(),
                                      tmp_path, 42, totals)


def test_coordinate_audit_accepts_json_empty_class_list() -> None:
    verifier._verify_coordinates({"negative_grid_coordinates": []}, "negative",
                                 np.empty((0, 2), dtype=np.int32), {"id": "positive-only"})


def test_training_producer_audits_reconcile_all_sampled_class_states(monkeypatch, tmp_path) -> None:
    gray = np.full((12, 12), 255, dtype=np.uint8)
    shared = np.zeros((3, 3, 46), dtype=np.float32)
    lag = hatch_cadence.LagState(None, None, (None, None, None, None))
    query = np.full(51, 0.5, dtype=np.float32)
    domains: dict[str, tuple[np.ndarray, np.ndarray, None, None]] = {}
    rows: list[dict[str, object]] = []
    cases = ("dual", "positive-only", "negative-only", "empty")
    for case in cases:
        positive = np.zeros_like(gray, dtype=bool)
        known = np.zeros_like(gray, dtype=bool)
        if case in {"dual", "positive-only"}:
            positive[2, 2] = True
            known[2, 2] = True
        if case in {"dual", "negative-only"}:
            known[2, 6] = True
        if case == "empty":
            known[0, 0] = True
        row = {"id": case, "document_id": "doc", "image": "drawing.png",
               "height": 12, "width": 12, "query_box": [0, 0, 1, 1]}
        rows.append(row)
        domains[case] = (positive, known, None, None)

    monkeypatch.setattr(trainer.cadence_matching, "query_descriptor", lambda *_args: (query, lag))
    monkeypatch.setattr(trainer.cadence_matching, "pair_features_for_cells",
                        lambda _gray, _shared, coords, _query, _lag:
                        np.zeros((len(coords), 204), dtype=np.float32))
    monkeypatch.setattr(verifier, "reviewed_domains", lambda row, _root: domains[row["id"]])

    for index, row in enumerate(rows):
        pair, labels, weights, audit = trainer.extract_query_training_pairs(
            row, gray, shared, tmp_path, 42 + index,
            domain_reader=lambda current, _root: domains[current["id"]],
        )
        audit["drawing_pixel_sha256"] = hashlib.sha256(gray.tobytes()).hexdigest()
        totals = {"positive": 0, "negative": 0, "stable_lag_queries": 0}
        verifier._verify_query_record(row, audit, gray, hashlib.sha256(gray.tobytes()).hexdigest(),
                                      tmp_path, 42 + index, totals)
        assert pair.shape == (len(labels), 204)
        np.testing.assert_array_equal(weights, audit["sample_weights"])
        assert len(labels) == sum(totals[cls] for cls in ("positive", "negative"))
        if row["id"] == "negative-only":
            np.testing.assert_array_equal(weights, np.ones(1, dtype=np.float32))
        if row["id"] == "empty":
            assert labels.size == weights.size == 0


def test_model_verifier_requires_locked_shape_params_and_sidecar(monkeypatch, tmp_path) -> None:
    rng = np.random.default_rng(17)
    x = rng.normal(size=(12, 204)).astype(np.float32)
    y = np.array([0, 1] * 6)
    model = HistGradientBoostingClassifier(**verifier.MODEL_PARAMS).fit(x, y)
    model_path = tmp_path / "cadence.joblib"
    names = hatch_texture.texture_feature_names("base") + list(hatch_cadence.FEATURE_NAMES)
    metadata = {
        "schema": "cadence-pair-classifier/v1", "feature_set": "base+cadence",
        "query_descriptor_source": "crop", "texture_feature_set": "base",
        "base_feature_names": names,
        "pair_feature_names": texture_learning.pair_feature_names(names, "crop"),
        "params": dict(verifier.MODEL_PARAMS),
        "stride": 4, "source_manifest_sha256": "fixture",
        "training_ids": [], "training_documents": [],
        "sklearn_version": "fixture", "numpy_version": "fixture",
        "tuple_serialization_fixture": {"estimates": (None, None)},
    }
    texture_learning.save_texture_model(model, metadata, model_path)
    digest = verifier.sha256(model_path)
    model_path.with_suffix(".joblib.sha256").write_text(f"{digest}  {model_path.name}\n", encoding="utf-8")
    loaded_model, loaded_metadata = verifier._metadata(model_path)
    assert loaded_metadata == metadata
    assert verifier._verify_model(loaded_model, metadata, model_path)["features"] == 204

    metadata_path = model_path.with_suffix(".json")
    metadata_path.write_text('{"tuple_serialization_fixture":{"estimates":[1]}}', encoding="utf-8")
    with pytest.raises(EvaluationError, match="sidecar metadata"):
        verifier._metadata(model_path)

    model_path.with_suffix(".joblib.sha256").write_text(f"{'0' * 64}  {model_path.name}\n", encoding="utf-8")
    monkeypatch.setattr(texture_learning, "load_texture_model",
                        lambda _path: pytest.fail("artifact deserialization ran before SHA validation"))
    with pytest.raises(EvaluationError, match="sidecar"):
        verifier._metadata(model_path)
