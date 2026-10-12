from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hatch_cadence
import hatch_texture
import texture_learning
from evaluate import EvaluationError
from evaluate_dino_matching_sweep import save_known_probabilities
from evaluate_texture_sweep import find_required_grid_indices, remap_known_coordinates
from inference_texture import native_grid_coordinates
from tools import cadence_resource_pilot
from evaluate_cadence_matching_sweep import (
    DEVELOPMENT_MANIFEST_SHA256,
    _load_verified_model,
    _probabilities_for_known,
    _validate_model_metadata,
    _validate_partitions,
)


class _DeterministicClassifier:
    classes_ = np.asarray([0, 1])
    n_features_in_ = 204

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        score = 1 / (1 + np.exp(-np.clip(values[:, 0] - values[:, 46], -20, 20)))
        return np.column_stack((1 - score, score))


def _metadata() -> dict[str, object]:
    names = hatch_texture.texture_feature_names("base") + list(hatch_cadence.FEATURE_NAMES)
    def sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "schema": "cadence-pair-classifier/v1", "stride": 4,
        "query_descriptor_source": "crop", "texture_feature_set": "base",
        "feature_set": "base+cadence", "base_feature_names": names,
        "pair_feature_names": texture_learning.pair_feature_names(names, "crop"),
        "params": {"learning_rate": 0.08, "max_leaf_nodes": 31, "l2_regularization": 1.0,
                   "max_iter": 200, "early_stopping": False, "random_state": 42},
        "cadence_protocol_sha256": sha(cadence_resource_pilot.PROTOCOL),
        "resource_pilot_sha256": sha(
            cadence_resource_pilot.ROOT / "artifacts/research-v4/cadence-resource-pilot-proof.json"),
        "source_sha256": {"cadence_matching.py": sha(cadence_resource_pilot.ROOT / "cadence_matching.py")},
        "source_manifest_sha256": "fit-sha", "fit_manifest_sha256": "fit-sha",
        "training_ids": ["fit-q"], "training_documents": ["fit-doc"],
        "sklearn_version": "synthetic", "numpy_version": "synthetic",
    }


def test_model_persistence_verifies_embedded_sidecar_and_file_digest(tmp_path: Path) -> None:
    metadata = _metadata()
    model = HistGradientBoostingClassifier(**{
        "learning_rate": 0.08, "max_leaf_nodes": 31, "l2_regularization": 1.0,
        "max_iter": 200, "early_stopping": False, "random_state": 42,
    }).fit(
        np.zeros((4, 204), dtype=np.float32), np.asarray([0, 1, 0, 1]))
    path = tmp_path / "cadence.joblib"
    texture_learning.save_texture_model(model, metadata, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(".joblib.sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    loaded, loaded_metadata, loaded_digest = _load_verified_model(path)
    assert loaded.n_features_in_ == 204
    assert loaded_metadata == metadata
    assert loaded_digest == digest

    path.with_suffix(".joblib.sha256").write_text(f"{'0' * 64}  {path.name}\n", encoding="utf-8")
    with pytest.raises(EvaluationError, match="SHA sidecar"):
        _load_verified_model(path)


def test_model_metadata_rejects_wrong_cadence_feature_order() -> None:
    metadata = _metadata()
    metadata["base_feature_names"] = list(metadata["base_feature_names"][:-1])
    with pytest.raises(EvaluationError, match="base feature signature"):
        _validate_model_metadata(metadata)  # type: ignore[arg-type]


def _partition_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    fit = [{"id": f"f{i}", "document_id": f"fit-doc-{i % 105}"} for i in range(357)]
    dev = [{"id": f"d{i}", "document_id": f"dev-doc-{i % 12}", "kind": "real"} for i in range(55)]
    return fit, dev


def test_partition_validation_requires_full_disjoint_pinned_splits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit, dev = _partition_rows()
    fit_path, dev_path = tmp_path / "fit.json", tmp_path / "development.json"
    hashes = {fit_path: cadence_resource_pilot.FIT_MANIFEST_SHA256,
              dev_path: DEVELOPMENT_MANIFEST_SHA256}
    monkeypatch.setattr("evaluate_cadence_matching_sweep._sha256", lambda path: hashes[Path(path)])
    monkeypatch.setattr("evaluate_cadence_matching_sweep.validate_split_safety", lambda *_args: None)
    metadata = {
        "fit_manifest_sha256": hashes[fit_path], "source_manifest_sha256": hashes[fit_path],
        "training_ids": [row["id"] for row in fit],
        "training_documents": sorted({row["document_id"] for row in fit}),
    }
    assert _validate_partitions(dev_path, fit_path, tmp_path, dev, fit, metadata) == (
        hashes[fit_path], hashes[dev_path])
    for index in (0, 105, 210, 315):
        fit[index]["document_id"] = dev[0]["document_id"]
    with pytest.raises(EvaluationError, match="share query or document"):
        _validate_partitions(dev_path, fit_path, tmp_path, dev, fit, metadata)


def test_sparse_native_scoring_matches_dense_remap_at_edges_and_partial_cells(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(12)
    gray = rng.integers(0, 256, size=(9, 10), dtype=np.uint8)
    shared = rng.uniform(0, 1, size=(3, 3, 46)).astype(np.float32)
    query = rng.uniform(0, 1, size=51).astype(np.float32)
    lag = hatch_cadence.LagState(8.5, 0.7, (None, None, None, None))
    known = np.zeros(gray.shape, dtype=bool)
    known[0, 0] = True
    known[1, 4] = True
    known[4, 6] = True
    known[8, 9] = True
    calls: list[np.ndarray] = []
    tiled = hatch_cadence.cadence_features_tiled

    def track(gray_arg: np.ndarray, lag_arg: hatch_cadence.LagState, coords: np.ndarray) -> np.ndarray:
        calls.append(coords.copy())
        return tiled(gray_arg, lag_arg, coords)

    monkeypatch.setattr("evaluate_cadence_matching_sweep.hatch_cadence.cadence_features_tiled", track)
    classifier = _DeterministicClassifier()
    sparse, native_y, native_x = _probabilities_for_known(
        classifier, gray, shared, query, lag, known, batch_size=2)
    assert len(calls) == 1

    grid_y, grid_x = np.indices((3, 3))
    dense_coords = np.column_stack((grid_y.ravel(), grid_x.ravel()))
    dense_cadence = tiled(gray, lag, dense_coords)
    dense_local = np.concatenate((shared.reshape(-1, 46), dense_cadence), axis=1)
    dense_pairs = texture_learning.pair_features(dense_local, query)
    dense_grid = classifier.predict_proba(dense_pairs)[:, 1].reshape(3, 3).astype(np.float32)
    ymap, xmap = native_grid_coordinates(*gray.shape, 4)
    expected = remap_known_coordinates(dense_grid, xmap[native_x], ymap[native_y])
    assert np.allclose(sparse, expected, rtol=0, atol=1e-7)
    assert (native_y.tolist(), native_x.tolist()) == ([0, 1, 4, 8], [0, 4, 6, 9])


def test_saved_sparse_cache_uses_existing_consumer_contract(tmp_path: Path) -> None:
    row = {"id": "q", "document_id": "doc", "kind": "real", "width": 3, "height": 2}
    probabilities = np.asarray([0.2, 0.8], dtype=np.float32)
    path = save_known_probabilities(
        tmp_path, row, probabilities, np.asarray([True, False]), np.asarray([False, True]),
        np.asarray([0, 1], dtype=np.int32), np.asarray([0, 2], dtype=np.int32),
    )
    with np.load(path, allow_pickle=False) as cache:
        assert {"probability", "positive", "blank", "y", "x", "id", "document_id", "kind",
                "width", "height", "official_reviewed_pixels"} <= set(cache.files)
        assert np.array_equal(cache["probability"], probabilities)
        assert cache["official_reviewed_pixels"] == 2
