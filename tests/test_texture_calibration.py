"""Focused checks for shared query-self calibration helpers."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate import EvaluationError
import evaluate_texture_calibration
import hatch_texture
import inference_texture
from evaluate_texture_calibration import (
    _nearest_query_cell as compatibility_nearest_query_cell,
    _predict_probabilities as compatibility_predict_probabilities,
    query_self_quantiles as compatibility_query_self_quantiles,
)
from inference_texture import (
    _nearest_query_cell,
    _predict_probabilities,
    predict_probability_grid,
    query_self_quantiles,
    run_texture_inference,
    threshold_from_query_self,
)
from texture_learning import grid_centers, pair_features


class _NonconstantModel:
    def predict_proba(self, pairs: np.ndarray) -> np.ndarray:
        scores = 1.0 / (1.0 + np.exp(-pairs[:, 0]))
        return np.column_stack((1.0 - scores, scores))


class _InvalidGridModel:
    def __init__(self, mode: str) -> None:
        self.mode = mode

    def predict_proba(self, pairs: np.ndarray) -> np.ndarray:
        count = len(pairs)
        if self.mode == "short":
            return np.zeros((max(0, count - 1), 2), dtype=np.float32)
        positive = np.full(count, 0.5, dtype=np.float32)
        if self.mode == "nan":
            positive[0] = np.nan
        else:
            positive[0] = 1.1
        return np.column_stack((1.0 - positive, positive))


def test_calibration_module_reexports_shared_helpers() -> None:
    assert compatibility_query_self_quantiles is query_self_quantiles
    assert compatibility_nearest_query_cell is _nearest_query_cell
    assert compatibility_predict_probabilities is _predict_probabilities


def test_query_self_quantiles_preserve_probability_quantiles() -> None:
    features = np.arange(4 * 3 * 2, dtype=np.float32).reshape(4, 3, 2) / 10.0
    query_desc = np.array([0.1, 0.2], dtype=np.float32)
    stats = query_self_quantiles(
        _NonconstantModel(), features, query_desc, (4, 4, 8, 8), (16, 12), 4
    )
    centers_y, centers_x = grid_centers(16, 12, 4)
    y_indices = np.flatnonzero((centers_y >= 4) & (centers_y < 8))
    x_indices = np.flatnonzero((centers_x >= 4) & (centers_x < 8))
    vectors = features[np.ix_(y_indices, x_indices)].reshape(-1, 2)
    probabilities = _NonconstantModel().predict_proba(pair_features(vectors, query_desc))[:, 1]
    assert stats["count"] == float(len(probabilities))
    assert stats["median"] == float(np.quantile(probabilities, 0.5))
    assert stats["q10"] == float(np.quantile(probabilities, 0.1))
    assert stats["q25"] == float(np.quantile(probabilities, 0.25))
    assert stats["q75"] == float(np.quantile(probabilities, 0.75))


def test_threshold_from_query_self_clips_and_rejects_nonfinite_values() -> None:
    stats = {"q10": 0.5, "median": 0.5, "q25": 0.5, "q75": 0.5}
    assert threshold_from_query_self(stats, "q10", 0.7, -0.1) == pytest.approx(0.25)
    assert threshold_from_query_self(stats, "q10", 10.0, 0.0) == pytest.approx(0.99)
    assert threshold_from_query_self(stats, "q10", -10.0, 0.0) == pytest.approx(0.01)

    with pytest.raises(ValueError, match="quantile"):
        threshold_from_query_self(stats, "q99", 0.7, -0.1)
    with pytest.raises(ValueError, match="finite"):
        threshold_from_query_self({**stats, "q10": np.nan}, "q10", 0.7, -0.1)
    with pytest.raises(ValueError, match="finite"):
        threshold_from_query_self(stats, "q10", np.inf, -0.1)
    with pytest.raises(ValueError, match="finite"):
        threshold_from_query_self(stats, "q10", 0.7, np.nan)


def test_verify_model_metadata_routes_feature_set_and_rejects_invalid_signature() -> None:
    from inference_texture import verify_model_metadata

    base_names = hatch_texture.texture_feature_names()
    variance8_names = hatch_texture.texture_feature_names("variance8")
    verify_model_metadata({"base_feature_names": base_names})
    verify_model_metadata(
        {"texture_feature_set": "variance8", "base_feature_names": variance8_names}
    )

    with pytest.raises(EvaluationError, match="feature set"):
        verify_model_metadata(
            {"texture_feature_set": "unknown", "base_feature_names": base_names}
        )
    with pytest.raises(EvaluationError, match="mismatch"):
        verify_model_metadata(
            {"texture_feature_set": "variance8", "base_feature_names": base_names}
        )


def test_calibration_routes_variance8_metadata_to_features_and_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    row = {
        "id": "q1",
        "document_id": "d1",
        "kind": "synthetic",
        "image": "drawing.png",
        "width": 8,
        "height": 8,
        "query_box": [0, 0, 8, 8],
        "context_boxes": [],
    }
    metadata = {
        "texture_feature_set": "variance8",
        "base_feature_names": hatch_texture.texture_feature_names("variance8"),
    }
    image = np.arange(64, dtype=np.uint8).reshape(8, 8)
    known = np.ones((8, 8), dtype=bool)
    positive = np.zeros((8, 8), dtype=bool)
    blank = np.zeros((8, 8), dtype=bool)
    feature_calls: list[tuple[str, int]] = []
    descriptor_calls: list[tuple[str, int]] = []
    real_features = hatch_texture.texture_features
    real_descriptor = hatch_texture.descriptor_for_query

    def spy_features(
        gray: np.ndarray, stride: int = 4, feature_set: str = "base"
    ) -> np.ndarray:
        features = real_features(gray, stride=stride, feature_set=feature_set)
        feature_calls.append((feature_set, int(features.shape[-1])))
        return features

    def spy_descriptor(
        gray: np.ndarray,
        features: np.ndarray,
        query_box: tuple[int, int, int, int],
        stride: int = 4,
        source: str = "drawing",
        feature_set: str = "base",
    ) -> np.ndarray:
        descriptor_calls.append((feature_set, int(features.shape[-1])))
        return real_descriptor(gray, features, query_box, stride, source, feature_set)

    monkeypatch.setattr(evaluate_texture_calibration, "read_drawing", lambda root, row: image)
    monkeypatch.setattr(
        evaluate_texture_calibration,
        "reviewed_domains",
        lambda row, root: (positive, known, blank, {}),
    )
    monkeypatch.setattr(
        evaluate_texture_calibration,
        "_load_saved_probabilities",
        lambda path, positive, known, blank, row: np.full(64, 0.5, dtype=np.float32),
    )
    monkeypatch.setattr(evaluate_texture_calibration.hatch_texture, "texture_features", spy_features)
    monkeypatch.setattr(
        evaluate_texture_calibration.hatch_texture,
        "descriptor_for_query",
        spy_descriptor,
    )

    self_rows, baseline, calibrated = evaluate_texture_calibration.run_calibration(
        [row], tmp_path, _NonconstantModel(), metadata, tmp_path
    )

    assert len(self_rows) == 1
    assert baseline
    assert calibrated
    assert feature_calls == [("variance8", 47)]
    assert descriptor_calls == [("variance8", 47)]


def test_native_texture_inference_routes_variance8_metadata_to_features_and_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    row = {
        "id": "q1",
        "image": "drawing.png",
        "width": 8,
        "height": 8,
        "query_box": [0, 0, 8, 8],
        "context_boxes": [],
    }
    metadata = {
        "texture_feature_set": "variance8",
        "base_feature_names": hatch_texture.texture_feature_names("variance8"),
    }
    image = np.arange(64, dtype=np.uint8).reshape(8, 8)
    feature_calls: list[tuple[str, int]] = []
    descriptor_calls: list[tuple[str, int]] = []
    real_features = hatch_texture.texture_features
    real_descriptor = hatch_texture.descriptor_for_query

    def spy_features(
        gray: np.ndarray, stride: int = 4, feature_set: str = "base"
    ) -> np.ndarray:
        features = real_features(gray, stride=stride, feature_set=feature_set)
        feature_calls.append((feature_set, int(features.shape[-1])))
        return features

    def spy_descriptor(
        gray: np.ndarray,
        features: np.ndarray,
        query_box: tuple[int, int, int, int],
        stride: int = 4,
        source: str = "drawing",
        feature_set: str = "base",
    ) -> np.ndarray:
        descriptor_calls.append((feature_set, int(features.shape[-1])))
        return real_descriptor(gray, features, query_box, stride, source, feature_set)

    monkeypatch.setattr(inference_texture, "read_drawing", lambda root, row: image)
    monkeypatch.setattr(inference_texture.hatch_texture, "texture_features", spy_features)
    monkeypatch.setattr(
        inference_texture.hatch_texture,
        "descriptor_for_query",
        spy_descriptor,
    )
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    records = run_texture_inference(
        [row], tmp_path, output_dir, _NonconstantModel(), metadata, threshold=0.5
    )

    assert len(records) == 1
    assert feature_calls == [("variance8", 47)]
    assert descriptor_calls == [("variance8", 47)]


@pytest.mark.parametrize("mode", ("short", "nan", "out_of_range"))
def test_predict_probability_grid_rejects_invalid_model_probabilities(mode: str) -> None:
    features = np.ones((2, 2, 2), dtype=np.float32)
    query_desc = np.ones(2, dtype=np.float32)
    with pytest.raises(EvaluationError, match="probabilit"):
        predict_probability_grid(_InvalidGridModel(mode), features, query_desc)

