"""Tests for query-conditioned empirical feature distributions."""
from __future__ import annotations

import argparse
from unittest.mock import patch

import numpy as np
import pytest

import hatch_texture
from inference_texture import predict_probability_grid, query_self_quantiles
from texture_learning import pair_feature_names, pair_features
from train_texture import build_metadata, parse_args


def _log_features(values: np.ndarray) -> np.ndarray:
    return np.where(values >= 0, np.log1p(np.maximum(values, 0)), values).astype(np.float32)


def test_distribution_descriptor_is_ordered_quantiles_of_exact_crop() -> None:
    image = np.full((45, 53), 255, dtype=np.uint8)
    image[7:38, 9:44:5] = 0
    image[7:38, 11:44:5] = 96
    features = hatch_texture.texture_features(image, stride=4)
    crop = image[7:38, 9:44]
    crop_features = hatch_texture.texture_features(crop, stride=4)

    descriptor = hatch_texture.descriptor_for_query(
        image, features, (9, 7, 44, 38), stride=4, source="distribution"
    )

    expected = np.quantile(crop_features.reshape(-1, crop_features.shape[-1]), (0.1, 0.5, 0.9), axis=0)
    assert descriptor.shape == (3, features.shape[-1])
    assert descriptor.dtype == np.float32
    np.testing.assert_array_equal(descriptor, expected.astype(np.float32))
    assert np.all(descriptor[0] <= descriptor[1])
    assert np.all(descriptor[1] <= descriptor[2])
    assert np.any(descriptor[0] < descriptor[2])


def test_distribution_keeps_odd_native_query_dimensions_without_resizing() -> None:
    image = np.arange(19 * 23, dtype=np.uint8).reshape(19, 23)
    features = hatch_texture.texture_features(image, stride=4)
    real = hatch_texture.texture_features
    observed_shapes: list[tuple[int, int]] = []

    def record(gray: np.ndarray, stride: int = 4, feature_set: str = "base") -> np.ndarray:
        observed_shapes.append(gray.shape)
        return real(gray, stride=stride, feature_set=feature_set)

    with patch("hatch_texture.texture_features", side_effect=record):
        result = hatch_texture.descriptor_for_query(
            image, features, (3, 5, 14, 18), stride=4, source="distribution"
        )

    assert observed_shapes == [(13, 11)]
    assert result.shape == (3, features.shape[-1])


def test_distribution_pair_features_include_interval_and_normalized_spread() -> None:
    local = np.array([[0.0, 1.0, 5.0], [1.0, 2.0, 8.0]], dtype=np.float32)
    query = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]], dtype=np.float32)

    result = pair_features(local, query)

    assert result.shape == (2, 8 * local.shape[1])
    transformed_local = _log_features(local)
    transformed_query = np.stack([_log_features(row) for row in query])
    q10, q50, q90 = transformed_query
    np.testing.assert_array_equal(result[:, 12:15], np.broadcast_to(q10, local.shape))
    np.testing.assert_array_equal(result[:, 15:18], np.broadcast_to(q90, local.shape))
    expected_outside = np.maximum(q10 - transformed_local, 0) + np.maximum(transformed_local - q90, 0)
    np.testing.assert_array_equal(result[:, 18:21], expected_outside)
    expected_spread = np.abs(transformed_local - q50) / np.maximum(q90 - q10, 1e-3)
    np.testing.assert_array_equal(result[:, 21:24], expected_spread)
    assert result[0, 18] > 0
    assert result[1, 20] > 0


def test_legacy_pair_features_remain_exact_and_names_stay_4c() -> None:
    vectors = np.array([[0.0, 1.0], [3.0, 5.0]], dtype=np.float32)
    query = np.array([1.0, 2.0], dtype=np.float32)
    local = _log_features(vectors)
    transformed_query = _log_features(query)
    broadcast = np.broadcast_to(transformed_query, local.shape).astype(np.float32)
    difference = np.abs(local - broadcast).astype(np.float32)
    relative = (difference / (local + broadcast + 1e-4)).astype(np.float32)
    expected = np.concatenate((local, broadcast, difference, relative), axis=1).astype(np.float32)

    result = pair_features(vectors, query)

    np.testing.assert_array_equal(result, expected)
    names = pair_feature_names(["a", "b"])
    assert names == ["local_a", "local_b", "query_a", "query_b", "abs_diff_a", "abs_diff_b", "sym_rel_diff_a", "sym_rel_diff_b"]
    assert len(pair_feature_names(["a", "b"], "distribution")) == 16


@pytest.mark.parametrize(
    "query",
    [
        np.ones((2, 3), dtype=np.float32),
        np.array([[2.0, 0.0], [1.0, 1.0], [3.0, 2.0]], dtype=np.float32),
        np.array([[0.0, 0.0], [1.0, np.nan], [2.0, 2.0]], dtype=np.float32),
        np.ones((3, 4), dtype=np.float32),
    ],
)
def test_invalid_query_distribution_is_rejected(query: np.ndarray) -> None:
    with pytest.raises(ValueError):
        pair_features(np.ones((2, 2), dtype=np.float32), query)


def test_training_metadata_and_cli_use_distribution_feature_signature() -> None:
    args = argparse.Namespace(
        query_source="distribution", max_iter=1, seed=0, samples_per_class=1
    )
    base_names = hatch_texture.texture_feature_names()
    metadata = build_metadata(args, "manifest", base_names, [], np.array([0, 1]), 0.1)
    assert metadata["query_descriptor_source"] == "distribution"
    assert metadata["pair_feature_names"] == pair_feature_names(base_names, "distribution")
    with patch(
        "sys.argv",
        ["train_texture.py", "--manifest", "m.json", "--data-root", "data", "--output", "out.joblib", "--query-source", "distribution"],
    ):
        assert parse_args().query_source == "distribution"


def test_existing_probability_helpers_accept_distribution_query() -> None:
    class RecordingModel:
        observed_width: int | None = None

        def predict_proba(self, values: np.ndarray) -> np.ndarray:
            self.observed_width = values.shape[1]
            return np.tile(np.array([[0.4, 0.6]], dtype=np.float32), (len(values), 1))

    model = RecordingModel()
    features = np.ones((4, 5, 2), dtype=np.float32)
    query = np.array([[0.5, 0.5], [1.0, 1.0], [1.5, 1.5]], dtype=np.float32)
    probabilities = predict_probability_grid(model, features, query)
    self_stats = query_self_quantiles(model, features, query, (0, 0, 8, 8), (16, 20), 4)

    assert probabilities.shape == (4, 5)
    np.testing.assert_array_equal(probabilities, np.full((4, 5), 0.6, dtype=np.float32))
    assert model.observed_width == 16
    assert self_stats["median"] == pytest.approx(0.6)
    assert self_stats["count"] == 4
