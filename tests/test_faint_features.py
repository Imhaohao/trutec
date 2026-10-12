"""Focused checks for the weak-ink feature family."""
from __future__ import annotations

import argparse
from unittest.mock import patch

import numpy as np
import pytest

import hatch_texture
from texture_learning import pair_feature_names
from train_texture import build_metadata, parse_args


def _drawing_with_weak_repetition() -> np.ndarray:
    image = np.full((73, 81), 255, dtype=np.uint8)
    for x in range(8, 72, 6):
        for y in range(28, 64):
            image[y, x + (y % 3)] = 218
    image[8:18, 10:70] = 0
    image[10:16, 14:66] = 24
    return image


def test_faint_features_append_exact_legacy_prefix() -> None:
    image = _drawing_with_weak_repetition()

    base = hatch_texture.texture_features(image, stride=4, feature_set="base")
    faint = hatch_texture.texture_features(image, stride=4, feature_set="faint")

    assert faint.shape == (*base.shape[:2], 92)
    np.testing.assert_array_equal(faint[:, :, :46], base)
    weak_only = hatch_texture._remove_strong_ink(image)
    expected_faint = hatch_texture.texture_features(weak_only, stride=4, feature_set="base")
    np.testing.assert_array_equal(faint[:, :, 46:], expected_faint)
    assert np.isfinite(faint).all()


def test_faint_preprocessing_removes_strong_annotation_and_keeps_weak_pattern() -> None:
    image = _drawing_with_weak_repetition()

    weak_only = hatch_texture._remove_strong_ink(image)

    assert np.all(weak_only[8:18, 10:70] == 255)
    assert weak_only[39, 20] == 218
    assert weak_only[7, 12] == 255  # One-pixel dilation around the dark annotation.
    assert np.count_nonzero(image == 218) > 0
    assert np.count_nonzero(weak_only == 218) == np.count_nonzero(image == 218)


def test_faint_feature_grid_keeps_native_odd_dimensions_and_white_fallback() -> None:
    image = np.full((17, 23), 255, dtype=np.uint8)

    features = hatch_texture.texture_features(image, stride=4, feature_set="faint")

    assert features.shape == (5, 6, 92)
    assert features.dtype == np.float32
    np.testing.assert_array_equal(features[:, :, 46:], np.zeros_like(features[:, :, 46:]))


@pytest.mark.parametrize(
    "image",
    [np.zeros((8, 9), dtype=np.float32), np.zeros((8, 9, 3), dtype=np.uint8)],
)
def test_faint_feature_input_validation_matches_legacy(image: np.ndarray) -> None:
    with pytest.raises(ValueError):
        hatch_texture.texture_features(image, feature_set="faint")


def test_faint_names_metadata_cli_and_descriptor_sources() -> None:
    image = _drawing_with_weak_repetition()
    features = hatch_texture.texture_features(image, stride=4, feature_set="faint")
    names = hatch_texture.texture_feature_names("faint")
    base_names = hatch_texture.texture_feature_names("base")
    assert len(names) == 92
    assert names[:46] == base_names
    assert names[46:] == [f"faint_{name}" for name in base_names]

    args = argparse.Namespace(
        query_source="distribution", features="faint", max_iter=1, seed=0,
        samples_per_class=1,
    )
    metadata = build_metadata(args, "manifest", names, [], np.array([0, 1]), 0.1)
    assert metadata["texture_feature_set"] == "faint"
    assert metadata["base_feature_names"] == names
    assert metadata["pair_feature_names"] == pair_feature_names(names, "distribution")
    assert len(metadata["pair_feature_names"]) == 8 * 92
    assert hatch_texture.feature_set_from_metadata(metadata) == "faint"

    for source, shape in (("drawing", (92,)), ("crop", (92,)), ("distribution", (3, 92))):
        descriptor = hatch_texture.descriptor_for_query(
            image, features, (8, 28, 70, 64), 4, source, feature_set="faint"
        )
        assert descriptor.shape == shape

    with patch(
        "sys.argv",
        ["train_texture.py", "--manifest", "m.json", "--data-root", "data", "--output", "out.joblib", "--features", "faint"],
    ):
        assert parse_args().features == "faint"
    with patch(
        "sys.argv",
        ["train_texture.py", "--manifest", "m.json", "--data-root", "data", "--output", "out.joblib", "--features", "unknown"],
    ), pytest.raises(SystemExit):
        parse_args()


def test_faint_feature_set_rejects_unknown_metadata_and_names() -> None:
    with pytest.raises(ValueError):
        hatch_texture.texture_feature_names("unknown")
    with pytest.raises(ValueError):
        hatch_texture.feature_set_from_metadata({"texture_feature_set": "unknown"})
