from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier

import dino_matching
from tools.dino_classifier_pilot import select_pilot_drawing
from dino_matching import (
    FEATURE_CHANNELS,
    extract_tiled_features,
    load_dino_model,
    load_verified_probe,
    model_signature,
    pair_features, grayscale_to_rgb,
    query_embedding,
    remap_grid,
    sample_pure_patch_pixels,
    sample_weights,
    score_grid,
    score_known_sparse,
    tile_core_slices,
)


class FakeExtractor:
    def __init__(self) -> None:
        self.input_shapes: list[tuple[int, int]] = []
        self.inputs: list[np.ndarray] = []

    def extract(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.input_shapes.append(rgb.shape[:2])
        self.inputs.append(rgb.copy())
        height, width = rgb.shape[:2]
        grid = np.zeros((height // 14, width // 14, FEATURE_CHANNELS), dtype=np.float32)
        for y in range(grid.shape[0]):
            for x in range(grid.shape[1]):
                grid[y, x, :3] = rgb[y * 14, x * 14]
        return grid, np.ones(grid.shape[:2], dtype=bool)

    def query_embedding(self, rgb: np.ndarray) -> np.ndarray:
        self.query = rgb.copy()
        return np.ones(FEATURE_CHANNELS, dtype=np.float32)


class ProbabilityModel:
    def __init__(self) -> None:
        self.batch_lengths: list[int] = []

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        self.batch_lengths.append(len(features))
        probability = np.clip((features[:, 0] + 1.0) / 2.0, 0.0, 1.0)
        return np.column_stack((1.0 - probability, probability))


def test_tile_core_slices_cover_edges_without_gaps_or_overlaps() -> None:
    windows = tile_core_slices(101, 109)
    covered = np.zeros((101, 109), dtype=np.int8)
    for source_y, source_x, core_y, core_x in windows:
        covered[core_y, core_x] += 1
        assert source_y.stop - source_y.start == 64
        assert source_x.stop - source_x.start == 64
    assert covered.min() == covered.max() == 1


def test_tiled_features_use_fixed_white_padded_windows_and_global_core_alignment() -> None:
    rgb = np.zeros((901, 1031, 3), dtype=np.uint8)
    rgb[..., 0] = np.arange(rgb.shape[1], dtype=np.uint16)[None, :] % 256
    rgb[..., 1] = np.arange(rgb.shape[0], dtype=np.uint16)[:, None] % 256
    extractor = FakeExtractor()
    features = extract_tiled_features(rgb, extractor)

    assert features.shape == (65, 74, FEATURE_CHANNELS)
    assert extractor.input_shapes == [(896, 896)] * 4
    assert all(image.shape == (896, 896, 3) for image in extractor.inputs)
    assert np.all(extractor.inputs[0][:112, :] == 255)
    assert np.all(extractor.inputs[0][:, :112] == 255)
    np.testing.assert_array_equal(features[0, 0, :2], [0, 0])
    np.testing.assert_array_equal(features[48, 48, :2], [160, 160])
    np.testing.assert_array_equal(features[64, 73, :2], [254, 128])
    assert np.isfinite(features).all()


def test_tiny_native_image_is_white_padded_to_one_full_model_tile() -> None:
    rgb = np.zeros((15, 29, 3), dtype=np.uint8)
    rgb[..., 0] = np.arange(29, dtype=np.uint8)[None, :]
    rgb[..., 1] = np.arange(15, dtype=np.uint8)[:, None]
    extractor = FakeExtractor()
    features = extract_tiled_features(rgb, extractor)
    assert features.shape == (2, 3, FEATURE_CHANNELS)
    assert extractor.input_shapes == [(896, 896)]
    np.testing.assert_array_equal(features[1, 2, :2], [28, 14])


def test_query_embedding_uses_exact_small_crop_and_rejects_large_crop() -> None:
    extractor = FakeExtractor()
    crop = np.full((15, 29, 3), 37, dtype=np.uint8)
    vector = query_embedding(crop, extractor)
    np.testing.assert_array_equal(extractor.query, crop)
    assert vector.shape == (FEATURE_CHANNELS,)
    assert np.isclose(np.linalg.norm(vector), 1.0)
    with pytest.raises(ValueError, match="query crop exceeds"):
        query_embedding(np.zeros((897, 1, 3), dtype=np.uint8), extractor)


def test_pair_features_keep_signed_difference_and_unit_normalized_inputs() -> None:
    local = np.zeros((2, FEATURE_CHANNELS), dtype=np.float32)
    local[:, 0] = (3, -2)
    query = np.zeros(FEATURE_CHANNELS, dtype=np.float32)
    query[0] = 4
    result = pair_features(local, query)
    assert result.shape == (2, 769)
    np.testing.assert_allclose(np.linalg.norm(result[:, :384], axis=1), 1.0)
    np.testing.assert_allclose(result[:, 384:768], np.array([[0.0] * 384, [-2.0] + [0.0] * 383], dtype=np.float32))
    assert np.isfinite(result).all()


def test_pure_patch_sampling_returns_known_pixels_from_distinct_pure_tokens() -> None:
    positive = np.zeros((14 * 40, 14 * 4), dtype=bool)
    known = np.zeros_like(positive)
    positive[:, :14 * 2] = True
    known[:, :14 * 2] = True
    known[:, 14 * 2:] = True
    pos_pixels, neg_pixels = sample_pure_patch_pixels(positive, known, seed=45, max_per_class=32)
    assert len(pos_pixels) == len(neg_pixels) == 32
    assert positive[tuple(pos_pixels.T)].all()
    assert (known & ~positive)[tuple(neg_pixels.T)].all()
    assert len({tuple(point // 14) for point in pos_pixels}) == 32
    assert len({tuple(point // 14) for point in neg_pixels}) == 32
    pos_only, no_negative = sample_pure_patch_pixels(positive, positive, seed=45)
    assert len(pos_only) == 32
    assert len(no_negative) == 0


def test_mixed_label_tokens_are_excluded_but_unknown_pixels_do_not_make_them_mixed() -> None:
    positive = np.zeros((28, 28), dtype=bool)
    known = np.zeros_like(positive)
    positive[0:7, 0:14] = True
    known[0:14, 0:14] = True
    positive[0:14, 14:28] = True
    known[0:14, 14:28] = True
    known[14:28, 0:14] = True
    pos, neg = sample_pure_patch_pixels(positive, known, seed=45, max_per_class=32)
    assert (pos[:, 1] >= 14).all()
    assert (neg[:, 0] >= 14).all()


def test_per_query_weights_balance_dual_and_keep_positive_only_weight_one() -> None:
    dual = sample_weights(np.array([1, 1, 0, 0], dtype=np.int8))
    np.testing.assert_allclose(dual, [0.25, 0.25, 0.25, 0.25])
    positive_only = sample_weights(np.array([1, 1], dtype=np.int8))
    np.testing.assert_allclose(positive_only, [0.5, 0.5])


def test_sparse_scoring_matches_full_native_remap_exactly() -> None:
    height, width = 31, 47
    rng = np.random.default_rng(7)
    features = rng.normal(size=((height + 13) // 14, (width + 13) // 14, FEATURE_CHANNELS)).astype(np.float32)
    query = rng.normal(size=FEATURE_CHANNELS).astype(np.float32)
    known = np.zeros((height, width), dtype=bool)
    known[::3, ::5] = True
    y, x = np.nonzero(known)
    model = ProbabilityModel()
    dense = remap_grid(score_grid(model, features, query), height, width)[y, x]
    model.batch_lengths.clear()
    sparse = score_known_sparse(model, features, query, y, x, height, width, batch_size=2)
    np.testing.assert_array_equal(sparse, dense)
    assert max(model.batch_lengths) <= 2
    assert len(model.batch_lengths) > 1


def test_source_pin_mismatch_fails_before_probe_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    constructed = False

    def forbidden_constructor(*args: object, **kwargs: object) -> None:
        nonlocal constructed
        constructed = True

    monkeypatch.setattr(dino_matching, "verify_pinned_assets", lambda *_: {"source_commit": "wrong", "checkpoint_sha256": "wrong"})
    monkeypatch.setattr(dino_matching, "DinoFeatureProbe", forbidden_constructor)
    with pytest.raises(ValueError, match="pins differ"):
        load_verified_probe(Path("source"), Path("weights"), expected_pins={"source_commit": "pinned", "checkpoint_sha256": "pinned"})
    assert not constructed


def test_model_artifact_and_metadata_round_trip(tmp_path) -> None:
    model_path = tmp_path / "candidate.joblib"
    metadata = {"feature_signature": dino_matching.FEATURE_SIGNATURE, "fit_ids": ["q1"]}
    dino_matching.save_dino_model(ProbabilityModel(), metadata, model_path)
    _, loaded = load_dino_model(model_path)
    assert len(loaded["artifact_sha256"]) == 64
    assert loaded["model_signature"] == model_signature({k: v for k, v in loaded.items() if k != "model_signature"})


def test_real_histogram_boosting_artifact_round_trip(tmp_path) -> None:
    rng = np.random.default_rng(13)
    features = rng.normal(size=(24, 769)).astype(np.float32)
    labels = np.array([0, 1] * 12, dtype=np.int8)
    weights = np.full(24, 1 / 24, dtype=np.float32)
    model = HistGradientBoostingClassifier(**dino_matching.MODEL_PARAMS).fit(
        features, labels, sample_weight=weights
    )
    expected = model.predict_proba(features)
    path = tmp_path / "real-hgb.joblib"
    dino_matching.save_dino_model(model, {"feature_signature": dino_matching.FEATURE_SIGNATURE}, path)
    loaded, metadata = load_dino_model(path)
    np.testing.assert_array_equal(loaded.predict_proba(features), expected)
    assert {key: loaded.get_params()[key] for key in dino_matching.MODEL_PARAMS} == dino_matching.MODEL_PARAMS
    assert "classifier_sha256" not in metadata


def test_pilot_selects_largest_dual_supported_fit_drawing_deterministically() -> None:
    rows = [
        {"id": "q-small", "image": "b.png", "height": 280, "width": 280, "document_id": "d2"},
        {"id": "q-big-a", "image": "a.png", "height": 560, "width": 560, "document_id": "d1"},
        {"id": "q-big-b", "image": "a.png", "height": 560, "width": 560, "document_id": "d1"},
        {"id": "q-unsupported", "image": "c.png", "height": 900, "width": 900, "document_id": "d3"},
    ]
    support = {
        "per_query": [
            {"query_id": "q-small", "pure_positive_patch_cells": 2, "pure_negative_patch_cells": 2},
            {"query_id": "q-big-a", "pure_positive_patch_cells": 3, "pure_negative_patch_cells": 1},
            {"query_id": "q-big-b", "pure_positive_patch_cells": 4, "pure_negative_patch_cells": 1},
            {"query_id": "q-unsupported", "pure_positive_patch_cells": 5, "pure_negative_patch_cells": 0},
        ]
    }
    selected = select_pilot_drawing(rows, support)
    assert [row["id"] for row in selected] == ["q-big-a", "q-big-b"]


def test_grayscale_loader_conversion_replicates_native_values() -> None:
    gray = np.array([[0, 17], [128, 255]], dtype=np.uint8)
    rgb = grayscale_to_rgb(gray)
    assert rgb.shape == (2, 2, 3)
    np.testing.assert_array_equal(rgb[:, :, 0], gray)
    np.testing.assert_array_equal(rgb[:, :, 1], gray)
    np.testing.assert_array_equal(rgb[:, :, 2], gray)
