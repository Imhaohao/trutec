"""Synthetic native-resolution checks for texture/CNN fusion."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hatch_texture
import inference_fusion
from evaluate import EvaluationError
from evaluate_texture_sweep import score_known_probabilities
from inference import read_drawing, read_requests
from inference_texture import (
    interpolate_and_threshold_mask,
    predict_probability_grid,
    query_self_quantiles,
    threshold_from_query_self,
)
from texture_learning import pair_features


class _TextureModel:
    def predict_proba(self, pairs: np.ndarray) -> np.ndarray:
        values = np.tanh(pairs[:, 0] * 2.0 - pairs[:, 1])
        positive = np.asarray(0.5 * (values + 1.0), dtype=np.float32)
        return np.column_stack((1.0 - positive, positive))


def _request() -> dict[str, object]:
    return {
        "id": "q1",
        "image": "drawing.png",
        "width": 13,
        "height": 17,
        "query_box": [2, 2, 8, 8],
        "context_boxes": [],
    }


def _write_drawing(root: Path) -> np.ndarray:
    gray = np.arange(17 * 13, dtype=np.uint8).reshape(17, 13)
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    Image.fromarray(rgb).save(root / "drawing.png", format="PNG")
    return gray


def _texture_metadata(feature_set: str | None = None) -> dict[str, object]:
    selected = "base" if feature_set is None else feature_set
    metadata: dict[str, object] = {
        "stride": 4,
        "base_feature_names": hatch_texture.texture_feature_names(selected),
    }
    if feature_set is not None:
        metadata["texture_feature_set"] = feature_set
    return metadata


@pytest.mark.parametrize(
    ("metadata_feature_set", "expected_feature_set", "expected_channels"),
    ((None, "base", 46), ("variance8", "variance8", 47)),
)
def test_fusion_routes_metadata_feature_set_to_features_and_query(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    metadata_feature_set: str | None,
    expected_feature_set: str,
    expected_channels: int,
) -> None:
    _write_drawing(tmp_path)
    row = _request()
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

    monkeypatch.setattr(inference_fusion.hatch_texture, "texture_features", spy_features)
    monkeypatch.setattr(
        inference_fusion.hatch_texture, "descriptor_for_query", spy_descriptor
    )
    monkeypatch.setattr(
        inference_fusion,
        "tiled_probability",
        lambda model, rgb, query_box, tile_size, overlap, device: np.ones(
            rgb.shape[:2], dtype=np.float32
        ),
    )

    records = inference_fusion.run_fusion_inference(
        [row],
        tmp_path,
        tmp_path / "out",
        _TextureModel(),
        _texture_metadata(metadata_feature_set),
        object(),
        "cpu",
    )

    assert len(records) == 1
    assert feature_calls == [(expected_feature_set, expected_channels)]
    assert descriptor_calls == [(expected_feature_set, expected_channels)]


def test_parser_defaults_and_annotation_rejection(tmp_path: Path) -> None:
    args = inference_fusion.parse_args(
        [
            "--inputs", "inputs.json",
            "--data-root", str(tmp_path),
            "--output-dir", str(tmp_path / "out"),
            "--texture-model", "texture.joblib",
            "--cnn-model", "cnn.pt",
        ]
    )
    assert (args.quantile, args.scale, args.offset, args.cnn_threshold) == ("q10", 0.7, -0.1, 0.4)
    assert (args.tile_size, args.overlap) == (896, 128)

    dirty = tmp_path / "dirty.json"
    row = _request()
    row["labels"] = {"positive_boxes": [[1, 1, 2, 2]]}
    dirty.write_text(json.dumps({"schema": "hatch-matching-inputs/v1", "examples": [row]}))
    with pytest.raises(EvaluationError):
        read_requests(dirty)


def test_native_fusion_matches_sparse_texture_scores_on_odd_shape_and_cnn_seam(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gray = _write_drawing(tmp_path)
    row = _request()
    np.testing.assert_array_equal(read_drawing(tmp_path, row), gray)
    output = tmp_path / "out"
    texture_model = _TextureModel()
    metadata = _texture_metadata()
    seen: dict[str, object] = {}

    def fake_tiled_probability(
        model: object,
        rgb: np.ndarray,
        query_box: tuple[int, int, int, int],
        tile_size: int,
        overlap: int,
        device: str,
    ) -> np.ndarray:
        del model, query_box
        seen.update({"tile_size": tile_size, "overlap": overlap, "device": device})
        y = np.arange(rgb.shape[0], dtype=np.float32)[:, None]
        x = np.arange(rgb.shape[1], dtype=np.float32)[None, :]
        seam = np.where(y < 8.0, 0.15, 0.85)
        return np.asarray(np.clip(0.1 + 0.01 * x + seam, 0.0, 1.0), dtype=np.float32)

    monkeypatch.setattr(inference_fusion, "tiled_probability", fake_tiled_probability)
    records = inference_fusion.run_fusion_inference(
        [row],
        tmp_path,
        output,
        texture_model,
        metadata,
        object(),
        "cpu",
        quantile="q10",
        scale=1.0,
        offset=-0.1,
    )

    assert seen == {"tile_size": 896, "overlap": 128, "device": "cpu"}
    assert len(records) == 1
    assert not (output / "q1.partial.png").exists()
    with Image.open(output / "q1.png") as image:
        assert image.size == (13, 17)
        assert image.mode == "L"
        native_mask = np.asarray(image) != 0
        assert set(np.unique(np.asarray(image))) <= {0, 255}

    features = hatch_texture.texture_features(gray, stride=4)
    query_box: tuple[int, int, int, int] = (2, 2, 8, 8)
    query_desc = hatch_texture.descriptor_for_query(gray, features, query_box, 4, "drawing")
    query_self = query_self_quantiles(
        texture_model, features, query_desc, query_box, gray.shape, 4
    )
    texture_threshold = threshold_from_query_self(query_self, "q10", 1.0, -0.1)
    texture_grid = predict_probability_grid(texture_model, features, query_desc)
    texture_mask = interpolate_and_threshold_mask(
        texture_grid, gray.shape[0], gray.shape[1], 4, texture_threshold
    )
    cnn_map = fake_tiled_probability(object(), np.zeros((17, 13, 3), np.uint8), query_box, 896, 128, "cpu")
    cnn_mask = cnn_map >= 0.4
    expected_native = texture_mask & cnn_mask
    known = np.zeros(gray.shape, dtype=bool)
    known[1:16, 1:12] = True
    assert set(np.unique(texture_mask[known])) == {False, True}
    assert set(np.unique(cnn_mask[known])) == {False, True}
    assert set(np.unique(expected_native[known])) == {False, True}
    assert np.any(cnn_mask[known] & ~texture_mask[known])
    np.testing.assert_array_equal(native_mask, expected_native)

    sparse_texture = score_known_probabilities(
        texture_model, features, query_desc, known, gray.shape[0], gray.shape[1], 4
    )
    known_y, known_x = np.nonzero(known)
    np.testing.assert_array_equal(
        native_mask[known], (sparse_texture >= texture_threshold) & (cnn_map[known_y, known_x] >= 0.4)
    )


def test_cli_rejects_nonfinite_calibration_settings(tmp_path: Path) -> None:
    base = [
        "--inputs", "inputs.json",
        "--data-root", str(tmp_path),
        "--output-dir", str(tmp_path / "out"),
        "--texture-model", "texture.joblib",
        "--cnn-model", "cnn.pt",
    ]
    with pytest.raises(SystemExit):
        inference_fusion.parse_args(base + ["--scale", "nan"])
    with pytest.raises(SystemExit):
        inference_fusion.parse_args(base + ["--cnn-threshold", "2"])


def test_zero_scale_calibration_matches_a_global_threshold() -> None:
    stats = {"q10": 0.12, "median": 0.91, "q25": 0.44, "q75": 0.98}
    assert threshold_from_query_self(stats, "q10", 0.0, 0.7) == pytest.approx(0.7)
    assert threshold_from_query_self(stats, "median", 0.0, 0.7) == pytest.approx(0.7)

