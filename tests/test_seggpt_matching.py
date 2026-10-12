"""Synthetic geometry and RGB-distance score checks for SegGPT."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import seggpt_matching as matching


def test_prompt_keeps_query_pixels_and_pads_with_known_background() -> None:
    query = np.arange(5 * 7, dtype=np.uint8).reshape(5, 7)
    prompt, mask = matching.prepare_prompt(query)

    assert prompt.shape == (448, 448, 3)
    assert mask.shape == (448, 448)
    assert np.array_equal(prompt[:5, :7], np.repeat(query[:, :, None], 3, axis=2))
    assert bool((prompt[5:, :, :] == 255).all())
    assert bool((prompt[:, 7:, :] == 255).all())
    assert bool((mask[:5, :7] == 1).all())
    assert bool((mask[5:, :] == 0).all())
    assert bool((mask[:, 7:] == 0).all())


def test_target_canvas_reports_unpadded_native_rectangle() -> None:
    tile = np.full((13, 19, 3), (12, 67, 123), dtype=np.uint8)
    canvas, valid_size = matching.prepare_target(tile)

    assert canvas.shape == (448, 448, 3)
    assert valid_size == (13, 19)
    assert np.array_equal(canvas[:13, :19], tile)
    assert bool((canvas[13:, :, :] == 255).all())
    assert bool((canvas[:, 19:, :] == 255).all())


def test_query_and_target_larger_than_canvas_are_rejected() -> None:
    too_wide = np.zeros((10, 449), dtype=np.uint8)
    with pytest.raises(matching.SegGptError):
        matching.prepare_prompt(too_wide)
    with pytest.raises(matching.SegGptError):
        matching.prepare_target(too_wide)


def test_default_patch_mask_matches_one_prompt_and_one_target() -> None:
    mask = matching.default_patch_mask_values()

    assert mask.shape == (1568,)
    assert int((~mask[:784]).sum()) == 784
    assert int(mask[:784].sum()) == 0
    assert int(mask[784:].sum()) == 784
    assert int((~mask[784:]).sum()) == 0


def test_rgb_mean_score_has_background_foreground_and_tie_values() -> None:
    rgb = np.array([[[0, 0, 0], [127.5, 127.5, 127.5], [255, 255, 255]]], dtype=np.float32)

    scores = matching.rgb_mean_score(rgb)

    assert np.array_equal(scores, np.array([[0.0, 0.5, 1.0]], dtype=np.float32))
    assert int(np.argmin([np.square(rgb[0, 1]).sum(), np.square(rgb[0, 1] - 255).sum()])) == 0


def test_rgb_mean_score_clips_each_channel_before_averaging() -> None:
    rgb = np.array([[[-30, 255, 255]]], dtype=np.float32)

    scores = matching.rgb_mean_score(rgb)

    assert np.allclose(scores, np.array([[170 / 255]], dtype=np.float32))


def test_palette_distance_margin_matches_rgb_mean_score() -> None:
    rgb = np.array([[[0, 0, 0], [32, 129, 245], [255, 255, 255]]], dtype=np.float32)
    background_distance = np.square(rgb).sum(axis=2)
    foreground_distance = np.square(255 - rgb).sum(axis=2)
    negative_distances = np.stack((-background_distance, -foreground_distance))

    distance_scores = matching.palette_margin_score(negative_distances)
    rgb_scores = matching.rgb_mean_score(rgb)

    assert np.allclose(distance_scores, rgb_scores, atol=1e-7)


def test_palette_scores_reject_bad_shape_and_nonfinite_values() -> None:
    with pytest.raises(matching.SegGptError):
        matching.palette_margin_score(np.zeros((1, 2, 2), dtype=np.float32))
    invalid = np.zeros((2, 2, 2), dtype=np.float32)
    invalid[1, 0, 0] = np.nan
    with pytest.raises(matching.SegGptError):
        matching.palette_margin_score(invalid)


def test_device_requirement_never_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    available = _FakeTorch(mps_available=True)
    unavailable = _FakeTorch(mps_available=False)

    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    assert matching.select_mps(available) == "mps"
    with pytest.raises(matching.SegGptError):
        matching.select_mps(unavailable)
    with pytest.raises(matching.SegGptError):
        matching._require_mps_device(available, "cpu")
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    with pytest.raises(matching.SegGptError):
        matching.select_mps(available)


def test_processor_preserves_prepared_canvas_and_binary_palette() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    transformers = pytest.importorskip("transformers")
    if transformers.__version__ != "5.14.1":
        pytest.skip("processor behavior is pinned to Transformers 5.14.1")
    try:
        from transformers import SegGptImageProcessor
    except ImportError as exc:
        pytest.skip(f"SegGPT processor is unavailable: {exc}")

    query = np.full((5, 7), 200, dtype=np.uint8)
    prompt, mask = matching.prepare_prompt(query)
    target, _ = matching.prepare_target(np.full((448, 448), 80, dtype=np.uint8))
    processor = SegGptImageProcessor()
    encoded = processor(
        images=target,
        prompt_images=prompt,
        prompt_masks=mask,
        num_labels=1,
        do_resize=False,
        return_tensors="pt",
    )

    assert set(encoded) == {"pixel_values", "prompt_pixel_values", "prompt_masks"}
    assert all(tuple(value.shape) == (1, 3, 448, 448) for value in encoded.values())
    foreground = encoded["prompt_masks"][0, :, 0, 0]
    padding = encoded["prompt_masks"][0, :, 10, 10]
    expected_foreground = (np.ones(3) - np.array(processor.image_mean)) / np.array(processor.image_std)
    expected_padding = (np.zeros(3) - np.array(processor.image_mean)) / np.array(processor.image_std)
    assert np.allclose(foreground.cpu().numpy(), expected_foreground, atol=1e-5)
    assert np.allclose(padding.cpu().numpy(), expected_padding, atol=1e-5)

    _assert_synthetic_output_decodes_lower_target_half(processor, torch)


def _assert_synthetic_output_decodes_lower_target_half(processor, torch) -> None:
    from types import SimpleNamespace

    target_rgb = np.full((448, 448, 3), 180, dtype=np.float32)
    target_rgb[0, 0] = 0
    target_rgb[0, 1] = 127.5
    target_rgb[0, 2] = 255
    target_rgb[0, 3] = [-30, 255, 255]
    upper_rgb = np.zeros_like(target_rgb)
    upper_rgb[:, :, 0] = 255
    stacked_rgb = np.concatenate((upper_rgb, target_rgb), axis=0) / np.float32(255)
    mean = np.asarray(processor.image_mean, dtype=np.float32)
    std = np.asarray(processor.image_std, dtype=np.float32)
    normalized = (stacked_rgb - mean) / std
    predictions = torch.from_numpy(normalized.transpose(2, 0, 1).copy()).unsqueeze(0)
    output = SimpleNamespace(pred_masks=predictions)

    decoded = processor.post_process_semantic_segmentation(
        output,
        target_sizes=[(448, 448)],
        num_labels=1,
        return_segmentation_scores=True,
    )[0]
    actual_scores = matching.palette_margin_score(decoded.segmentation_scores.cpu().numpy())
    expected_scores = matching.rgb_mean_score(np.clip(target_rgb, 0, 255))

    assert tuple(decoded.segmentation.shape) == (448, 448)
    assert np.allclose(actual_scores, expected_scores, atol=1e-6)
    assert int(decoded.segmentation[0, 0]) == 0
    assert int(decoded.segmentation[0, 1]) == 0
    assert int(decoded.segmentation[0, 2]) == 1
    assert int(decoded.segmentation[0, 3]) == 1


class _FakeMps:
    def __init__(self, available: bool) -> None:
        self.available = available

    def is_available(self) -> bool:
        return self.available


class _FakeBackends:
    def __init__(self, available: bool) -> None:
        self.mps = _FakeMps(available)


class _FakeTorch:
    def __init__(self, mps_available: bool) -> None:
        self.backends = _FakeBackends(mps_available)
