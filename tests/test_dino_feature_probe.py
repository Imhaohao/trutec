import numpy as np
import pytest

from tools.dino_feature_probe import (
    DinoFeatureProbe,
    FeatureProbeError,
    FEATURE_CHANNELS,
    PATCH_SIZE,
    pad_rgb_to_patches,
)


def test_odd_sized_rgb_is_white_padded_to_patch_multiple() -> None:
    rgb = np.full((15, 29, 3), 12, dtype=np.uint8)
    padded, valid = pad_rgb_to_patches(rgb)

    assert padded.shape == (28, 42, 3)
    assert np.all(padded[:15, :29] == 12)
    assert np.all(padded[15:, :] == 255)
    assert np.all(padded[:, 29:] == 255)
    assert valid.shape == (2, 3)
    assert valid.all()


def test_patch_valid_mask_excludes_only_wholly_padded_patches() -> None:
    rgb = np.zeros((15, 15, 3), dtype=np.uint8)
    _, valid = pad_rgb_to_patches(rgb)

    assert valid.tolist() == [[True, True], [True, True]]


def test_padding_does_not_resample_native_pixels() -> None:
    rgb = np.arange(9 * 11 * 3, dtype=np.uint8).reshape(9, 11, 3)
    padded, _ = pad_rgb_to_patches(rgb)

    assert padded[:9, :11].shape == rgb.shape
    np.testing.assert_array_equal(padded[:9, :11], rgb)
    assert padded.shape == (PATCH_SIZE, PATCH_SIZE, 3)


def test_extract_enforces_vit_small_patch_contract_and_returns_mask() -> None:
    torch = pytest.importorskip("torch")
    class SyntheticModel:
        def get_intermediate_layers(self, tensor: torch.Tensor, n: int) -> tuple[torch.Tensor]:
            assert n == 1
            count = (tensor.shape[-2] // PATCH_SIZE) * (tensor.shape[-1] // PATCH_SIZE)
            return (torch.ones((1, count, FEATURE_CHANNELS), dtype=torch.float32),)

    probe = DinoFeatureProbe.__new__(DinoFeatureProbe)
    probe.torch = torch
    probe.model = SyntheticModel()
    probe.device = "cpu"

    features, valid = probe.extract(np.zeros((14, 15, 3), dtype=np.uint8))

    assert features.shape == (1, 2, 384)
    assert valid.shape == (1, 2)
    assert np.isfinite(features).all()


def test_extract_rejects_wrong_feature_width() -> None:
    torch = pytest.importorskip("torch")
    class WrongWidthModel:
        def get_intermediate_layers(self, tensor: torch.Tensor, n: int) -> tuple[torch.Tensor]:
            return (torch.ones((1, 1, 128)),)

    probe = DinoFeatureProbe.__new__(DinoFeatureProbe)
    probe.torch = torch
    probe.model = WrongWidthModel()
    probe.device = "cpu"

    with pytest.raises(FeatureProbeError, match="channel contract"):
        probe.extract(np.zeros((PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8))


def test_query_embedding_means_only_valid_patches() -> None:
    torch = pytest.importorskip("torch")
    class SyntheticModel:
        def get_intermediate_layers(self, tensor: torch.Tensor, n: int) -> tuple[torch.Tensor]:
            return (torch.tensor([[[2.0] * FEATURE_CHANNELS, [8.0] * FEATURE_CHANNELS]]),)

    probe = DinoFeatureProbe.__new__(DinoFeatureProbe)
    probe.torch = torch
    probe.model = SyntheticModel()
    probe.device = "cpu"

    embedding = probe.query_embedding(np.zeros((PATCH_SIZE, PATCH_SIZE + 1, 3), dtype=np.uint8))

    assert embedding.shape == (FEATURE_CHANNELS,)
    np.testing.assert_array_equal(embedding, np.full(FEATURE_CHANNELS, 5.0, dtype=np.float32))


def test_invalid_rgb_geometry_is_rejected() -> None:
    with pytest.raises(FeatureProbeError):
        pad_rgb_to_patches(np.zeros((10, 10), dtype=np.uint8))
