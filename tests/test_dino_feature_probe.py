import numpy as np
import pytest
from pathlib import Path

from tools.dino_feature_probe import (
    DinoFeatureProbe,
    FeatureProbeError,
    FEATURE_CHANNELS,
    PATCH_SIZE,
    pad_rgb_to_patches,
    run_smoke,
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


def test_smoke_proof_records_the_two_fixed_native_shapes_without_torch(tmp_path: Path, monkeypatch) -> None:
    import tools.dino_feature_probe as module

    class FakeParameter:
        requires_grad = False
        device = "cpu"

        def numel(self) -> int:
            return 7

    class FakeModel:
        training = False

        def parameters(self):
            return [FakeParameter()]

    class FakeTorch:
        __version__ = "test"

    class FakeProbe:
        torch = FakeTorch()
        model = FakeModel()

        def __init__(self, *_args) -> None:
            pass

        def extract(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            _, valid = pad_rgb_to_patches(image)
            return np.zeros((*valid.shape, FEATURE_CHANNELS), dtype=np.float32), valid

    source, weights = tmp_path / "source", tmp_path / "weights.pth"
    source.mkdir()
    weights.write_bytes(b"synthetic test only")
    monkeypatch.setattr(module, "source_git_head", lambda _path: module.PINNED_SOURCE_COMMIT)
    monkeypatch.setattr(module, "source_has_changes", lambda _path: False)
    monkeypatch.setattr(module, "file_sha256", lambda _path: module.PINNED_WEIGHTS_SHA256)
    monkeypatch.setattr(module, "DinoFeatureProbe", FakeProbe)

    proof = run_smoke(source, weights)

    assert [item["input_shape"] for item in proof["images"]] == [[15, 29, 3], [896, 896, 3]]
    assert [item["feature_shape"] for item in proof["images"]] == [[2, 3, 384], [64, 64, 384]]
    assert proof["source_commit"] == module.PINNED_SOURCE_COMMIT
    assert proof["checkpoint_sha256"] == module.PINNED_WEIGHTS_SHA256
    assert proof["eval_mode"] and proof["all_parameters_frozen"]
    assert all(all(image["checks"].values()) for image in proof["images"])
    assert "does not capture all Metal/MPS" in proof["memory_measurement_limitation"]


@pytest.mark.parametrize("failure", ["source_commit", "dirty_source", "weight_hash"])
def test_smoke_rejects_unpinned_inputs_before_model_load(tmp_path: Path, monkeypatch, failure: str) -> None:
    import tools.dino_feature_probe as module

    source, weights = tmp_path / "source", tmp_path / "weights.pth"
    source.mkdir()
    weights.write_bytes(b"fixture")
    monkeypatch.setattr(module, "source_git_head", lambda _path: "wrong" if failure == "source_commit" else module.PINNED_SOURCE_COMMIT)
    monkeypatch.setattr(module, "source_has_changes", lambda _path: failure == "dirty_source")
    monkeypatch.setattr(module, "file_sha256", lambda _path: "wrong" if failure == "weight_hash" else module.PINNED_WEIGHTS_SHA256)
    monkeypatch.setattr(module, "DinoFeatureProbe", lambda *_args: pytest.fail("model loading must not occur"))

    expected_error = {
        "source_commit": "source commit must be",
        "dirty_source": "checkout must be clean",
        "weight_hash": "do not match the pinned checkpoint",
    }[failure]
    with pytest.raises(module.FeatureProbeError, match=expected_error):
        module.run_smoke(source, weights)
