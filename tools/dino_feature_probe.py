"""Local-only DINOv2 ViT-S/14 feature extraction for research probes."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

PATCH_SIZE = 14
FEATURE_CHANNELS = 384
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class FeatureProbeError(ValueError):
    """Raised when input geometry or the model output violates the probe contract."""


def _checked_rgb(rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3 or image.shape[0] == 0 or image.shape[1] == 0:
        raise FeatureProbeError("RGB input must have shape (height, width, 3) with positive sides")
    if image.dtype != np.uint8:
        raise FeatureProbeError("RGB input must use uint8 values in [0, 255]")
    return image


def pad_rgb_to_patches(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """White-pad HWC uint8 RGB to patch multiples; return HWC image and patch-valid mask."""
    image = _checked_rgb(rgb)
    height, width = image.shape[:2]
    padded_height = ((height + PATCH_SIZE - 1) // PATCH_SIZE) * PATCH_SIZE
    padded_width = ((width + PATCH_SIZE - 1) // PATCH_SIZE) * PATCH_SIZE
    padded = np.full((padded_height, padded_width, 3), 255, dtype=np.uint8)
    padded[:height, :width] = image
    valid = np.zeros((padded_height // PATCH_SIZE, padded_width // PATCH_SIZE), dtype=bool)
    valid[: (height + PATCH_SIZE - 1) // PATCH_SIZE, : (width + PATCH_SIZE - 1) // PATCH_SIZE] = True
    return padded, valid


def _load_model(source_dir: Path, weights_path: Path, device: str) -> tuple[Any, Any]:
    import torch

    source = source_dir.expanduser().resolve()
    weights = weights_path.expanduser().resolve()
    if not source.is_dir():
        raise FeatureProbeError("DINOv2 source_dir must be an existing local checkout")
    if not weights.is_file():
        raise FeatureProbeError("weights_path must be an existing local weights file")
    model = torch.hub.load(str(source), "dinov2_vits14", source="local", pretrained=False)
    state = torch.load(str(weights), map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return torch, model


class DinoFeatureProbe:
    """Frozen, local DINOv2 ViT-S/14 patch features with native-size padding."""

    def __init__(self, source_dir: Path, weights_path: Path, device: str = "cpu") -> None:
        self.torch, self.model = _load_model(source_dir, weights_path, device)
        self.device = device

    def extract(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        image, valid = pad_rgb_to_patches(rgb)
        normalized = image.astype(np.float32) / 255.0
        normalized = (normalized - IMAGENET_MEAN) / IMAGENET_STD
        tensor = self.torch.from_numpy(normalized.transpose(2, 0, 1).copy()).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            tokens = self.model.get_intermediate_layers(tensor, n=1)[0]
        if tokens.ndim != 3 or tokens.shape[0] != 1:
            raise FeatureProbeError("DINOv2 patch output must have shape (1, patch_count, channels)")
        expected_count = valid.shape[0] * valid.shape[1]
        if tokens.shape[1] != expected_count or tokens.shape[2] != FEATURE_CHANNELS:
            raise FeatureProbeError("DINOv2 output violates the ViT-S/14 geometry or channel contract")
        features = tokens[0].reshape(valid.shape[0], valid.shape[1], FEATURE_CHANNELS)
        array = features.detach().float().cpu().numpy()
        if not np.isfinite(array).all():
            raise FeatureProbeError("DINOv2 patch output contains non-finite values")
        return array, valid

    def query_embedding(self, rgb: np.ndarray) -> np.ndarray:
        """Mean valid-patch embedding baseline for a query crop."""
        features, valid = self.extract(rgb)
        selected = features[valid]
        if selected.size == 0:
            raise FeatureProbeError("query has no valid patch embeddings")
        embedding = selected.mean(axis=0, dtype=np.float32)
        if embedding.shape != (FEATURE_CHANNELS,) or not np.isfinite(embedding).all():
            raise FeatureProbeError("query embedding violates the finite 384-channel contract")
        return embedding
