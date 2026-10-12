"""Local-only DINOv2 ViT-S/14 feature extraction for research probes."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import resource
import subprocess
import time
from typing import Any

import numpy as np

PATCH_SIZE = 14
FEATURE_CHANNELS = 384
PINNED_SOURCE_COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
PINNED_WEIGHTS_SHA256 = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class FeatureProbeError(ValueError):
    """Raised when input geometry or the model output violates the probe contract."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_git_head(source_dir: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(source_dir), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FeatureProbeError("DINOv2 source_dir must be a readable Git checkout") from exc
    return result.stdout.strip()


def source_has_changes(source_dir: Path) -> bool:
    result = subprocess.run(
        ["git", "-C", str(source_dir), "status", "--porcelain"],
        check=True, capture_output=True, text=True,
    )
    return bool(result.stdout.strip())


def process_peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if platform.system() == "Darwin" else peak * 1024)


def _sync_device(torch: Any, device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


def run_smoke(source_dir: Path, weights_path: Path, device: str = "cpu") -> dict[str, Any]:
    """Run the fixed synthetic geometry probe; never reads dataset files or labels."""
    source = source_dir.expanduser().resolve()
    weights = weights_path.expanduser().resolve()
    source_commit = source_git_head(source)
    if source_commit != PINNED_SOURCE_COMMIT:
        raise FeatureProbeError(f"DINOv2 source commit must be {PINNED_SOURCE_COMMIT}, got {source_commit}")
    if source_has_changes(source):
        raise FeatureProbeError("DINOv2 source checkout must be clean for a reproducible smoke test")
    if not weights.is_file():
        raise FeatureProbeError("weights_path must be an existing local weights file")
    checkpoint_sha256 = file_sha256(weights)
    if checkpoint_sha256 != PINNED_WEIGHTS_SHA256:
        raise FeatureProbeError("DINOv2 weights do not match the pinned checkpoint SHA-256")

    rss = {"before_load": process_peak_rss_bytes()}
    load_started = time.perf_counter()
    probe = DinoFeatureProbe(source, weights, device)
    _sync_device(probe.torch, device)
    load_seconds = time.perf_counter() - load_started
    rss["after_load"] = process_peak_rss_bytes()

    parameters = list(probe.model.parameters())
    eval_mode = not bool(probe.model.training)
    frozen = all(not parameter.requires_grad for parameter in parameters)
    parameter_count = sum(parameter.numel() for parameter in parameters)
    parameter_devices = sorted({str(parameter.device) for parameter in parameters})
    if not eval_mode or not frozen:
        raise FeatureProbeError("Loaded DINOv2 model must be in eval mode with all parameters frozen")

    image_results = []
    for height, width in ((15, 29), (896, 896)):
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        forward_started = time.perf_counter()
        features, valid = probe.extract(rgb)
        _sync_device(probe.torch, device)
        forward_seconds = time.perf_counter() - forward_started
        embedding = features[valid].mean(axis=0, dtype=np.float32)
        expected_shape = (((height + 13) // 14), ((width + 13) // 14), FEATURE_CHANNELS)
        checks = {
            "feature_geometry": features.shape == expected_shape,
            "valid_mask_geometry": valid.shape == expected_shape[:2] and bool(valid.all()),
            "finite_features": bool(np.isfinite(features).all()),
            "finite_embedding": bool(np.isfinite(embedding).all()),
            "embedding_width": embedding.shape == (FEATURE_CHANNELS,),
        }
        if not all(checks.values()):
            raise FeatureProbeError(f"Synthetic {height}x{width} image failed feature checks: {checks}")
        rss[f"after_{height}x{width}"] = process_peak_rss_bytes()
        padded, _ = pad_rgb_to_patches(rgb)
        image_results.append({
            "input_shape": list(rgb.shape), "input_dtype": str(rgb.dtype),
            "padded_shape": list(padded.shape), "feature_shape": list(features.shape),
            "feature_dtype": str(features.dtype), "valid_patch_count": int(valid.sum()),
            "query_embedding_shape": list(embedding.shape), "checks": checks,
            "forward_seconds": forward_seconds,
        })

    return {
        "schema": "dino-feature-smoke-proof/v1",
        "source_path": str(source), "source_commit": source_commit,
        "checkpoint_url": CHECKPOINT_URL, "checkpoint_path": str(weights),
        "checkpoint_sha256": checkpoint_sha256, "checkpoint_bytes": weights.stat().st_size,
        "device": device, "torch_version": str(probe.torch.__version__),
        "parameter_count": parameter_count, "parameter_devices": parameter_devices,
        "eval_mode": eval_mode, "all_parameters_frozen": frozen,
        "load_seconds": load_seconds, "process_peak_rss_bytes": rss,
        "rss_method": "resource.getrusage(RUSAGE_SELF).ru_maxrss cumulative process high-water mark",
        "memory_measurement_limitation": (
            "macOS ru_maxrss does not capture all Metal/MPS unified-memory allocations or system-wide pressure; "
            "the root run should retain /usr/bin/time -l output and inspect peak footprint and swap fields"
        ),
        "images": image_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local, synthetic DINOv2 feature smoke probe")
    parser.add_argument("--source", type=Path, required=True, help="Pinned local DINOv2 source checkout")
    parser.add_argument("--weights", type=Path, required=True, help="Pinned local ViT-S/14 checkpoint")
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--output", type=Path, help="Optional JSON proof path; defaults to stdout")
    args = parser.parse_args()
    try:
        proof = run_smoke(args.source, args.weights, args.device)
    except FeatureProbeError as exc:
        parser.error(str(exc))
    serialized = json.dumps(proof, indent=2) + "\n"
    if args.output:
        args.output.write_text(serialized, encoding="utf-8")
    else:
        print(serialized, end="")


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


if __name__ == "__main__":
    main()
