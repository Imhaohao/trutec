"""Single-tile native-scale SegGPT adapter for static and synthetic probes."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np

MODEL_ID = "BAAI/seggpt-vit-large"
MODEL_REVISION = "77b179b5a1b50a48bd04ef3138804cfacc2ced73"
MODEL_SOURCE_COMMIT = "cfec84587da2a2cc4e0b9acebe6fba1666e66477"
WEIGHTS_SHA256 = "809d3b653873a6fb9e9b5addab94ce2be9f7e49e56ac77da26a1ca0fc03f5c8f"
CONFIG_SHA256 = "88cf14d056f3f14d2fa0c13bd19948b8f87e11992b3ca8018f0632fbd93f8827"
PREPROCESSOR_CONFIG_SHA256 = "535519227630f792c3b6cb75c5b09073d367fd7f0964eefc93d66fc2cb99fc5d"
PROCESSOR_SOURCE_URL = "https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/seggpt/image_processing_seggpt.py"
MODEL_SOURCE_URL = "https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/seggpt/modeling_seggpt.py"
OFFICIAL_INFERENCE_URL = "https://github.com/baaivision/Painter/blob/cfec84587da2a2cc4e0b9acebe6fba1666e66477/SegGPT/SegGPT_inference/seggpt_engine.py"
INPUT_SIDE = 448
STACKED_HEIGHT = INPUT_SIDE * 2
PATCH_SIZE = 16
PATCHES_PER_VIEW = (INPUT_SIDE // PATCH_SIZE) ** 2
TOTAL_PATCHES = PATCHES_PER_VIEW * 2
PALETTE_SEPARATION_SQUARED = 3 * 255 * 255
WHITE = 255


class SegGptError(ValueError):
    """Raised when SegGPT inputs, pins, or outputs violate the adapter contract."""


def as_rgb(rgb: np.ndarray) -> np.ndarray:
    """Convert grayscale or RGB uint8 input to a validated HWC RGB array."""
    image = np.asarray(rgb)
    if image.dtype != np.uint8 or image.ndim not in (2, 3):
        raise SegGptError("image must be a uint8 grayscale or RGB array")
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.shape[2] != 3 or min(image.shape[:2]) <= 0:
        raise SegGptError("image must have positive height and width and three RGB channels")
    return np.ascontiguousarray(image)


def prepare_prompt(query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Place the exact query in a white 448 canvas and mark only it as class 1."""
    image = as_rgb(query)
    height, width = image.shape[:2]
    _check_canvas_fit(height, width, "query")
    prompt = np.full((INPUT_SIDE, INPUT_SIDE, 3), WHITE, dtype=np.uint8)
    prompt[:height, :width] = image
    mask = np.zeros((INPUT_SIDE, INPUT_SIDE), dtype=np.uint8)
    mask[:height, :width] = 1
    return prompt, mask


def prepare_target(tile: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """Place an edge tile in a white 448 canvas and return its native valid size."""
    image = as_rgb(tile)
    height, width = image.shape[:2]
    _check_canvas_fit(height, width, "target tile")
    canvas = np.full((INPUT_SIDE, INPUT_SIDE, 3), WHITE, dtype=np.uint8)
    canvas[:height, :width] = image
    return canvas, (height, width)


def palette_margin_score(negative_distances: np.ndarray) -> np.ndarray:
    """Convert two negative squared palette distances to a bounded margin score."""
    values = np.asarray(negative_distances, dtype=np.float32)
    if values.ndim != 3 or values.shape[0] != 2:
        raise SegGptError("palette scores must have shape (2, height, width)")
    if not np.isfinite(values).all():
        raise SegGptError("palette scores must be finite")
    margin = (values[1] - values[0]) / np.float32(PALETTE_SEPARATION_SQUARED)
    return np.clip(np.float32(0.5) * (np.float32(1) + margin), 0, 1).astype(np.float32)


def rgb_mean_score(reconstructed_rgb: np.ndarray) -> np.ndarray:
    """Equivalent continuous score for clipped RGB predictions and black/white palette."""
    rgb = np.asarray(reconstructed_rgb, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or not np.isfinite(rgb).all():
        raise SegGptError("reconstructed mask must be a finite HWC RGB array")
    rgb = np.clip(rgb, 0, 255)
    return np.clip(rgb.mean(axis=2) / np.float32(255), 0, 1).astype(np.float32)


def default_patch_mask_values() -> np.ndarray:
    """Return the default prompt-visible and target-masked patch positions."""
    return np.concatenate((np.zeros(PATCHES_PER_VIEW, dtype=bool), np.ones(PATCHES_PER_VIEW, dtype=bool)))


def select_mps(torch_module: Any) -> str:
    """Require MPS explicitly; this probe never falls back to CPU."""
    ensure_mps_fallback_disabled()
    if not bool(torch_module.backends.mps.is_available()):
        raise SegGptError("MPS is unavailable; SegGPT probe requires MPS")
    return "mps"


def ensure_mps_fallback_disabled() -> None:
    if _mps_fallback_enabled():
        raise SegGptError("PYTORCH_ENABLE_MPS_FALLBACK must not enable CPU fallback")


def load_local_model(checkpoint_dir: Path, torch_module: Any | None = None) -> tuple[Any, Any, dict[str, Any]]:
    """Load pinned Safetensors and processor from an existing local snapshot only."""
    checkpoint = checkpoint_dir.expanduser().resolve()
    weights = checkpoint / "model.safetensors"
    if not checkpoint.is_dir() or not weights.is_file():
        raise SegGptError("checkpoint_dir must contain the pinned model.safetensors file")
    ensure_mps_fallback_disabled()
    config_digest = _verify_file_pin(checkpoint / "config.json", CONFIG_SHA256, "config.json")
    processor_digest = _verify_file_pin(
        checkpoint / "preprocessor_config.json", PREPROCESSOR_CONFIG_SHA256, "preprocessor_config.json"
    )
    digest = sha256_file(weights)
    if digest != WEIGHTS_SHA256:
        raise SegGptError("local SegGPT weights do not match the pinned checkpoint SHA-256")
    torch = torch_module or _torch()
    device = select_mps(torch)
    torch.mps.set_per_process_memory_fraction(0.75)
    try:
        import transformers  # type: ignore[import-not-found]
        from transformers import SegGptForImageSegmentation, SegGptImageProcessor  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SegGptError("Transformers SegGPT and torchvision must be installed in the runtime") from exc
    if transformers.__version__ != "5.14.1":
        raise SegGptError(f"Transformers must be pinned to 5.14.1, got {transformers.__version__}")
    model: Any = SegGptForImageSegmentation.from_pretrained(
        str(checkpoint), local_files_only=True, use_safetensors=True
    )
    processor = SegGptImageProcessor.from_pretrained(str(checkpoint), local_files_only=True)
    model.to(device=device, dtype=torch.float32)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    _validate_model_state(model)
    pins = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_source_commit": MODEL_SOURCE_COMMIT,
        "model_source_url": MODEL_SOURCE_URL,
        "official_inference_url": OFFICIAL_INFERENCE_URL,
        "processor_source_url": PROCESSOR_SOURCE_URL,
        "checkpoint_path": str(weights),
        "checkpoint_sha256": digest,
        "checkpoint_bytes": weights.stat().st_size,
        "config_sha256": config_digest,
        "preprocessor_config_sha256": processor_digest,
        "device": device,
        "torch_version": str(torch.__version__),
        "transformers_version": transformers.__version__,
    }
    return model, processor, pins


def predict_tile(
    model: Any,
    processor: Any,
    target_tile: np.ndarray,
    query_crop: np.ndarray,
    device: str = "mps",
    torch_module: Any | None = None,
) -> np.ndarray:
    """Return palette-margin scores for one native target tile, cropped to valid pixels."""
    torch = torch_module or _torch()
    _require_mps_device(torch, device)
    encoded, mask_positions, valid_size = encode_pair(processor, target_tile, query_crop, device, torch)
    return predict_encoded(model, processor, encoded, mask_positions, valid_size, device, torch)


def encode_pair(
    processor: Any,
    target_tile: np.ndarray,
    query_crop: np.ndarray,
    device: str,
    torch_module: Any | None = None,
) -> tuple[Any, Any, tuple[int, int]]:
    """Create the native-scale prompt, target, and default transformer patch mask."""
    torch = torch_module or _torch()
    _require_mps_device(torch, device)
    target, valid_size = prepare_target(target_tile)
    prompt, prompt_mask = prepare_prompt(query_crop)
    encoded = processor(
        images=target,
        prompt_images=prompt,
        prompt_masks=prompt_mask,
        num_labels=1,
        do_resize=False,
        return_tensors="pt",
    ).to(device)
    mask_positions = torch.as_tensor(default_patch_mask_values(), dtype=torch.bool, device=device).unsqueeze(0)
    return encoded, mask_positions, valid_size


def predict_encoded(
    model: Any,
    processor: Any,
    encoded: Any,
    mask_positions: Any,
    valid_size: tuple[int, int],
    device: str,
    torch_module: Any | None = None,
) -> np.ndarray:
    """Run one batch-one forward and return scores on the target's native valid area."""
    torch = torch_module or _torch()
    _require_mps_device(torch, device)
    with torch.inference_mode():
        outputs = model(**encoded, bool_masked_pos=mask_positions, feature_ensemble=False)
        decoded = processor.post_process_semantic_segmentation(
            outputs,
            target_sizes=[(INPUT_SIDE, INPUT_SIDE)],
            num_labels=1,
            return_segmentation_scores=True,
        )[0]
    scores = _negative_distance_array(decoded.segmentation_scores)
    margin_scores = palette_margin_score(scores)
    height, width = valid_size
    return margin_scores[:height, :width]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_canvas_fit(height: int, width: int, name: str) -> None:
    if height > INPUT_SIDE or width > INPUT_SIDE:
        raise SegGptError(f"{name} must fit within {INPUT_SIDE}x{INPUT_SIDE} without resizing")


def _mps_fallback_enabled() -> bool:
    value = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "").strip().lower()
    return value not in ("", "0", "false", "no", "off")


def _verify_file_pin(path: Path, expected: str, name: str) -> str:
    if not path.is_file():
        raise SegGptError(f"checkpoint snapshot is missing pinned {name}")
    digest = sha256_file(path)
    if digest != expected:
        raise SegGptError(f"local SegGPT {name} does not match the pinned SHA-256")
    return digest


def _negative_distance_array(scores: Any) -> np.ndarray:
    if hasattr(scores, "detach"):
        scores = scores.detach().to("cpu").numpy()
    return np.asarray(scores, dtype=np.float32)


def _require_mps_device(torch_module: Any, device: str) -> None:
    if device != "mps":
        raise SegGptError("device must be mps; CPU fallback is disabled")
    select_mps(torch_module)


def _validate_model_state(model: Any) -> None:
    parameters = list(model.parameters())
    if model.training or any(parameter.requires_grad for parameter in parameters):
        raise SegGptError("SegGPT must be in eval mode with all parameters frozen")
    if any(parameter.dtype != _torch().float32 for parameter in parameters):
        raise SegGptError("SegGPT parameters must remain float32 for this probe")


def _torch() -> Any:
    import torch  # type: ignore[import-not-found]

    return torch
