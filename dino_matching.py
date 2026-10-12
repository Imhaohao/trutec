"""Frozen DINO patch pairing, native-grid scoring, and model persistence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Protocol

import joblib
import numpy as np

from tools.dino_feature_probe import (
    CHECKPOINT_URL, FEATURE_CHANNELS, PINNED_SOURCE_COMMIT, PINNED_WEIGHTS_SHA256,
    DinoFeatureProbe, FeatureProbeError, file_sha256, verify_pinned_assets,
)
from texture_learning import grid_centers

PATCH_SIZE = 14
INPUT_TILE = 896
CORE_SIZE = 672
CONTEXT_MARGIN = 112
MAX_QUERY_SIZE = 896
SAMPLES_PER_CLASS = 32
MAX_TRAINING_SAMPLES = 22848
MODEL_PARAMS = {
    "learning_rate": 0.08,
    "max_leaf_nodes": 31,
    "l2_regularization": 1.0,
    "max_iter": 100,
    "early_stopping": False,
    "random_state": 45,
}
FEATURE_SIGNATURE = ["l2_local_384", "l2_local_minus_l2_query_384", "dot_product_1"]


class FeatureExtractor(Protocol):
    def extract(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...
    def query_embedding(self, rgb: np.ndarray) -> np.ndarray: ...


def verify_pinned_sources(source_dir: Path, weights_path: Path) -> dict[str, Any]:
    """Expose the shared pre-construction pin check used by this experiment."""
    return verify_pinned_assets(source_dir, weights_path)


def load_verified_probe(
    source_dir: Path, weights_path: Path, device: str = "cpu",
    expected_pins: dict[str, Any] | None = None,
    verified_pins: dict[str, Any] | None = None,
) -> tuple[DinoFeatureProbe, dict[str, Any]]:
    pins = verified_pins if verified_pins is not None else verify_pinned_sources(source_dir, weights_path)
    if expected_pins is not None and any(pins.get(key) != expected_pins.get(key) for key in ("source_commit", "checkpoint_sha256")):
        raise FeatureProbeError("runtime DINO source/checkpoint pins differ from the trained model")
    return DinoFeatureProbe(source_dir, weights_path, device), pins


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != FEATURE_CHANNELS:
        raise ValueError("DINO vectors must have shape (N, 384)")
    if not np.isfinite(values).all():
        raise ValueError("DINO vectors must be finite")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, np.float32(1e-12))


def pair_features(local_vectors: np.ndarray, query_vector: np.ndarray) -> np.ndarray:
    local = normalize_vectors(local_vectors)
    query = normalize_vectors(np.asarray(query_vector, dtype=np.float32).reshape(1, -1))[0]
    difference = local - query[None, :]
    similarity = local @ query
    result = np.concatenate((local, difference, similarity[:, None]), axis=1).astype(np.float32)
    if result.shape != (len(local), 2 * FEATURE_CHANNELS + 1) or not np.isfinite(result).all():
        raise ValueError("DINO signed pair features are invalid")
    return result


def tile_core_slices(grid_height: int, grid_width: int) -> list[tuple[slice, slice, slice, slice]]:
    """Return fixed 64-token context windows and their nonoverlapping global cores."""
    if grid_height <= 0 or grid_width <= 0:
        raise ValueError("feature grid dimensions must be positive")
    core_patches = CORE_SIZE // PATCH_SIZE
    margin_patches = CONTEXT_MARGIN // PATCH_SIZE
    tile_patches = INPUT_TILE // PATCH_SIZE
    if core_patches + 2 * margin_patches != tile_patches:
        raise ValueError("DINO tile geometry must equal core plus two context margins")
    result = []
    for core_y0 in range(0, grid_height, core_patches):
        core_y1 = min(core_y0 + core_patches, grid_height)
        for core_x0 in range(0, grid_width, core_patches):
            core_x1 = min(core_x0 + core_patches, grid_width)
            result.append((slice(core_y0, core_y0 + tile_patches), slice(core_x0, core_x0 + tile_patches), slice(core_y0, core_y1), slice(core_x0, core_x1)))
    return result


def extract_tiled_features(rgb: np.ndarray, extractor: FeatureExtractor) -> np.ndarray:
    """Extract one drawing through fixed 896px tiles and globally anchored 672px cores."""
    image = _validate_rgb(rgb)
    height, width = image.shape[:2]
    grid_height, grid_width = (height + 13) // 14, (width + 13) // 14
    halo = CONTEXT_MARGIN // PATCH_SIZE
    core_patches = CORE_SIZE // PATCH_SIZE
    tile_patches = INPUT_TILE // PATCH_SIZE
    padded_grid_h = ((grid_height + core_patches - 1) // core_patches - 1) * core_patches + tile_patches
    padded_grid_w = ((grid_width + core_patches - 1) // core_patches - 1) * core_patches + tile_patches
    padded = np.full((padded_grid_h * PATCH_SIZE, padded_grid_w * PATCH_SIZE, 3), 255, dtype=np.uint8)
    padded[halo * PATCH_SIZE:halo * PATCH_SIZE + height, halo * PATCH_SIZE:halo * PATCH_SIZE + width] = image
    output = np.empty((grid_height, grid_width, FEATURE_CHANNELS), dtype=np.float32)
    covered = np.zeros((grid_height, grid_width), dtype=bool)
    halo = CONTEXT_MARGIN // PATCH_SIZE
    tile_patches = INPUT_TILE // PATCH_SIZE
    for sy, sx, cy, cx in tile_core_slices(grid_height, grid_width):
        source_y0, source_x0 = sy.start * PATCH_SIZE, sx.start * PATCH_SIZE
        tile = padded[source_y0:source_y0 + INPUT_TILE, source_x0:source_x0 + INPUT_TILE]
        tile_features, valid = extractor.extract(tile)
        expected = (tile_patches, tile_patches, FEATURE_CHANNELS)
        if tile.shape[:2] != (INPUT_TILE, INPUT_TILE) or tile_features.shape != expected or valid.shape != expected[:2] or not bool(valid.all()):
            raise FeatureProbeError("DINO tile must produce a valid 64x64x384 feature grid")
        target = (cy, cx)
        local_y = slice(halo, halo + cy.stop - cy.start)
        local_x = slice(halo, halo + cx.stop - cx.start)
        if bool(covered[target].any()):
            raise FeatureProbeError("DINO tile cores overlap")
        output[target] = tile_features[local_y, local_x]
        covered[target] = True
    if not bool(covered.all()) or not bool(np.isfinite(output).all()):
        raise FeatureProbeError("DINO tiling left feature cells uncovered or non-finite")
    return output


def query_embedding(rgb: np.ndarray, extractor: FeatureExtractor) -> np.ndarray:
    image = _validate_rgb(rgb)
    height, width = image.shape[:2]
    if max(height, width) > MAX_QUERY_SIZE:
        raise ValueError(f"query crop exceeds the {MAX_QUERY_SIZE}px DINO query limit")
    embedding = np.asarray(extractor.query_embedding(image), dtype=np.float32)
    if embedding.shape != (FEATURE_CHANNELS,) or not np.isfinite(embedding).all():
        raise FeatureProbeError("query embedding must be finite with width 384")
    return normalize_vectors(embedding.reshape(1, -1))[0]


def sample_pure_patch_pixels(
    positive: np.ndarray, known: np.ndarray, seed: int = 45,
    max_per_class: int = SAMPLES_PER_CLASS,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample reviewed native pixels from distinct stride-14 pure-label tokens."""
    positive = np.asarray(positive, dtype=bool)
    known = np.asarray(known, dtype=bool)
    if positive.ndim != 2 or positive.shape != known.shape or bool((positive & ~known).any()):
        raise ValueError("positive and known masks must be matching 2D domains")
    if max_per_class <= 0:
        raise ValueError("max_per_class must be positive")
    height, width = positive.shape
    grid_h, grid_w = (height + 13) // 14, (width + 13) // 14
    pos_pad = np.zeros((grid_h * 14, grid_w * 14), dtype=bool)
    neg_pad = np.zeros_like(pos_pad)
    pos_pad[:height, :width] = positive
    neg_pad[:height, :width] = known & ~positive
    pos_counts = pos_pad.reshape(grid_h, 14, grid_w, 14).sum(axis=(1, 3))
    neg_counts = neg_pad.reshape(grid_h, 14, grid_w, 14).sum(axis=(1, 3))
    pure_pos = (pos_counts > 0) & (neg_counts == 0)
    pure_neg = (neg_counts > 0) & (pos_counts == 0)
    rng = np.random.default_rng(seed)
    return (_sample_pixels(positive, pure_pos, max_per_class, rng),
            _sample_pixels(known & ~positive, pure_neg, max_per_class, rng))


def _sample_pixels(domain: np.ndarray, pure_cells: np.ndarray, maximum: int, rng: np.random.Generator) -> np.ndarray:
    candidates = np.argwhere(pure_cells)
    if len(candidates) > maximum:
        candidates = candidates[np.sort(rng.choice(len(candidates), size=maximum, replace=False))]
    sampled = []
    for patch_y, patch_x in candidates:
        y0, x0 = patch_y * PATCH_SIZE, patch_x * PATCH_SIZE
        pixel_candidates = np.argwhere(domain[y0:y0 + PATCH_SIZE, x0:x0 + PATCH_SIZE])
        if len(pixel_candidates) == 0:
            raise ValueError("pure patch unexpectedly has no sampled class pixels")
        chosen = pixel_candidates[int(rng.integers(0, len(pixel_candidates)))]
        sampled.append((y0 + int(chosen[0]), x0 + int(chosen[1])))
    return np.asarray(sampled, dtype=np.int32).reshape(-1, 2)



def sample_weights(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim != 1 or not len(labels) or not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must be a nonempty vector of binary values")
    positive, negative = int((labels == 1).sum()), int((labels == 0).sum())
    weights = np.where(labels == 1, 1.0 / positive, 1.0 / negative if negative else 0.0)
    if positive and negative:
        weights *= 0.5
    return weights.astype(np.float32)


def score_grid(model: Any, features: np.ndarray, query: np.ndarray, batch_size: int = 65536) -> np.ndarray:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    height, width, channels = features.shape
    if channels != FEATURE_CHANNELS:
        raise ValueError("DINO feature grid must have 384 channels")
    vectors = features.reshape(-1, channels)
    probabilities = np.empty(len(vectors), dtype=np.float32)
    for start in range(0, len(vectors), batch_size):
        stop = min(start + batch_size, len(vectors))
        raw = np.asarray(model.predict_proba(pair_features(vectors[start:stop], query)))
        if raw.ndim != 2 or raw.shape != (stop - start, 2):
            raise ValueError("classifier must return two-column class probabilities")
        probabilities[start:stop] = raw[:, 1].astype(np.float32)
    if not np.isfinite(probabilities).all() or bool(((probabilities < 0) | (probabilities > 1)).any()):
        raise ValueError("classifier produced invalid probabilities")
    return probabilities.reshape(height, width)



def score_known_sparse(model: Any, features: np.ndarray, query: np.ndarray, y: np.ndarray, x: np.ndarray, height: int, width: int, batch_size: int = 65536) -> np.ndarray:
    """Predict only bilinear neighbor cells needed by reviewed native coordinates."""
    from evaluate_texture_sweep import find_required_grid_indices, remap_known_coordinates

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    centers_y, centers_x = grid_centers(height, width, PATCH_SIZE)
    sample_gy = np.interp(y, centers_y, np.arange(len(centers_y))).astype(np.float32)
    sample_gx = np.interp(x, centers_x, np.arange(len(centers_x))).astype(np.float32)
    grid_y, grid_x = find_required_grid_indices(sample_gy, sample_gx, features.shape[0], features.shape[1])
    cell_probabilities = np.empty(len(grid_y), dtype=np.float32)
    for start in range(0, len(grid_y), batch_size):
        stop = min(start + batch_size, len(grid_y))
        pairs = pair_features(features[grid_y[start:stop], grid_x[start:stop]], query)
        raw = np.asarray(model.predict_proba(pairs))
        if raw.ndim != 2 or raw.shape != (stop - start, 2):
            raise ValueError("classifier must return two-column class probabilities")
        cell_probabilities[start:stop] = raw[:, 1].astype(np.float32)
    if not np.isfinite(cell_probabilities).all() or bool(((cell_probabilities < 0) | (cell_probabilities > 1)).any()):
        raise ValueError("classifier produced invalid sparse probabilities")
    grid = np.zeros(features.shape[:2], dtype=np.float32)
    grid[grid_y, grid_x] = cell_probabilities
    return remap_known_coordinates(grid, sample_gx, sample_gy)

def native_grid_coordinates(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    center_y, center_x = grid_centers(height, width, PATCH_SIZE)
    return (
        np.interp(np.arange(height), center_y, np.arange(len(center_y))).astype(np.float32),
        np.interp(np.arange(width), center_x, np.arange(len(center_x))).astype(np.float32),
    )


def remap_grid(probability: np.ndarray, height: int, width: int) -> np.ndarray:
    import cv2

    if probability.shape != ((height + 13) // 14, (width + 13) // 14):
        raise ValueError("probability grid does not match native image geometry")
    gy, gx = native_grid_coordinates(height, width)
    map_x, map_y = np.meshgrid(gx, gy)
    result = cv2.remap(probability, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return np.asarray(result, dtype=np.float32)


def sparse_native_probabilities(probability_grid: np.ndarray, y: np.ndarray, x: np.ndarray, height: int, width: int) -> np.ndarray:
    import cv2

    if probability_grid.shape != ((height + 13) // 14, (width + 13) // 14):
        raise ValueError("probability grid does not match native image geometry")
    centers_y, centers_x = grid_centers(height, width, PATCH_SIZE)
    map_y = np.interp(y, centers_y, np.arange(len(centers_y))).astype(np.float32)[None, :]
    map_x = np.interp(x, centers_x, np.arange(len(centers_x))).astype(np.float32)[None, :]
    sampled = cv2.remap(probability_grid, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return np.asarray(sampled[0], dtype=np.float32)



def model_signature(metadata: dict[str, Any]) -> str:
    stable = {key: value for key, value in metadata.items() if key not in {"wall_time_seconds", "artifact_sha256"}}
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def save_dino_model(model: Any, metadata: dict[str, Any], output_path: Path) -> None:
    output_path = Path(output_path).resolve()
    metadata = dict(metadata)
    metadata["model_signature"] = model_signature(metadata)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "metadata": metadata}, output_path)
    sidecar = {**metadata, "artifact_sha256": file_sha256(output_path)}
    output_path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")


def load_dino_model(path: Path) -> tuple[Any, dict[str, Any]]:
    artifact = joblib.load(Path(path))
    if not isinstance(artifact, dict) or not isinstance(artifact.get("metadata"), dict) or "model" not in artifact:
        raise ValueError("DINO model artifact must contain model and metadata")
    metadata = artifact["metadata"]
    sidecar = json.loads(Path(path).with_suffix(".json").read_text(encoding="utf-8"))
    if sidecar.get("artifact_sha256") != file_sha256(Path(path)):
        raise ValueError("DINO model artifact SHA-256 mismatch")
    if any(sidecar.get(key) != value for key, value in metadata.items()):
        raise ValueError("DINO model JSON metadata does not match its artifact")
    if metadata.get("feature_signature") != FEATURE_SIGNATURE:
        raise ValueError("DINO model feature signature does not match this runtime")
    if metadata.get("model_signature") != model_signature({k: v for k, v in metadata.items() if k != "model_signature"}):
        raise ValueError("DINO model metadata signature mismatch")
    return artifact["model"], sidecar



def grayscale_to_rgb(gray: np.ndarray) -> np.ndarray:
    image = np.asarray(gray)
    if image.ndim != 2 or image.dtype != np.uint8 or not image.size:
        raise ValueError("DINO drawing must be a nonempty uint8 grayscale image")
    return np.repeat(image[:, :, None], 3, axis=2)


def _validate_rgb(rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3 or not image.size or image.dtype != np.uint8:
        raise ValueError("DINO input must be nonempty uint8 RGB HWC")
    return image
