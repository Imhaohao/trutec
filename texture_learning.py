"""Learned pair representations and model persistence for hatch texture matching."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np

DESCR_SCALE: float = 1.0
EPSILON: float = 1e-4


def pair_feature_names(base_names: list[str]) -> list[str]:
    """Return the ordered list of 4*C pair feature channel names.

    The 4 groups of features are:
    1. local_<name>: transformed local patch descriptor
    2. query_<name>: transformed query patch descriptor (broadcast)
    3. abs_diff_<name>: absolute descriptor difference abs(f - q)
    4. sym_rel_diff_<name>: symmetric relative difference abs(f - q) / (f + q + 1e-4)
    """
    names: list[str] = []
    names.extend(f"local_{b}" for b in base_names)
    names.extend(f"query_{b}" for b in base_names)
    names.extend(f"abs_diff_{b}" for b in base_names)
    names.extend(f"sym_rel_diff_{b}" for b in base_names)
    return names


def _transform_nonnegative(arr: np.ndarray) -> np.ndarray:
    """Apply log1p scaling to nonnegative channels while preserving negative values."""
    scaled = np.where(arr >= 0.0, np.log1p(DESCR_SCALE * np.maximum(0.0, arr)), arr)
    return scaled.astype(np.float32)


def pair_features(vectors: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Build pairwise features from grid descriptor vectors and a query descriptor.

    Order of channels in output:
    1. local descriptors (transformed)
    2. query descriptor broadcast (transformed)
    3. absolute descriptor difference: abs(f - q)
    4. symmetric relative difference: abs(f - q) / (f + q + 1e-4)

    Positional, document, and query identifiers are excluded by design.
    """
    if vectors.ndim != 2:
        raise ValueError(f"vectors must be a 2D array of shape (N, C), got ndim={vectors.ndim}")
    if query.ndim != 1:
        raise ValueError(f"query must be a 1D array of shape (C,), got ndim={query.ndim}")
    if vectors.shape[1] != query.shape[0]:
        raise ValueError(
            f"Feature dimension mismatch: vectors has C={vectors.shape[1]}, "
            f"query has C={query.shape[0]}"
        )
    if not np.isfinite(vectors).all():
        raise ValueError("vectors contains non-finite (NaN or Inf) values")
    if not np.isfinite(query).all():
        raise ValueError("query contains non-finite (NaN or Inf) values")

    f = _transform_nonnegative(vectors)
    q = _transform_nonnegative(query)
    q_broadcast = np.broadcast_to(q[np.newaxis, :], f.shape).astype(np.float32)

    abs_diff = np.abs(f - q_broadcast).astype(np.float32)
    denom = f + q_broadcast + EPSILON
    sym_rel_diff = (abs_diff / denom).astype(np.float32)

    result = np.concatenate([f, q_broadcast, abs_diff, sym_rel_diff], axis=1).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("pair_features result contains non-finite values")
    return result


def grid_centers(height: int, width: int, stride: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Compute the native image coordinate centers (cy, cx) for all grid cells.

    For cell (gy, gx), the native coordinates are:
      cy = min(gy * stride + stride // 2, height - 1)
      cx = min(gx * stride + stride // 2, width - 1)
    """
    if height <= 0 or width <= 0:
        raise ValueError(f"Dimensions must be positive, got height={height}, width={width}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got stride={stride}")

    grid_h = math.ceil(height / stride)
    grid_w = math.ceil(width / stride)
    centers_y = np.minimum(np.arange(grid_h) * stride + stride // 2, height - 1)
    centers_x = np.minimum(np.arange(grid_w) * stride + stride // 2, width - 1)
    return centers_y, centers_x


def _sample_from_indices(
    candidates: np.ndarray, max_samples: int, rng: np.random.Generator
) -> np.ndarray:
    """Deterministically choose up to max_samples rows from candidate grid coordinates."""
    count = len(candidates)
    if count == 0:
        return np.empty((0, 2), dtype=int)
    if count <= max_samples:
        return candidates
    chosen = rng.choice(count, size=max_samples, replace=False)
    chosen.sort()
    return candidates[chosen]


def sample_grid_domain_centers(
    height: int,
    width: int,
    positive_domain: np.ndarray,
    negative_domain: np.ndarray,
    stride: int = 4,
    samples_per_class: int = 512,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministically sample grid centers falling inside reviewed positive and negative domains.

    Returns:
      (positive_coords, negative_coords) where each is an array of shape (K, 2)
      containing [grid_y, grid_x] indices.
    """
    if positive_domain.shape != (height, width):
        raise ValueError(f"positive_domain shape {positive_domain.shape} != {(height, width)}")
    if negative_domain.shape != (height, width):
        raise ValueError(f"negative_domain shape {negative_domain.shape} != {(height, width)}")

    centers_y, centers_x = grid_centers(height, width, stride=stride)
    pos_grid = positive_domain[centers_y[:, np.newaxis], centers_x[np.newaxis, :]]
    neg_grid = negative_domain[centers_y[:, np.newaxis], centers_x[np.newaxis, :]]
    # Ensure disjointness between domains
    neg_grid = neg_grid & ~pos_grid

    pos_candidates = np.argwhere(pos_grid)
    neg_candidates = np.argwhere(neg_grid)

    rng = np.random.default_rng(seed)
    pos_sampled = _sample_from_indices(pos_candidates, samples_per_class, rng)
    neg_sampled = _sample_from_indices(neg_candidates, samples_per_class, rng)
    return pos_sampled, neg_sampled


REQUIRED_METADATA_KEYS = {
    "stride",
    "base_feature_names",
    "pair_feature_names",
    "source_manifest_sha256",
    "training_ids",
    "training_documents",
    "sklearn_version",
    "numpy_version",
}


def save_texture_model(model: Any, metadata: dict[str, Any], output_path: Path) -> None:
    """Persist trained classifier and required metadata as a trusted local joblib artifact."""
    output_path = Path(output_path).resolve()
    missing = REQUIRED_METADATA_KEYS - set(metadata)
    if missing:
        raise ValueError(f"Missing required metadata keys: {sorted(missing)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model, "metadata": metadata}
    joblib.dump(payload, output_path)

    json_path = output_path.with_suffix(".json")
    json_path.write_text(json.dumps(metadata, indent=2))


def load_texture_model(model_path: Path) -> tuple[Any, dict[str, Any]]:
    """Load a trusted local model artifact and verify required metadata."""
    path = Path(model_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    if path.suffix != ".joblib":
        raise ValueError(f"Expected a trusted .joblib artifact, got {path.suffix}")

    data = joblib.load(path)
    if not isinstance(data, dict) or "model" not in data or "metadata" not in data:
        raise ValueError("Invalid artifact format: expected dictionary with 'model' and 'metadata'")

    metadata = data["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError("Model metadata must be a dictionary")

    missing = REQUIRED_METADATA_KEYS - set(metadata)
    if missing:
        raise ValueError(f"Model metadata is missing required keys: {sorted(missing)}")

    return data["model"], metadata
