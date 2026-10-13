"""Verify model loading and reproduction of HGB cache probabilities."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research_v5"))

import devharness  # noqa: E402
import hatch_texture  # noqa: E402
from inference_texture import (  # noqa: E402
    native_grid_coordinates,
    predict_probability_grid,
    verify_model_metadata,
)
from texture_learning import load_texture_model  # noqa: E402


def get_model_path() -> Path:
    candidates = [
        ROOT / "models/texture-hgb-crop-v1.joblib",
        ROOT / "models/models/texture-hgb-crop-v1.joblib",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("Could not find texture-hgb-crop-v1.joblib")


def bilinear_sample_grid(
    grid: np.ndarray,
    height: int,
    width: int,
    stride: int,
    y: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    gy_native, gx_native = native_grid_coordinates(height, width, stride)
    gy = gy_native[y]
    gx = gx_native[x]

    gh, gw = grid.shape
    x0 = np.clip(np.floor(gx).astype(np.int32), 0, gw - 1)
    x1 = np.clip(x0 + 1, 0, gw - 1)
    y0 = np.clip(np.floor(gy).astype(np.int32), 0, gh - 1)
    y1 = np.clip(y0 + 1, 0, gh - 1)

    wx = (gx - x0).astype(np.float32)
    wy = (gy - y0).astype(np.float32)

    val00 = grid[y0, x0]
    val01 = grid[y0, x1]
    val10 = grid[y1, x0]
    val11 = grid[y1, x1]

    top = val00 * (1.0 - wx) + val01 * wx
    bottom = val10 * (1.0 - wx) + val11 * wx
    return (top * (1.0 - wy) + bottom * wy).astype(np.float32)


def main() -> None:
    model_path = get_model_path()
    model, metadata = load_texture_model(model_path)
    verify_model_metadata(metadata)
    stride = int(metadata.get("stride", 4))
    source = hatch_texture.descriptor_source_from_metadata(metadata)

    queries = devharness.development_queries()
    hgb_cache = devharness.cached_scores(devharness.HGB_CACHE)

    # Test first 3 queries
    for query in queries[:3]:
        gray = devharness.load_gray(query.image)
        features = hatch_texture.texture_features(gray, stride=stride)
        q_desc = hatch_texture.descriptor_for_query(
            gray, features, query.query_box, stride, source
        )
        prob_grid = predict_probability_grid(model, features, q_desc)
        sampled = bilinear_sample_grid(
            prob_grid, query.height, query.width, stride, query.y, query.x
        )
        cached = hgb_cache[query.id]
        diff = np.abs(sampled - cached)
        max_diff = float(np.max(diff))
        mean_diff = float(np.mean(diff))
        print(f"Query {query.id}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")


if __name__ == "__main__":
    main()
