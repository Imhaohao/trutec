"""Per-drawing feature volume and inference scoring wrapper."""
from __future__ import annotations

import gc
from typing import Callable

import bootstrap  # noqa: F401

import numpy as np

import hatch_texture
from color.color_features import (
    apply_gate,
    build_image_color_cache,
    compute_sample_mismatch,
    compute_window_maps,
    extract_ink_and_chroma,
    extract_robust_query_signature,
)
from constants import COLOR_WINDOWS, DATA_ROOT
from features import assemble, fraction_maps, mismatch_grid
from inference import read_drawing
from inference_texture import predict_probability_grid
from model import scores_at_coordinates
from texture_learning import grid_centers

COLOR_GATE_WINDOW = 16
COLOR_GATE_STRENGTH = 0.6


def score_drawing_queries(
    spec: dict,
    queries: list,
    stage1,
    stage2,
    descriptor_fn: Callable[[object, np.ndarray, np.ndarray], np.ndarray],
    data_root=DATA_ROOT,
) -> dict[str, np.ndarray]:
    """Compute gated stage-2 scores for all queries on a single drawing."""
    gray = read_drawing(data_root, spec)
    rgb = _load_rgb(data_root, spec["image"])
    features = hatch_texture.texture_features(gray, stride=8)

    descriptors = [descriptor_fn(q, rgb, gray) for q in queries]
    probabilities = [predict_probability_grid(stage1, features, d) for d in descriptors]

    signatures = [
        extract_robust_query_signature(rgb, tuple(int(v) for v in q.query_box))
        for q in queries
    ]
    ink, chroma, hue = extract_ink_and_chroma(rgb)
    height, width = int(gray.shape[0]), int(gray.shape[1])
    centers_y, centers_x = grid_centers(height, width, 8)

    ink_maps = fraction_maps(ink, centers_y, centers_x)
    chroma_maps = fraction_maps(chroma, centers_y, centers_x)
    mismatches = _compute_mismatches(signatures, ink, chroma, hue, centers_y, centers_x)
    color_cache = build_image_color_cache(rgb, window_sizes=(COLOR_GATE_WINDOW,))[COLOR_GATE_WINDOW]

    del gray, features, ink, chroma, hue
    gc.collect()

    scores = {}
    for idx, q in enumerate(queries):
        volume = assemble(
            probabilities[idx],
            tuple(int(v) for v in q.query_box),
            height,
            width,
            ink_maps,
            chroma_maps,
            mismatches[idx],
        )
        raw = scores_at_coordinates(stage2, volume, q.y, q.x)
        mismatch = compute_sample_mismatch(signatures[idx], color_cache, q.y, q.x)
        scores[q.id] = apply_gate(raw, mismatch, gate_type="linear", strength=COLOR_GATE_STRENGTH)
        del volume

    del rgb, color_cache, ink_maps, chroma_maps, mismatches, probabilities, signatures
    gc.collect()
    return scores


def _compute_mismatches(signatures, ink, chroma, hue, centers_y, centers_x) -> list[dict]:
    rows = [{} for _ in signatures]
    for window in COLOR_WINDOWS:
        maps = compute_window_maps(ink, chroma, hue, window)
        for idx, sig in enumerate(signatures):
            rows[idx][window] = mismatch_grid(sig, maps, centers_y, centers_x)
        del maps
    return rows


def _load_rgb(data_root, relative: str) -> np.ndarray:
    from PIL import Image

    with Image.open(data_root / relative) as image:
        return np.asarray(image.convert("RGB"))
