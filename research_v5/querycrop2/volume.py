"""One-drawing volumes. Wraps the stacker's drawing helpers and swaps in cleaned descriptors."""
from __future__ import annotations

import gc
from typing import Callable, NamedTuple

import bootstrap  # noqa: F401

import numpy as np

from bootstrap import atomic_savez

import drawing
import hatch_texture
from color.color_features import (
    apply_gate,
    build_image_color_cache,
    compute_sample_mismatch,
    extract_ink_and_chroma,
    extract_robust_query_signature,
)
from constants import DATA_ROOT, GRID_STRIDE
from features import assemble, fraction_maps
from guards import CACHE_VERSION, VARIANTS, descriptor_for
from inference import read_drawing
from inference_texture import predict_probability_grid
from inference_v6 import COLOR_STRENGTH, COLOR_WINDOW
from model import scores_at_coordinates
from texture_learning import grid_centers

if (COLOR_WINDOW, COLOR_STRENGTH) != (16, 0.6):
    raise RuntimeError("color gate does not match the frozen v6 gate")


class Context(NamedTuple):
    features: np.ndarray
    signatures: list
    ink_maps: dict
    chroma_maps: dict
    mismatches: list
    color_maps: object
    height: int
    width: int


def prepare(spec: dict, boxes: list) -> tuple[Context, dict[str, list]]:
    """Load one drawing and build the descriptor-independent half of each volume."""
    gray = read_drawing(DATA_ROOT, spec)
    rgb = drawing._load_rgb(spec["image"])
    features = hatch_texture.texture_features(gray, stride=GRID_STRIDE)
    normalized = [_box(box) for box in boxes]
    descriptors = _descriptors(gray, normalized)
    signatures = [extract_robust_query_signature(rgb, box) for box in normalized]
    ink, chroma, hue = extract_ink_and_chroma(rgb)
    height, width = int(gray.shape[0]), int(gray.shape[1])
    del gray
    centers_y, centers_x = grid_centers(height, width, GRID_STRIDE)
    ink_maps = fraction_maps(ink, centers_y, centers_x)
    chroma_maps = fraction_maps(chroma, centers_y, centers_x)
    mismatches = drawing._mismatches(spec["image"], signatures, ink, chroma, hue, centers_y, centers_x)
    del ink, chroma, hue
    gc.collect()
    color_maps = build_image_color_cache(rgb, window_sizes=(COLOR_WINDOW,))[COLOR_WINDOW]
    del rgb
    gc.collect()
    context = Context(features, signatures, ink_maps, chroma_maps, mismatches, color_maps, height, width)
    return context, descriptors


def score_jobs(context: Context, jobs: list, descriptors: dict, stage1, stage2, coordinates: Callable) -> None:
    probabilities = _probabilities(context, descriptors, stage1)
    try:
        for index, job in enumerate(jobs):
            _save_job(context, job, probabilities, index, stage2, coordinates)
    finally:
        del probabilities
        gc.collect()


def _descriptors(gray: np.ndarray, boxes: list[tuple]) -> dict[str, list]:
    return {name: [descriptor_for(name, gray, box) for box in boxes] for name in VARIANTS}


def _probabilities(context: Context, descriptors: dict, stage1) -> dict[str, list]:
    grids = {}
    for name in VARIANTS:
        grids[name] = [predict_probability_grid(stage1, context.features, desc) for desc in descriptors[name]]
    return grids


def _save_job(context, job, probabilities, index: int, stage2, coordinates: Callable) -> None:
    y, x, positive = coordinates(job)
    payload = {"positive": np.asarray(positive, dtype=np.uint8), "version": np.int32(CACHE_VERSION)}
    for name in VARIANTS:
        payload[name] = _gated(context, probabilities[name][index], job["query_box"], index, y, x, stage2)
    _require_scores(job["id"], payload)
    atomic_savez(job["cache_path"], **payload)


def _gated(context, probability, box, index: int, y, x, stage2) -> np.ndarray:
    volume = assemble(
        probability,
        _box(box),
        context.height,
        context.width,
        context.ink_maps,
        context.chroma_maps,
        context.mismatches[index],
    )
    raw = scores_at_coordinates(stage2, volume, y, x)
    del volume
    mismatch = compute_sample_mismatch(context.signatures[index], context.color_maps, y, x)
    gated = apply_gate(raw, mismatch, gate_type="linear", strength=COLOR_STRENGTH)
    return np.asarray(gated, dtype=np.float32)


def _require_scores(query_id: str, payload: dict) -> None:
    length = len(payload["positive"])
    for name in VARIANTS:
        scores = payload[name]
        if scores.shape != (length,) or not np.isfinite(scores).all():
            raise RuntimeError(f"bad scores for {query_id} {name}")


def _box(box) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = (int(value) for value in box)
    return x0, y0, x1, y1
