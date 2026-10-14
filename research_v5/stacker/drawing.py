"""Per-drawing, label-free context volumes.

The query descriptor stays at the texture model's training stride (4). Local
evidence is the same descriptor family sampled on a stride-8 lattice.
"""
from __future__ import annotations

import gc

import bootstrap  # noqa: F401

import numpy as np
from PIL import Image

import hatch_texture
from color.color_features import (
    compute_window_maps,
    extract_ink_and_chroma,
    extract_robust_query_signature,
)
from constants import COLOR_WINDOWS, DATA_ROOT, GRID_STRIDE, MODEL_STRIDE, QUERY_SOURCE, log, require
from features import assemble, fraction_maps, mismatch_grid
from inference import read_drawing
from inference_texture import predict_probability_grid
from texture_learning import grid_centers

Image.MAX_IMAGE_PIXELS = None


def consume_volumes(spec: dict, boxes: list, models: list, consumer) -> None:
    """Build one feature volume per box and hand it to consumer(index, volume)."""
    require(len(boxes) == len(models) and len(boxes) > 0, "need one model per box")
    prepared = _load_and_prepare(spec, boxes, models)
    try:
        _emit(prepared, boxes, consumer)
    finally:
        del prepared
        gc.collect()


def _load_and_prepare(spec: dict, boxes: list, models: list) -> dict:
    gray = read_drawing(DATA_ROOT, spec)
    rgb = _load_rgb(spec["image"])
    features = hatch_texture.texture_features(gray, stride=GRID_STRIDE)
    probabilities = _probabilities(models, gray, features, boxes)
    signatures = [extract_robust_query_signature(rgb, tuple(box)) for box in boxes]
    ink, chroma, hue = extract_ink_and_chroma(rgb)
    height, width = int(gray.shape[0]), int(gray.shape[1])
    del gray, rgb, features
    gc.collect()
    centers_y, centers_x = grid_centers(height, width, GRID_STRIDE)
    ink_maps = fraction_maps(ink, centers_y, centers_x)
    chroma_maps = fraction_maps(chroma, centers_y, centers_x)
    mismatches = _mismatches(spec["image"], signatures, ink, chroma, hue, centers_y, centers_x)
    del ink, chroma, hue, signatures
    gc.collect()
    return {
        "probabilities": probabilities,
        "ink_maps": ink_maps,
        "chroma_maps": chroma_maps,
        "mismatches": mismatches,
        "height": height,
        "width": width,
    }


def _emit(prepared: dict, boxes: list, consumer) -> None:
    for index, box in enumerate(boxes):
        volume = assemble(
            prepared["probabilities"][index],
            tuple(int(value) for value in box),
            prepared["height"],
            prepared["width"],
            prepared["ink_maps"],
            prepared["chroma_maps"],
            prepared["mismatches"][index],
        )
        try:
            consumer(index, volume)
        finally:
            del volume


def _probabilities(models, gray, features, boxes) -> list[np.ndarray]:
    return [_probability(model, gray, features, box) for model, box in zip(models, boxes)]


def _probability(model, gray, features, box) -> np.ndarray:
    # Crop descriptor uses the stride the HGB was trained with. `features` is ignored.
    descriptor = hatch_texture.descriptor_for_query(
        gray, features, tuple(int(value) for value in box), MODEL_STRIDE, QUERY_SOURCE
    )
    return predict_probability_grid(model, features, descriptor)


def _mismatches(image, signatures, ink, chroma, hue, centers_y, centers_x):
    rows = [{} for _ in signatures]
    for window in COLOR_WINDOWS:
        maps = compute_window_maps(ink, chroma, hue, window)
        for index, signature in enumerate(signatures):
            rows[index][window] = mismatch_grid(signature, maps, centers_y, centers_x)
        del maps
        log(f"color window {window} {image}")
    return rows


def _load_rgb(relative: str) -> np.ndarray:
    with Image.open(DATA_ROOT / relative) as image:
        return np.asarray(image.convert("RGB"))
