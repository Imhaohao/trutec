"""Candidate v7 maps: cleaned stage-1 descriptor, structure stage 2, color gate.

One drawing at a time. The cleaner argument returns a stage-1 descriptor from
the grayscale drawing and the query box. Identity leaves the crop median
unchanged. A stage-2 model whose width matches the stacker volume is the v6
model and skips structure channels; the hardneg_0.75 model consumes the same
joined channels as research_v5/structure/develop.py.
"""
from __future__ import annotations

import bootstrap  # noqa: F401

import gc
from typing import Callable

import numpy as np

import drawing
import hatch_texture
from bootstrap import pin_paths
from color.color_features import (
    build_image_color_cache,
    extract_ink_and_chroma,
    extract_robust_query_signature,
)
from constants import GRID_STRIDE, MODEL_STRIDE, QUERY_SOURCE
from features import assemble, feature_names, fraction_maps
from inference import read_drawing
from inference_texture import predict_probability_grid
from inference_v6 import COLOR_STRENGTH, COLOR_WINDOW, gated_native, native_cell_lookup
from model import positive_probability
from texture_learning import grid_centers

if (COLOR_WINDOW, COLOR_STRENGTH) != (16, 0.6):
    raise RuntimeError("color gate does not match the frozen v6 gate")

Cleaner = Callable[[np.ndarray, object], np.ndarray]
NAMED_CLEANERS = ("identity", "v6", "rule_d", "g1", "g2", "g3")


class DrawingContext:
    def __init__(self, image, requests, gray, rgb, features, ink_maps, chroma_maps, mismatches, color_maps):
        self.image = image
        self.requests = requests
        self.gray = gray
        self.rgb = rgb
        self.features = features
        self.ink_maps = ink_maps
        self.chroma_maps = chroma_maps
        self.mismatches = mismatches
        self.color_maps = color_maps
        self.height = int(gray.shape[0])
        self.width = int(gray.shape[1])
        self.bank = None
        self.references = None


def identity_descriptor(gray: np.ndarray, box) -> np.ndarray:
    """Uncleaned crop median. This is the v6 stage-1 descriptor."""
    return hatch_texture.descriptor_for_query(gray, gray, _box(box), MODEL_STRIDE, QUERY_SOURCE)


def cleaner_by_name(name: str) -> Cleaner:
    if name in ("identity", "v6"):
        return identity_descriptor
    if name not in ("rule_d", "g1", "g2", "g3"):
        raise KeyError(name)
    return _named_cleaner(name)


def gated_maps(requests: list[dict], stage1, stage2, cleaner: Cleaner, data_root) -> list[np.ndarray]:
    """Gated native-resolution score map for each label-free request on one drawing."""
    maps: list[np.ndarray] = []
    visit_maps(requests, stage1, stage2, cleaner, data_root, lambda _request, gated: maps.append(gated))
    return maps


def visit_maps(requests: list[dict], stage1, stage2, cleaner: Cleaner, data_root, visit) -> None:
    context = prepare_drawing(requests, data_root)
    try:
        for index, request in enumerate(requests):
            visit(request, map_request(context, index, request, stage1, stage2, cleaner))
    finally:
        release_drawing(context)


def prepare_drawing(requests: list[dict], data_root) -> DrawingContext:
    _require_one_drawing(requests)
    drawing.DATA_ROOT = data_root
    spec = {key: requests[0][key] for key in ("id", "image", "width", "height")}
    gray = read_drawing(data_root, spec)
    rgb = drawing._load_rgb(spec["image"])
    features = hatch_texture.texture_features(gray, stride=GRID_STRIDE)
    boxes = [_box(request["query_box"]) for request in requests]
    signatures = [extract_robust_query_signature(rgb, box) for box in boxes]
    ink, chroma, hue = extract_ink_and_chroma(rgb)
    centers_y, centers_x = grid_centers(int(gray.shape[0]), int(gray.shape[1]), GRID_STRIDE)
    ink_maps = fraction_maps(ink, centers_y, centers_x)
    chroma_maps = fraction_maps(chroma, centers_y, centers_x)
    mismatches = drawing._mismatches(spec["image"], signatures, ink, chroma, hue, centers_y, centers_x)
    del ink, chroma, hue, signatures
    color_maps = build_image_color_cache(rgb, window_sizes=(COLOR_WINDOW,))[COLOR_WINDOW]
    return DrawingContext(
        spec["image"], requests, gray, rgb, features, ink_maps, chroma_maps, mismatches, color_maps
    )


def map_request(context: DrawingContext, index: int, request: dict, stage1, stage2, cleaner: Cleaner) -> np.ndarray:
    probability = predict_probability_grid(stage1, context.features, cleaner(context.gray, request["query_box"]))
    volume = _assemble(context, probability, request["query_box"], index)
    del probability
    model_input = _model_input(stage2, volume, context, index)
    del volume
    grid = _probability_grid(stage2, model_input)
    del model_input
    native = native_cell_lookup(grid, int(request["height"]), int(request["width"]))
    del grid
    return gated_native(native, context.rgb, context.color_maps, request["query_box"])


def release_drawing(context: DrawingContext) -> None:
    context.gray = None
    context.rgb = None
    context.features = None
    context.ink_maps = None
    context.chroma_maps = None
    context.mismatches = None
    context.color_maps = None
    context.bank = None
    context.references = None
    gc.collect()


def _named_cleaner(name: str) -> Cleaner:
    pin_paths()
    from guards import descriptor_for

    def clean(gray, box, cleaner_name=name):
        return descriptor_for(cleaner_name, gray, _box(box))

    return clean


def _assemble(context: DrawingContext, probability, box, index: int) -> np.ndarray:
    return assemble(
        probability,
        _box(box),
        context.height,
        context.width,
        context.ink_maps,
        context.chroma_maps,
        context.mismatches[index],
    )


def _model_input(stage2, volume: np.ndarray, context: DrawingContext, index: int) -> np.ndarray:
    if int(stage2.n_features_in_) == volume.shape[-1]:
        return volume
    _ensure_structure(context)
    joined = np.concatenate([volume, _structure_channels(context, index)], axis=-1)
    if joined.shape[-1] != int(stage2.n_features_in_):
        raise RuntimeError(
            f"stage2 width {int(stage2.n_features_in_)} != stacker {volume.shape[-1]} or joined {joined.shape[-1]}"
        )
    return joined


def _ensure_structure(context: DrawingContext) -> None:
    if context.references is not None:
        return
    pin_paths()
    import cv2
    from structure_features import component_maps, query_reference

    gray = cv2.cvtColor(context.rgb, cv2.COLOR_RGB2GRAY)
    area_map, side_map, ink = component_maps(context.rgb)
    context.references = [
        query_reference(gray, _box(request["query_box"]), area_map, side_map, ink) for request in context.requests
    ]
    del gray, area_map, side_map, ink
    context.bank = _load_bank(context.image, context.rgb)
    gc.collect()


def _structure_channels(context: DrawingContext, index: int) -> np.ndarray:
    from structure_features import feature_names as structure_names
    from structure_features import structure_volume

    channels = structure_volume(context.bank, context.references[index])
    expected = len(feature_names()) + len(structure_names())
    if channels.shape[-1] + len(feature_names()) != expected:
        raise RuntimeError("structure width")
    return channels


def _load_bank(image: str, rgb: np.ndarray) -> dict:
    from banks import _load_bank as read_cached
    from banks import bank_path
    from structure_features import bank_from_rgb

    cached = read_cached(bank_path(image), image, rgb.shape)
    if cached is not None:
        return cached
    return bank_from_rgb(rgb)


def _probability_grid(stage2, volume: np.ndarray) -> np.ndarray:
    height, width, channels = volume.shape
    flat = positive_probability(stage2, volume.reshape(-1, channels))
    return flat.reshape(height, width)


def _require_one_drawing(requests: list[dict]) -> None:
    if not requests:
        raise RuntimeError("no requests")
    image = requests[0]["image"]
    if any(request["image"] != image for request in requests):
        raise RuntimeError("requests span more than one drawing")


def _box(box) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = (int(value) for value in box)
    return x0, y0, x1, y1
