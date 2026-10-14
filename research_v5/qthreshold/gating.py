"""Build one drawing's gated native score maps and hand each map to a consumer.

The consumer sees the full map. This module does not read labels.
"""
from __future__ import annotations

import gc

import limits  # noqa: F401

import numpy as np
from PIL import Image

from color.color_features import build_image_color_cache
from constants import DATA_ROOT
from drawing import consume_volumes
from experiment import COLOR_GATE, COLOR_STRENGTH, COLOR_WINDOW
from inference_v6 import COLOR_STRENGTH as V6_STRENGTH
from inference_v6 import COLOR_WINDOW as V6_WINDOW
from inference_v6 import gated_native, native_cell_lookup, stage2_grid

if (COLOR_WINDOW, COLOR_STRENGTH, COLOR_GATE) != (V6_WINDOW, V6_STRENGTH, "linear"):
    raise RuntimeError("color gate does not match inference_v6")


def emit_gated(spec: dict, boxes: list, stage1, stage2, consumer) -> None:
    grids = _stage2_grids(spec, boxes, stage1, stage2)
    rgb = _load_rgb(spec["image"])
    color_maps = build_image_color_cache(rgb, window_sizes=(COLOR_WINDOW,))[COLOR_WINDOW]
    try:
        _gate_all(spec, boxes, grids, rgb, color_maps, consumer)
    finally:
        del rgb, color_maps, grids
        gc.collect()


def _stage2_grids(spec: dict, boxes: list, stage1, stage2) -> list:
    captured: list = []

    def take(index: int, volume) -> None:
        del index
        captured.append(stage2_grid(stage2, volume))

    consume_volumes(spec, list(boxes), [stage1] * len(boxes), take)
    if len(captured) != len(boxes):
        raise RuntimeError("stage-2 grid count does not match queries")
    return captured


def _gate_all(spec, boxes, grids, rgb, color_maps, consumer) -> None:
    height, width = int(spec["height"]), int(spec["width"])
    for index, box in enumerate(boxes):
        _gate_one(index, grids[index], rgb, color_maps, box, height, width, consumer)


def _gate_one(index, grid, rgb, color_maps, box, height, width, consumer) -> None:
    native = native_cell_lookup(grid, height, width)
    gated = gated_native(native, rgb, color_maps, box)
    del native
    if gated.shape != (height, width) or not np.isfinite(gated).all():
        raise RuntimeError("gated score map is the wrong shape or non-finite")
    try:
        consumer(index, gated)
    finally:
        del gated


def _load_rgb(relative: str) -> np.ndarray:
    with Image.open(DATA_ROOT / relative) as image:
        return np.asarray(image.convert("RGB"))
