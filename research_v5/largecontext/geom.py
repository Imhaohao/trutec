"""Image geometry shared by training and label-free inference."""

from __future__ import annotations

import math

import numpy as np

from .constants import NATIVE_PER_CELL, QUERY_SIZE

INK_WEIGHTS = (0.2126, 0.7152, 0.0722)


def as_float_rgb(rgb: np.ndarray) -> np.ndarray:
    array = np.asarray(rgb, dtype=np.float32)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("expected an HxWx3 RGB image")
    peak = float(array.max()) if array.size else 0.0
    if peak > 1.5:
        array = array / 255.0
    return array


def area_half(image: np.ndarray) -> np.ndarray:
    """Mean of each non-overlapping 2x2 block. A leftover odd row or column is dropped."""
    height, width = int(image.shape[0]), int(image.shape[1])
    out_h, out_w = height // 2, width // 2
    if out_h == 0 or out_w == 0:
        raise ValueError("image is too small to halve")
    cropped = image[: out_h * 2, : out_w * 2]
    if cropped.ndim == 2:
        blocks = cropped.reshape(out_h, 2, out_w, 2)
        return blocks.mean(axis=(1, 3))
    blocks = cropped.reshape(out_h, 2, out_w, 2, cropped.shape[2])
    return blocks.mean(axis=(1, 3))


def area_half_rgb(rgb: np.ndarray) -> np.ndarray:
    array = as_float_rgb(rgb)
    if min(array.shape[0], array.shape[1]) < 2:
        return array.mean(axis=(0, 1), keepdims=True)
    return area_half(array)


def pack_hwc(rgb_float: np.ndarray) -> np.ndarray:
    """CHW float32: red, green, blue, and ink (1 minus luminance)."""
    red, green, blue = INK_WEIGHTS
    luma = red * rgb_float[..., 0] + green * rgb_float[..., 1] + blue * rgb_float[..., 2]
    ink = 1.0 - luma
    packed = np.concatenate((rgb_float, ink[..., None]), axis=-1)
    moved = packed.transpose(2, 0, 1)
    return np.ascontiguousarray(moved.astype(np.float32))


def pack_half_rgb(rgb: np.ndarray) -> np.ndarray:
    return pack_hwc(area_half_rgb(rgb))


def tile_repeat(image: np.ndarray, size: int = QUERY_SIZE) -> np.ndarray:
    """Fit an image into a square by center-cropping and repeating, never by stretching."""
    height, width = int(image.shape[0]), int(image.shape[1])
    y0 = max((height - size) // 2, 0)
    x0 = max((width - size) // 2, 0)
    cropped = image[y0 : y0 + min(height, size), x0 : x0 + min(width, size)]
    tile_h, tile_w = int(cropped.shape[0]), int(cropped.shape[1])
    if tile_h == size and tile_w == size:
        return np.ascontiguousarray(cropped)
    reps = _reps(cropped.ndim, size, tile_h, tile_w)
    tiled = np.tile(cropped, reps)
    return np.ascontiguousarray(tiled[:size, :size])


def query_pack(rgb: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = (int(value) for value in box)
    crop = np.asarray(rgb)[y0:y1, x0:x1]
    return pack_hwc(tile_repeat(area_half_rgb(crop), QUERY_SIZE))


def apply_dihedral(arrays: list[np.ndarray], turns: int, flip_h: bool, flip_v: bool) -> list[np.ndarray]:
    """Rotate and flip every array the same way. Turns are quarter-turns counter-clockwise."""
    return [_one_dihedral(array, turns, flip_h, flip_v) for array in arrays]


def cell_fractions(known: np.ndarray, positive: np.ndarray, factor: int = NATIVE_PER_CELL):
    """Area averages of the native masks on the output grid."""
    if known.shape != positive.shape:
        raise ValueError("known and positive masks differ")
    height, width = known.shape
    if height % factor or width % factor:
        raise ValueError("mask sides must divide the output cell")
    cells_y, cells_x = height // factor, width // factor
    known_frac = _block_mean(known, cells_y, cells_x, factor)
    pos_frac = _block_mean(positive, cells_y, cells_x, factor)
    return known_frac, pos_frac


def square_window(rgb, known, positive, cy: int, cx: int, side: int):
    """A side-by-side native window. Sheets smaller than the window sit on white, unknown paper."""
    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    y0 = _clamp_origin(int(cy) - side // 2, height, side)
    x0 = _clamp_origin(int(cx) - side // 2, width, side)
    y1, x1 = min(y0 + side, height), min(x0 + side, width)
    canvas = np.full((side, side, 3), 255, np.uint8)
    known_c = np.zeros((side, side), dtype=bool)
    pos_c = np.zeros((side, side), dtype=bool)
    view = rgb[y0:y1, x0:x1]
    canvas[: view.shape[0], : view.shape[1]] = view
    known_c[: y1 - y0, : x1 - x0] = known[y0:y1, x0:x1]
    pos_c[: y1 - y0, : x1 - x0] = positive[y0:y1, x0:x1]
    return canvas, known_c, pos_c


def _reps(ndim: int, size: int, tile_h: int, tile_w: int):
    reps_y = math.ceil(size / tile_h)
    reps_x = math.ceil(size / tile_w)
    if ndim == 3:
        return reps_y, reps_x, 1
    return reps_y, reps_x


def _one_dihedral(array: np.ndarray, turns: int, flip_h: bool, flip_v: bool) -> np.ndarray:
    spun = np.rot90(array, int(turns))
    if flip_v:
        spun = np.flip(spun, axis=0)
    if flip_h:
        spun = np.flip(spun, axis=1)
    return np.ascontiguousarray(spun)


def _block_mean(mask: np.ndarray, cells_y: int, cells_x: int, factor: int) -> np.ndarray:
    blocks = np.asarray(mask, dtype=np.float32).reshape(cells_y, factor, cells_x, factor)
    return np.ascontiguousarray(blocks.mean(axis=(1, 3)))


def _clamp_origin(origin: int, length: int, side: int) -> int:
    if length <= side:
        return 0
    return int(min(max(origin, 0), length - side))
