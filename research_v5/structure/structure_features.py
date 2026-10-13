"""Label-free query-relative structure on the stride-8 lattice.

Windows are squares of side 32, 64 and 128 px centered on each cell.
Orientation uses eight bins of unsigned gradient energy, centered at
0, 22.5, ... degrees. Period is the shortest strong autocorrelation lag
along that bin's direction, estimated with lagged products on a 4x
downsampled image. Mark size is the ink-weighted connected-component
area and the smaller side of its box.
"""
from __future__ import annotations

import limits  # noqa: F401

import cv2
import numpy as np

from color.color_features import extract_ink_and_chroma
from constants import GRID_STRIDE
from features import integral_image
from texture_learning import grid_centers

WINDOWS = (32, 64, 128)
ORIENT_BINS = 8
PERIODS = (8, 12, 16, 20, 24, 32, 40, 48, 64)
DOWNSAMPLE = 4
HARMONIC_FRACTION = 0.9
MIN_PERIODICITY = 0.05
MIN_CROP_OVERLAP = 0.25
FEATURE_VERSION = 1

_WINDOW_FIELDS = (
    *(f"orient_b{bin_id}" for bin_id in range(ORIENT_BINS)),
    "orient_cos",
    "orient_l1",
    "period",
    "period_ratio",
    "period_abs",
    "periodicity",
    "periodicity_abs",
    "mark_area",
    "mark_area_abs",
    "mark_side",
    "mark_side_abs",
)
_QUERY_FIELDS = (
    *(f"query_orient_b{bin_id}" for bin_id in range(ORIENT_BINS)),
    "query_period",
    "query_periodicity",
    "query_mark_area",
    "query_mark_side",
)


class QueryReference:
    def __init__(self, histogram, period, periodicity, mark_area, mark_side):
        self.histogram = np.asarray(histogram, dtype=np.float32)
        self.period = float(period)
        self.periodicity = float(periodicity)
        self.mark_area = float(mark_area)
        self.mark_side = float(mark_side)


def feature_names() -> list[str]:
    names = [f"{field}_w{side}" for side in WINDOWS for field in _WINDOW_FIELDS]
    names.extend(_QUERY_FIELDS)
    if len(names) != len(set(names)):
        raise RuntimeError("duplicate structure feature")
    return names


def bank_from_rgb(rgb: np.ndarray) -> dict:
    energy, period, periodicity, mark_area, mark_side = build_arrays(rgb)
    return {
        "energy": energy,
        "period": period,
        "periodicity": periodicity,
        "mark_area": mark_area,
        "mark_side": mark_side,
    }


def build_arrays(rgb: np.ndarray):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    centers_y, centers_x = grid_centers(int(gray.shape[0]), int(gray.shape[1]), GRID_STRIDE)
    energy = orientation_energy(gray, centers_y, centers_x)
    dominant = np.argmax(energy, axis=-1).astype(np.int32)
    period, periodicity = period_maps(gray, centers_y, centers_x, dominant)
    del gray, dominant
    mark_area, mark_side = mark_maps(rgb, centers_y, centers_x)
    return energy, period, periodicity, mark_area, mark_side


def reference_from_rgb(rgb: np.ndarray, box) -> QueryReference:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    area_map, side_map, ink = component_maps(rgb)
    return query_reference(gray, box, area_map, side_map, ink)


def query_reference(gray, box, area_map, side_map, ink) -> QueryReference:
    histogram = _query_histogram(gray, box)
    dominant = int(np.argmax(histogram))
    period, periodicity = _crop_period(gray, box, dominant)
    mark_area, mark_side = _query_marks(area_map, side_map, ink, box)
    return QueryReference(histogram, period, periodicity, mark_area, mark_side)


def structure_volume(bank: dict, reference: QueryReference) -> np.ndarray:
    hist = _normalize_hist(bank["energy"])
    cosine = _cosine(hist, reference.histogram)
    distance = _l1(hist, reference.histogram)
    ratio = _period_ratio(bank["period"], reference.period)
    channels = _window_channels(hist, cosine, distance, ratio, bank, reference)
    channels.extend(_query_channels(hist.shape[:2], reference))
    volume = np.concatenate(channels, axis=-1)
    if volume.shape[-1] != len(feature_names()):
        raise RuntimeError("structure width")
    np.nan_to_num(volume, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return volume


def orientation_energy(gray, centers_y, centers_x) -> np.ndarray:
    energy, bins = gradient_bins(gray)
    grid = np.zeros((len(centers_y), len(centers_x), len(WINDOWS), ORIENT_BINS), np.float32)
    for bin_id in range(ORIENT_BINS):
        _fill_orientation_bin(grid, energy, bins, bin_id, centers_y, centers_x)
    return grid


def period_maps(gray, centers_y, centers_x, dominant):
    shape = (len(centers_y), len(centers_x), len(WINDOWS))
    period = np.zeros(shape, np.float32)
    strength = np.zeros(shape, np.float32)
    small = _downsample(gray)
    if small is None:
        return period, strength
    _fill_period_maps(small, centers_y, centers_x, dominant, period, strength)
    return period, strength


def mark_maps(rgb, centers_y, centers_x):
    area_map, side_map, ink = component_maps(rgb)
    return _window_mark_means(area_map, side_map, ink, centers_y, centers_x)


def component_maps(rgb):
    ink, _chroma, _hue = extract_ink_and_chroma(rgb)
    binary = np.ascontiguousarray((ink > 0).astype(np.uint8))
    del ink
    _count, components, stats, _centroids = cv2.connectedComponentsWithStats(binary, 8)
    area, side = _component_tables(stats)
    area_map = area[components]
    side_map = side[components]
    del components
    return area_map, side_map, binary


def gradient_bins(gray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(gray, dtype=np.float32)
    gx = cv2.Sobel(values, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(values, cv2.CV_32F, 0, 1, ksize=3)
    energy = gx * gx + gy * gy
    angle = np.mod(np.arctan2(gy, gx), np.pi)
    return energy, orientation_bin(angle)


def orientation_bin(angle: np.ndarray) -> np.ndarray:
    step = np.pi / ORIENT_BINS
    return np.mod(np.rint(angle / step), ORIENT_BINS).astype(np.int32)


def bin_direction(bin_id: int) -> tuple[float, float]:
    angle = bin_id * np.pi / ORIENT_BINS
    return float(np.cos(angle)), float(np.sin(angle))


def window_sums(sat, centers_y, centers_x, side: int) -> np.ndarray:
    half = side // 2
    height = sat.shape[0] - 1
    width = sat.shape[1] - 1
    y0, y1 = _clip_bounds(centers_y, half, 0, height)
    x0, x1 = _clip_bounds(centers_x, half, 0, width)
    return _rect(sat, y0, y1, x0, x1)


def _fundamental(stack: np.ndarray):
    best = np.max(stack, axis=-1)
    close = stack >= (HARMONIC_FRACTION * best[..., None])
    close &= stack > MIN_PERIODICITY
    any_close = np.any(close, axis=-1)
    index = np.argmax(close, axis=-1)
    periods = np.asarray(PERIODS, dtype=np.float32)[index]
    strength = np.take_along_axis(stack, index[..., None], axis=-1)[..., 0]
    periods = np.where(any_close, periods, np.float32(0))
    strength = np.where(any_close, strength, np.float32(0))
    return periods.astype(np.float32), strength.astype(np.float32)


def _fill_orientation_bin(grid, energy, bins, bin_id, centers_y, centers_x) -> None:
    layer = np.where(bins == bin_id, energy, np.float32(0))
    sat = integral_image(layer)
    for index, side in enumerate(WINDOWS):
        grid[:, :, index, bin_id] = window_sums(sat, centers_y, centers_x, side)


def _fill_period_maps(small, centers_y, centers_x, dominant, period, strength) -> None:
    sat_i = integral_image(small)
    sat_i2 = integral_image(small * small)
    ds_y = np.minimum(centers_y // DOWNSAMPLE, small.shape[0] - 1)
    ds_x = np.minimum(centers_x // DOWNSAMPLE, small.shape[1] - 1)
    for bin_id in range(ORIENT_BINS):
        stack = _orientation_ncc(small, sat_i, sat_i2, ds_y, ds_x, bin_id)
        chosen_period, chosen_strength = _fundamental(stack)
        _write_orientation(period, strength, dominant, bin_id, chosen_period, chosen_strength)


def _orientation_ncc(image, sat_i, sat_i2, ds_y, ds_x, bin_id: int) -> np.ndarray:
    stack = np.zeros((len(ds_y), len(ds_x), len(WINDOWS), len(PERIODS)), np.float32)
    for index, period in enumerate(PERIODS):
        stack[:, :, :, index] = _period_ncc(image, sat_i, sat_i2, ds_y, ds_x, bin_id, period)
    return stack


def _period_ncc(image, sat_i, sat_i2, ds_y, ds_x, bin_id: int, period: int) -> np.ndarray:
    dy, dx = _lag(bin_id, period, DOWNSAMPLE)
    out = np.zeros((len(ds_y), len(ds_x), len(WINDOWS)), np.float32)
    if dy == 0 and dx == 0:
        return out
    sat_ab = integral_image(_shifted_product(image, dy, dx))
    for index, side in enumerate(WINDOWS):
        out[:, :, index] = _one_window_ncc(sat_i, sat_i2, sat_ab, ds_y, ds_x, side // DOWNSAMPLE, dy, dx)
    return out


def _one_window_ncc(sat_i, sat_i2, sat_ab, ds_y, ds_x, side: int, dy: int, dx: int) -> np.ndarray:
    if side < 2:
        return np.zeros((len(ds_y), len(ds_x)), np.float32)
    half = side // 2
    limit_y = sat_i.shape[0] - 1
    limit_x = sat_i.shape[1] - 1
    y0, y1 = _clip_bounds(ds_y, half, 0, limit_y)
    x0, x1 = _clip_bounds(ds_x, half, 0, limit_x)
    sy0, sy1 = _clip_bounds(ds_y, half, dy, limit_y)
    sx0, sx1 = _clip_bounds(ds_x, half, dx, limit_x)
    ncc = _ncc_from_sums(
        _rect(sat_i, y0, y1, x0, x1),
        _rect(sat_i, sy0, sy1, sx0, sx1),
        _rect(sat_ab, y0, y1, x0, x1),
        _rect(sat_i2, y0, y1, x0, x1),
        _rect(sat_i2, sy0, sy1, sx0, sx1),
        float(side * side),
    )
    ncc[~_full_window(y0, y1, x0, x1, sy0, sy1, sx0, sx1, side)] = 0
    return ncc


def _window_mark_means(area_map, side_map, ink, centers_y, centers_x):
    sat_area = integral_image(area_map)
    sat_side = integral_image(side_map)
    sat_ink = integral_image(ink.astype(np.float32))
    areas = np.zeros((len(centers_y), len(centers_x), len(WINDOWS)), np.float32)
    sides = np.zeros_like(areas)
    for index, side in enumerate(WINDOWS):
        _fill_mark_window(areas, sides, sat_area, sat_side, sat_ink, centers_y, centers_x, index, side)
    return areas, sides


def _fill_mark_window(areas, sides, sat_area, sat_side, sat_ink, centers_y, centers_x, index, side) -> None:
    ink_sum = window_sums(sat_ink, centers_y, centers_x, side)
    present = ink_sum > 0
    area_sum = window_sums(sat_area, centers_y, centers_x, side)
    side_sum = window_sums(sat_side, centers_y, centers_x, side)
    areas[:, :, index][present] = area_sum[present] / ink_sum[present]
    sides[:, :, index][present] = side_sum[present] / ink_sum[present]


def _window_channels(hist, cosine, distance, ratio, bank, reference) -> list:
    period_abs = np.abs(bank["period"] - np.float32(reference.period))
    strength_abs = np.abs(bank["periodicity"] - np.float32(reference.periodicity))
    area_abs = np.abs(bank["mark_area"] - np.float32(reference.mark_area))
    side_abs = np.abs(bank["mark_side"] - np.float32(reference.mark_side))
    channels = []
    planes = (
        cosine,
        distance,
        bank["period"],
        ratio,
        period_abs,
        bank["periodicity"],
        strength_abs,
        bank["mark_area"],
        area_abs,
        bank["mark_side"],
        side_abs,
    )
    for index in range(len(WINDOWS)):
        channels.append(hist[:, :, index, :])
        channels.extend(plane[:, :, index, None] for plane in planes)
    return channels


def _query_channels(shape, reference: QueryReference) -> list:
    channels = [_full(shape, value) for value in reference.histogram]
    scalars = (reference.period, reference.periodicity, reference.mark_area, reference.mark_side)
    channels.extend(_full(shape, value) for value in scalars)
    return channels


def _query_histogram(gray, box) -> np.ndarray:
    energy, bins = _interior_gradient(gray, box)
    counts = np.zeros(ORIENT_BINS, np.float64)
    for bin_id in range(ORIENT_BINS):
        counts[bin_id] = float(energy[bins == bin_id].sum())
    total = float(counts.sum())
    if total <= 0.0:
        return np.zeros(ORIENT_BINS, np.float32)
    return (counts / total).astype(np.float32)


def _crop_period(gray, box, dominant: int) -> tuple[float, float]:
    crop = _box(gray, box).astype(np.float32)
    if min(crop.shape) < 4:
        return 0.0, 0.0
    values = np.zeros(len(PERIODS), np.float32)
    for index, period in enumerate(PERIODS):
        values[index] = _crop_lag_ncc(crop, dominant, period)
    chosen, strength = _fundamental(values.reshape(1, 1, 1, -1))
    return float(chosen[0, 0, 0]), float(strength[0, 0, 0])


def _crop_lag_ncc(crop, dominant: int, period: int) -> float:
    dy, dx = _lag(dominant, period, 1)
    if dy == 0 and dx == 0:
        return 0.0
    dest, src = _source_slices(dy, dx, crop.shape)
    if _empty_slice(dest):
        return 0.0
    if _slice_area(dest) < MIN_CROP_OVERLAP * crop.size:
        return 0.0
    return _flat_ncc(crop[dest], crop[src])


def _query_marks(area_map, side_map, ink, box) -> tuple[float, float]:
    x0, y0, x1, y1 = _box_tuple(box)
    ink_crop = ink[y0:y1, x0:x1] > 0
    if not np.any(ink_crop):
        return 0.0, 0.0
    area = area_map[y0:y1, x0:x1][ink_crop]
    side = side_map[y0:y1, x0:x1][ink_crop]
    return float(area.mean()), float(side.mean())


def _interior_gradient(gray, box):
    x0, y0, x1, y1 = _box_tuple(box)
    halo = 1
    yy0 = max(0, y0 - halo)
    xx0 = max(0, x0 - halo)
    yy1 = min(int(gray.shape[0]), y1 + halo)
    xx1 = min(int(gray.shape[1]), x1 + halo)
    energy, bins = gradient_bins(gray[yy0:yy1, xx0:xx1])
    top = y0 - yy0
    left = x0 - xx0
    return (
        energy[top : top + (y1 - y0), left : left + (x1 - x0)],
        bins[top : top + (y1 - y0), left : left + (x1 - x0)],
    )


def _normalize_hist(energy: np.ndarray) -> np.ndarray:
    total = energy.sum(axis=-1, keepdims=True)
    hist = np.zeros_like(energy)
    np.divide(energy, total, out=hist, where=total > 0)
    return hist


def _cosine(hist: np.ndarray, query: np.ndarray) -> np.ndarray:
    query_norm = float(np.linalg.norm(query))
    dots = hist @ np.asarray(query, dtype=np.float32)
    norms = np.linalg.norm(hist, axis=-1) * query_norm
    out = np.zeros(dots.shape, np.float32)
    np.divide(dots, norms, out=out, where=norms > 1e-8)
    return np.clip(out, -1.0, 1.0)


def _l1(hist: np.ndarray, query: np.ndarray) -> np.ndarray:
    return np.abs(hist - query.reshape(1, 1, 1, -1)).sum(axis=-1).astype(np.float32)


def _period_ratio(period: np.ndarray, query_period: float) -> np.ndarray:
    if query_period <= 0.0:
        return np.zeros_like(period)
    return (period / np.float32(query_period)).astype(np.float32)


def _downsample(gray):
    height, width = int(gray.shape[0]), int(gray.shape[1])
    ds_h = height // DOWNSAMPLE
    ds_w = width // DOWNSAMPLE
    if ds_h < 2 or ds_w < 2:
        return None
    return cv2.resize(np.asarray(gray, np.float32), (ds_w, ds_h), interpolation=cv2.INTER_AREA)


def _component_tables(stats):
    area = stats[:, cv2.CC_STAT_AREA].astype(np.float32)
    side = np.minimum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]).astype(np.float32)
    area[0] = 0
    side[0] = 0
    return area, side


def _write_orientation(period, strength, dominant, bin_id, chosen_period, chosen_strength) -> None:
    mask = dominant == bin_id
    period[mask] = chosen_period[mask]
    strength[mask] = chosen_strength[mask]


def _shifted_product(image, dy: int, dx: int) -> np.ndarray:
    product = np.zeros(image.shape, np.float32)
    dest, src = _source_slices(dy, dx, image.shape)
    if _empty_slice(dest):
        return product
    product[dest] = image[dest] * image[src]
    return product


def _ncc_from_sums(sum_a, sum_b, sum_ab, sum_a2, sum_b2, area: float) -> np.ndarray:
    inv = 1.0 / area
    mean_a = sum_a * inv
    mean_b = sum_b * inv
    var_a = np.maximum(sum_a2 * inv - mean_a * mean_a, 0.0)
    var_b = np.maximum(sum_b2 * inv - mean_b * mean_b, 0.0)
    cov = sum_ab * inv - mean_a * mean_b
    denom = np.sqrt(var_a * var_b)
    ncc = np.zeros(sum_a.shape, np.float32)
    ok = denom > 1e-6
    ncc[ok] = cov[ok] / denom[ok]
    return np.clip(ncc, -1.0, 1.0)


def _full_window(y0, y1, x0, x1, sy0, sy1, sx0, sx1, side: int) -> np.ndarray:
    return _span_ok(y0, y1, x0, x1, side) & _span_ok(sy0, sy1, sx0, sx1, side)


def _span_ok(y0, y1, x0, x1, side: int) -> np.ndarray:
    height = (y1 - y0)[:, None]
    width = (x1 - x0)[None, :]
    return (height == side) & (width == side)


def _clip_bounds(centers, half: int, shift: int, limit: int):
    start = np.clip(np.asarray(centers) - half + shift, 0, limit).astype(np.int64)
    stop = np.clip(np.asarray(centers) + half + shift, 0, limit).astype(np.int64)
    return start, stop


def _rect(sat, y0, y1, x0, x1) -> np.ndarray:
    return sat[y1[:, None], x1[None, :]] - sat[y0[:, None], x1[None, :]] - sat[y1[:, None], x0[None, :]] + sat[y0[:, None], x0[None, :]]


def _lag(bin_id: int, period: int, scale: int) -> tuple[int, int]:
    cosine, sine = bin_direction(bin_id)
    dx = int(round((period / scale) * cosine))
    dy = int(round((period / scale) * sine))
    return dy, dx


def _source_slices(dy: int, dx: int, shape):
    height, width = shape
    y_start = max(0, -dy)
    y_stop = min(height, height - dy)
    x_start = max(0, -dx)
    x_stop = min(width, width - dx)
    dest = (slice(y_start, y_stop), slice(x_start, x_stop))
    src = (slice(y_start + dy, y_stop + dy), slice(x_start + dx, x_stop + dx))
    return dest, src


def _empty_slice(dest) -> bool:
    y_slice, x_slice = dest
    return y_slice.stop <= y_slice.start or x_slice.stop <= x_slice.start


def _slice_area(dest) -> int:
    y_slice, x_slice = dest
    return (y_slice.stop - y_slice.start) * (x_slice.stop - x_slice.start)


def _flat_ncc(left, right) -> float:
    first = left.astype(np.float64).ravel()
    second = right.astype(np.float64).ravel()
    first = first - first.mean()
    second = second - second.mean()
    denom = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denom <= 1e-6:
        return 0.0
    return float(np.dot(first, second) / denom)


def _box(image, box) -> np.ndarray:
    x0, y0, x1, y1 = _box_tuple(box)
    return np.asarray(image[y0:y1, x0:x1])


def _box_tuple(box) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = (int(value) for value in box)
    return x0, y0, x1, y1


def _full(shape, value: float) -> np.ndarray:
    return np.full((shape[0], shape[1], 1), np.float32(value), dtype=np.float32)
