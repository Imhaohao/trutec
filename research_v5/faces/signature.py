"""Periodic exemplar and per-window ink signatures.

The exemplar window is the subwindow with the strongest autocorrelation peak
at a nonzero period. Standard deviation is never the selection score.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from settings import PERIODIC, PLAIN, SOLID, Settings

_SIDES = (24, 32, 48, 64, 96)


@dataclass(frozen=True)
class Exemplar:
    kind: int
    strength: float
    period: float
    orient: float
    strength2: float
    period2: float
    orient2: float
    thickness: float
    density: float
    hue: float
    sat: float
    value: float
    template: np.ndarray
    origin_y: int
    origin_x: int


@dataclass(frozen=True)
class WindowSig:
    kind: int
    peaks: tuple[tuple[float, float, float], ...]
    thickness: float
    density: float
    hue: float
    sat: float
    value: float


@dataclass
class SigGrid:
    kind: np.ndarray
    strength: np.ndarray
    period: np.ndarray
    orient: np.ndarray
    strength2: np.ndarray
    period2: np.ndarray
    orient2: np.ndarray
    peak_s: np.ndarray
    peak_p: np.ndarray
    peak_o: np.ndarray
    thickness: np.ndarray
    density: np.ndarray
    hue: np.ndarray
    sat: np.ndarray
    value: np.ndarray
    origin_y: np.ndarray
    origin_x: np.ndarray
    center_y: np.ndarray
    center_x: np.ndarray


def measure_sheet(rgb: np.ndarray, gray: np.ndarray, settings: Settings) -> SigGrid:
    origins_y = _origins(gray.shape[0], settings.tile, settings.tile_stride)
    origins_x = _origins(gray.shape[1], settings.tile, settings.tile_stride)
    grid = _empty_grid(origins_y, origins_x)
    side_y, side_x = _window_sides(gray.shape, settings.tile)
    grid.center_y = origins_y.astype(np.float32) + (side_y / 2.0)
    grid.center_x = origins_x.astype(np.float32) + (side_x / 2.0)
    for iy, origin_y in enumerate(origins_y.tolist()):
        _fill_row(grid, iy, rgb, gray, int(origin_y), origins_x, side_y, side_x, settings)
    return grid


def find_exemplar(
    rgb: np.ndarray,
    gray: np.ndarray,
    boxes: tuple[tuple[int, int, int, int], ...],
    settings: Settings,
) -> Exemplar | None:
    best: Exemplar | None = None
    best_rank = -1.0
    for box in boxes:
        found, rank = _best_in_crop(rgb, gray, box, settings)
        if found is not None and rank > best_rank:
            best = found
            best_rank = rank
    return best


def _fill_row(
    grid: SigGrid,
    iy: int,
    rgb: np.ndarray,
    gray: np.ndarray,
    origin_y: int,
    origins_x: np.ndarray,
    side_y: int,
    side_x: int,
    settings: Settings,
) -> None:
    for ix, origin_x in enumerate(origins_x.tolist()):
        x0 = int(origin_x)
        crop_gray = gray[origin_y:origin_y + side_y, x0:x0 + side_x]
        if int(crop_gray.min()) >= settings.paper_level:
            continue
        crop_rgb = rgb[origin_y:origin_y + side_y, x0:x0 + side_x]
        _store(grid, iy, ix, _measure(crop_rgb, crop_gray, settings))


def _best_in_crop(
    rgb: np.ndarray,
    gray: np.ndarray,
    box: tuple[int, int, int, int],
    settings: Settings,
) -> tuple[Exemplar | None, float]:
    x0, y0, x1, y1 = box
    crop_rgb = rgb[y0:y1, x0:x1]
    crop_gray = gray[y0:y1, x0:x1]
    if crop_gray.size == 0:
        return None, -1.0
    best: Exemplar | None = None
    best_rank = -1.0
    for origin_y, origin_x, side in _candidate_windows(crop_gray.shape):
        window_rgb = crop_rgb[origin_y:origin_y + side, origin_x:origin_x + side]
        window_gray = crop_gray[origin_y:origin_y + side, origin_x:origin_x + side]
        measured = _measure(window_rgb, window_gray, settings)
        rank = _rank(measured)
        if rank <= best_rank:
            continue
        best_rank = rank
        best = _exemplar_from(measured, window_gray, y0 + origin_y, x0 + origin_x)
    return best, best_rank


def _exemplar_from(measured: WindowSig, window_gray: np.ndarray, origin_y: int, origin_x: int) -> Exemplar:
    strength, period, orient = _nth_peak(measured.peaks, 0)
    strength2, period2, orient2 = _nth_peak(measured.peaks, 1)
    return Exemplar(
        measured.kind, strength, period, orient, strength2, period2, orient2,
        measured.thickness, measured.density, measured.hue, measured.sat, measured.value,
        template=np.ascontiguousarray(window_gray),
        origin_y=origin_y,
        origin_x=origin_x,
    )


def _measure(rgb: np.ndarray, gray: np.ndarray, settings: Settings) -> WindowSig:
    thickness, density, hue, sat, value = _ink_stats(rgb, gray, settings)
    if _is_solid(rgb, gray, settings):
        return WindowSig(SOLID, (), thickness, density, hue, sat, value)
    peaks = tuple(_periodic_peaks(gray, settings))
    if not peaks or peaks[0][0] < settings.min_strength:
        return WindowSig(PLAIN, peaks, thickness, density, hue, sat, value)
    return WindowSig(PERIODIC, peaks, thickness, density, hue, sat, value)


def _periodic_peaks(
    gray: np.ndarray, settings: Settings
) -> list[tuple[float, float, float]]:
    ac = _autocorr(gray)
    return _peaks_from_autocorr(ac, settings.min_period, _max_period(gray.shape, settings.max_period))


def _is_solid(rgb: np.ndarray, gray: np.ndarray, settings: Settings) -> bool:
    if rgb.size == 0:
        return False
    cover = float((~_paper(rgb, settings)).mean())
    strokes = float((gray <= 165).mean())
    deviation = float(rgb.astype(np.float32).std(axis=(0, 1)).mean())
    return cover >= settings.solid_cover and deviation <= settings.solid_std and strokes <= settings.solid_stroke


def _ink_stats(
    rgb: np.ndarray, gray: np.ndarray, settings: Settings
) -> tuple[float, float, float, float, float]:
    strokes = gray <= 165
    ink = strokes if float(strokes.mean()) >= 0.01 else ~_paper(rgb, settings)
    density = float(ink.mean())
    if not np.any(ink):
        return 0.0, 0.0, 0.0, 0.0, 1.0
    thickness = _median_thickness(ink)
    hue, sat, value = _median_color(rgb, ink)
    return thickness, density, hue, sat, value


def _median_thickness(ink: np.ndarray) -> float:
    dist = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 3)
    return float(np.median(dist[ink]))


def _median_color(rgb: np.ndarray, ink: np.ndarray) -> tuple[float, float, float]:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0][ink].astype(np.float64)
    sat = hsv[:, :, 1][ink].astype(np.float64) / 255.0
    value = hsv[:, :, 2][ink].astype(np.float64) / 255.0
    angle = hue * (np.pi / 90.0)
    weight = sat + 0.05
    mean_angle = math.atan2(float(np.average(np.sin(angle), weights=weight)), float(np.average(np.cos(angle), weights=weight)))
    hue_deg = (math.degrees(mean_angle) % 360.0) * 0.5
    return hue_deg, float(np.median(sat)), float(np.median(value))


def _autocorr(gray: np.ndarray) -> np.ndarray:
    field = gray.astype(np.float32)
    field -= float(field.mean())
    if min(field.shape) >= 48:
        field *= np.hanning(field.shape[0]).astype(np.float32)[:, None]
        field *= np.hanning(field.shape[1]).astype(np.float32)[None, :]
    spec = np.fft.rfft2(field)
    corr = np.fft.irfft2(spec * np.conj(spec), s=field.shape)
    corr = np.fft.fftshift(corr)
    center = corr[corr.shape[0] // 2, corr.shape[1] // 2]
    if center <= 1e-6:
        return np.zeros(field.shape, dtype=np.float32)
    return (corr / center).astype(np.float32)


def _peaks_from_autocorr(
    ac: np.ndarray, min_period: int, max_period: int
) -> list[tuple[float, float, float]]:
    if max_period <= min_period:
        return []
    work = _peak_band(ac, min_period, max_period)
    center_y, center_x = ac.shape[0] // 2, ac.shape[1] // 2
    found: list[tuple[float, float, float]] = []
    skips = 0
    while len(found) < 4 and skips < 24:
        picked = _read_peak(work, ac, center_y, center_x, min_period, max_period)
        if picked is None:
            break
        payload, py, px, ridge = picked
        _suppress(work, py, px, max(3, min_period // 2))
        _suppress(work, 2 * center_y - py, 2 * center_x - px, max(3, min_period // 2))
        if ridge:
            skips += 1
            continue
        found.append(payload)
    return found


def _peak_band(ac: np.ndarray, min_period: int, max_period: int) -> np.ndarray:
    height, width = ac.shape
    yy, xx = np.ogrid[:height, :width]
    radius = np.hypot(yy - (height // 2), xx - (width // 2))
    band = (radius >= min_period) & (radius <= max_period) & _local_maxima(ac)
    return np.where(band, ac, -np.inf)


def _local_maxima(ac: np.ndarray) -> np.ndarray:
    padded = np.pad(ac, 1, mode="constant", constant_values=-np.inf)
    center = padded[1:-1, 1:-1]
    local = np.ones(ac.shape, dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            _keep_neighbor(local, center, padded, ac.shape, dy, dx)
    return local


def _keep_neighbor(
    local: np.ndarray,
    center: np.ndarray,
    padded: np.ndarray,
    shape: tuple[int, int],
    dy: int,
    dx: int,
) -> None:
    if dy == 0 and dx == 0:
        return
    height, width = shape
    neighbor = padded[1 + dy:height + 1 + dy, 1 + dx:width + 1 + dx]
    local &= center >= neighbor


def _read_peak(
    work: np.ndarray,
    ac: np.ndarray,
    center_y: int,
    center_x: int,
    min_period: int,
    max_period: int,
) -> tuple[tuple[float, float, float], int, int, bool] | None:
    index = int(np.argmax(work))
    strength = float(work.flat[index])
    if not math.isfinite(strength) or strength < 0.12:
        return None
    py, px = (int(v) for v in np.unravel_index(index, work.shape))
    ridge = _is_ridge(ac, py, px, center_y, center_x, min_period, max_period)
    dy = py - center_y
    dx = px - center_x
    period = float(math.hypot(dy, dx))
    orient = (math.degrees(math.atan2(dy, dx)) + 90.0) % 180.0
    return (strength, period, orient), py, px, ridge


def _is_ridge(
    ac: np.ndarray,
    py: int,
    px: int,
    center_y: int,
    center_x: int,
    min_period: int,
    max_period: int,
) -> bool:
    dy = py - center_y
    dx = px - center_x
    period = math.hypot(dy, dx)
    if period >= max_period - 1.25 or period < 1.0:
        return True
    samples = _ray_samples(ac, center_y, center_x, dy / period, dx / period, min_period, max_period)
    if len(samples) < 3:
        return True
    return float(ac[py, px]) < float(np.median(samples)) + 0.12


def _ray_samples(
    ac: np.ndarray,
    center_y: int,
    center_x: int,
    uy: float,
    ux: float,
    min_period: int,
    max_period: int,
) -> list[float]:
    samples: list[float] = []
    height, width = ac.shape
    for radius in range(min_period, max_period + 1, 2):
        y = int(round(center_y + uy * radius))
        x = int(round(center_x + ux * radius))
        if 0 <= y < height and 0 <= x < width:
            samples.append(float(ac[y, x]))
    return samples


def _suppress(work: np.ndarray, y: int, x: int, radius: int) -> None:
    height, width = work.shape
    y0, y1 = max(0, y - radius), min(height, y + radius + 1)
    x0, x1 = max(0, x - radius), min(width, x + radius + 1)
    if y0 < y1 and x0 < x1:
        work[y0:y1, x0:x1] = -np.inf


def _paper(rgb: np.ndarray, settings: Settings) -> np.ndarray:
    return np.all(rgb >= settings.paper_level, axis=2)


def _nth_peak(peaks: tuple[tuple[float, float, float], ...], index: int) -> tuple[float, float, float]:
    if index < len(peaks):
        return peaks[index]
    return (0.0, 0.0, 0.0)


def _rank(measured: WindowSig) -> float:
    if measured.kind == PERIODIC:
        return 2.0 + measured.peaks[0][0]
    if measured.kind == SOLID:
        return 1.0
    if measured.peaks:
        return measured.peaks[0][0]
    return 0.0


def _candidate_windows(shape: tuple[int, int]) -> list[tuple[int, int, int]]:
    height, width = shape
    limit = min(height, width)
    sides = [side for side in _SIDES if side <= limit]
    if not sides:
        return [(0, 0, limit)] if limit >= 8 else []
    windows: list[tuple[int, int, int]] = []
    for side in sides:
        stride = max(2, side // 6)
        for y in range(0, height - side + 1, stride):
            for x in range(0, width - side + 1, stride):
                windows.append((y, x, side))
    return windows


def _origins(length: int, side: int, stride: int) -> np.ndarray:
    if length <= side:
        return np.zeros(1, dtype=np.int32)
    starts = list(range(0, length - side + 1, stride))
    if starts[-1] != length - side:
        starts.append(length - side)
    return np.asarray(starts, dtype=np.int32)


def _window_sides(shape: tuple[int, int], tile: int) -> tuple[int, int]:
    return min(tile, shape[0]), min(tile, shape[1])


def _max_period(shape: tuple[int, int], cap: int) -> int:
    return max(0, min(cap, min(shape) // 2 - 1))


def _empty_grid(origins_y: np.ndarray, origins_x: np.ndarray) -> SigGrid:
    shape = (origins_y.size, origins_x.size)
    zeros = np.zeros(shape, dtype=np.float32)
    return SigGrid(
        kind=np.zeros(shape, dtype=np.int8),
        strength=zeros.copy(),
        period=zeros.copy(),
        orient=zeros.copy(),
        strength2=zeros.copy(),
        period2=zeros.copy(),
        orient2=zeros.copy(),
        peak_s=np.zeros(shape + (4,), dtype=np.float32),
        peak_p=np.zeros(shape + (4,), dtype=np.float32),
        peak_o=np.zeros(shape + (4,), dtype=np.float32),
        thickness=zeros.copy(),
        density=zeros.copy(),
        hue=zeros.copy(),
        sat=zeros.copy(),
        value=np.ones(shape, dtype=np.float32),
        origin_y=origins_y,
        origin_x=origins_x,
        center_y=origins_y.astype(np.float32),
        center_x=origins_x.astype(np.float32),
    )


def _store(grid: SigGrid, iy: int, ix: int, measured: WindowSig) -> None:
    grid.kind[iy, ix] = measured.kind
    _store_peaks(grid, iy, ix, measured.peaks)
    grid.thickness[iy, ix] = measured.thickness
    grid.density[iy, ix] = measured.density
    grid.hue[iy, ix] = measured.hue
    grid.sat[iy, ix] = measured.sat
    grid.value[iy, ix] = measured.value


def _store_peaks(
    grid: SigGrid, iy: int, ix: int, peaks: tuple[tuple[float, float, float], ...]
) -> None:
    for index, (strength, period, orient) in enumerate(peaks[:4]):
        grid.peak_s[iy, ix, index] = strength
        grid.peak_p[iy, ix, index] = period
        grid.peak_o[iy, ix, index] = orient
    if peaks:
        grid.strength[iy, ix], grid.period[iy, ix], grid.orient[iy, ix] = peaks[0]
    if len(peaks) > 1:
        grid.strength2[iy, ix], grid.period2[iy, ix], grid.orient2[iy, ix] = peaks[1]
