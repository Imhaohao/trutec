"""Query-conditioned hatch cadence features on the native stride-four grid."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math

import numpy as np
import cv2

STRIDE = 4
WINDOW_SIZE = 64
FEATURE_NAMES = (
    "cadence_peak_01", "cadence_offpeak_01", "cadence_peak_contrast_01",
    "cadence_neighbor_consistency_01", "cadence_available",
)


@dataclass(frozen=True)
class QuadrantEstimate:
    period: float
    angle: float
    power: float
    quadrant: int


@dataclass(frozen=True)
class LagState:
    period: float | None
    angle: float | None
    estimates: tuple[QuadrantEstimate | None, ...]

    @property
    def available(self) -> bool:
        return self.period is not None and self.angle is not None


def estimate_query_lag(gray: np.ndarray) -> LagState:
    """Estimate stable period and line-normal angle from an exact uint8 crop."""
    _validate_gray(gray)
    h, w = gray.shape
    split_y, split_x = h // 2, w // 2
    bounds = ((0, split_y, 0, split_x), (0, split_y, split_x, w),
              (split_y, h, 0, split_x), (split_y, h, split_x, w))
    estimates = tuple(_estimate_quadrant(gray[y0:y1, x0:x1], i)
                      for i, (y0, y1, x0, x1) in enumerate(bounds))
    valid = [item for item in estimates if item is not None]
    if len(valid) < 3:
        return LagState(None, None, estimates)
    clusters = [group for size in range(3, len(valid) + 1)
                for group in combinations(valid, size) if _is_consensus(group)]
    if not clusters:
        return LagState(None, None, estimates)
    best_size = max(map(len, clusters))
    clusters = [group for group in clusters if len(group) == best_size]
    clusters.sort(key=lambda group: (-sum(item.power for item in group),
                                     _cluster_key(group)))
    chosen = clusters[0]
    strongest = max(chosen, key=lambda item: (item.power, -item.quadrant))
    unwrapped = [strongest.angle + _signed_angle_delta(item.angle, strongest.angle)
                 for item in chosen]
    angle = float(np.median(unwrapped) % math.pi)
    return LagState(float(np.median([item.period for item in chosen])), angle, estimates)


def cadence_features(
    gray: np.ndarray, lag: LagState, grid_coordinates: np.ndarray,
    batch_size: int = 32,
) -> np.ndarray:
    """Return five [0,1] cadence features for requested global (gy,gx) cells."""
    _validate_gray(gray)
    coords = np.asarray(grid_coordinates)
    if coords.ndim != 2 or coords.shape[1] != 2 or not np.issubdtype(coords.dtype, np.integer):
        raise ValueError("grid_coordinates must be an integer array of shape (N, 2)")
    _validate_lag(lag)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    h, w = gray.shape
    grid_h, grid_w = math.ceil(h / STRIDE), math.ceil(w / STRIDE)
    _validate_coordinates(coords, grid_h, grid_w)
    output: np.ndarray = np.empty((len(coords), 5), dtype=np.float32)
    if not lag.available:
        output[:] = (0.5, 0.5, 0.5, 0.5, 0.0)
        return output
    assert lag.period is not None and lag.angle is not None
    period, angle = lag.period, lag.angle
    cy = np.minimum(coords[:, 0] * STRIDE + 2, h - 1).astype(np.float64)
    cx = np.minimum(coords[:, 1] * STRIDE + 2, w - 1).astype(np.float64)
    dy, dx = math.sin(angle), math.cos(angle)
    neighbor_coords = np.concatenate((
        np.column_stack((coords[:, 0] - 8, coords[:, 1])),
        np.column_stack((coords[:, 0] + 8, coords[:, 1])),
        np.column_stack((coords[:, 0], coords[:, 1] - 8)),
        np.column_stack((coords[:, 0], coords[:, 1] + 8)),
    ))
    neighbor_coords[:, 0] = _reflect(neighbor_coords[:, 0], grid_h)
    neighbor_coords[:, 1] = _reflect(neighbor_coords[:, 1], grid_w)
    neighbor_y = np.minimum(neighbor_coords[:, 0] * STRIDE + 2, h - 1).astype(np.float64)
    neighbor_x = np.minimum(neighbor_coords[:, 1] * STRIDE + 2, w - 1).astype(np.float64)
    all_y = np.concatenate((cy, neighbor_y))
    all_x = np.concatenate((cx, neighbor_x))
    peaks: np.ndarray = np.empty(len(all_y), dtype=np.float64)
    offpeaks: np.ndarray = np.empty(len(cy), dtype=np.float64)
    for start in range(0, len(all_y), batch_size):
        stop = min(start + batch_size, len(all_y))
        peaks[start:stop] = _rho_batch(gray, all_y[start:stop], all_x[start:stop],
                                       period, dx, dy)
    center_count = len(cy)
    for start in range(0, center_count, batch_size):
        stop = min(start + batch_size, center_count)
        offpeaks[start:stop] = (_rho_batch(gray, cy[start:stop], cx[start:stop],
                                           period / 2, dx, dy)
                                + _rho_batch(gray, cy[start:stop], cx[start:stop],
                                             period * 1.5, dx, dy)) / 2
    peak = peaks[:center_count]
    consistency = (1 - np.mean(np.abs(peaks[center_count:].reshape(4, -1) - peak), axis=0))
    contrast = peak - offpeaks
    values = np.column_stack(((peak + 1) / 2, (offpeaks + 1) / 2,
                              (contrast + 2) / 4,
                              ((consistency + 1) / 2) * np.maximum(0, contrast / 2),
                              np.ones(center_count)))
    return np.clip(values, 0, 1).astype(np.float32)


def query_cadence_descriptor(gray: np.ndarray, lag: LagState, batch_size: int = 32) -> np.ndarray:
    """Median five-channel descriptor over the exact crop's own stride-four grid."""
    _validate_gray(gray)
    gy, gx = np.indices((math.ceil(gray.shape[0] / 4), math.ceil(gray.shape[1] / 4)))
    cells = cadence_features(gray, lag, np.column_stack((gy.ravel(), gx.ravel())), batch_size)
    return np.median(cells, axis=0).astype(np.float32)


def cadence_features_tiled(
    gray: np.ndarray, lag: LagState, grid_coordinates: np.ndarray,
    core_size: int = 672, batch_size: int = 32,
) -> np.ndarray:
    """Extract requested cells grouped by globally aligned nonoverlapping cores."""
    _validate_gray(gray)
    coords = np.asarray(grid_coordinates)
    if core_size <= 0 or core_size > 672 or core_size % STRIDE:
        raise ValueError("core_size must be a positive stride multiple at most 672")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if coords.ndim != 2 or coords.shape[1] != 2 or not np.issubdtype(coords.dtype, np.integer):
        raise ValueError("grid_coordinates must be an integer array of shape (N, 2)")
    _validate_lag(lag)
    h, w = gray.shape
    grid_h, grid_w = math.ceil(h / STRIDE), math.ceil(w / STRIDE)
    _validate_coordinates(coords, grid_h, grid_w)
    if len(coords) == 0:
        return np.empty((0, 5), dtype=np.float32)
    if not lag.available:
        return np.tile(np.array([[0.5, 0.5, 0.5, 0.5, 0.0]], dtype=np.float32), (len(coords), 1))
    assert lag.period is not None and lag.angle is not None
    core_cells = core_size // STRIDE
    output: np.ndarray = np.empty((len(coords), 5), dtype=np.float32)
    tile_ids = (coords[:, 0] // core_cells) * math.ceil(grid_w / core_cells)
    tile_ids += coords[:, 1] // core_cells
    for tile_id in np.unique(tile_ids):
        positions = np.flatnonzero(tile_ids == tile_id)
        requested = coords[positions]
        neighbors = _neighbor_grid_coordinates(requested, grid_h, grid_w)
        requested_flat = requested[:, 0] * grid_w + requested[:, 1]
        all_flat = np.concatenate((requested_flat, neighbors[:, 0] * grid_w + neighbors[:, 1]))
        unique_flat, inverse = np.unique(all_flat, return_inverse=True)
        score_coords = np.column_stack(np.unravel_index(unique_flat, (grid_h, grid_w)))
        peak_scores = _rho_tiled_multi(
            gray, score_coords, (lag.period,), lag.angle, core_cells
        )[:, 0][inverse]
        count = len(requested)
        center_peak = peak_scores[:count]
        neighbor_peak = peak_scores[count:].reshape(4, -1)
        offpeak_scores = _rho_tiled_multi(
            gray, requested, (lag.period / 2, lag.period * 1.5), lag.angle, core_cells
        )
        offpeak = np.mean(offpeak_scores, axis=1)
        consistency = 1 - np.mean(np.abs(neighbor_peak - center_peak), axis=0)
        contrast = center_peak - offpeak
        values = np.column_stack(((center_peak + 1) / 2, (offpeak + 1) / 2,
                                  (contrast + 2) / 4,
                                  ((consistency + 1) / 2) * np.maximum(0, contrast / 2),
                                  np.ones(count)))
        output[positions] = np.clip(values, 0, 1).astype(np.float32)
    return output


def _validate_lag(lag: LagState) -> None:
    if (lag.period is None) != (lag.angle is None):
        raise ValueError("lag period and angle must both be available or both unavailable")
    if lag.period is None:
        return
    assert lag.angle is not None
    if not math.isfinite(lag.period) or not 4.0 <= lag.period <= 32.0:
        raise ValueError("lag period must be finite and in [4, 32]")
    if not math.isfinite(lag.angle) or not 0.0 <= lag.angle < math.pi:
        raise ValueError("lag angle must be finite and in [0, pi)")


def _validate_coordinates(coords: np.ndarray, grid_h: int, grid_w: int) -> None:
    if (np.any(coords < 0) or np.any(coords[:, 0] >= grid_h)
            or np.any(coords[:, 1] >= grid_w)):
        raise ValueError("grid coordinate lies outside the global stride-four grid")


def _neighbor_grid_coordinates(coords: np.ndarray, grid_h: int, grid_w: int) -> np.ndarray:
    neighbors = np.concatenate((
        coords + np.array([-8, 0]), coords + np.array([8, 0]),
        coords + np.array([0, -8]), coords + np.array([0, 8]),
    )).astype(np.int64)
    neighbors[:, 0] = _reflect(neighbors[:, 0], grid_h)
    neighbors[:, 1] = _reflect(neighbors[:, 1], grid_w)
    return neighbors


def _rho_tiled_multi(
    gray: np.ndarray,
    coordinates: np.ndarray,
    lags: tuple[float, ...],
    angle: float,
    core_cells: int,
) -> np.ndarray:
    """Compute exact local NCC by tile integrals over the valid overlap rectangle."""
    if len(coordinates) == 0:
        return np.empty((0, len(lags)), dtype=np.float64)
    output: np.ndarray = np.empty((len(coordinates), len(lags)), dtype=np.float64)
    tile_ids = (coordinates[:, 0] // core_cells) * math.ceil(gray.shape[1] / (core_cells * STRIDE))
    tile_ids += coordinates[:, 1] // core_cells
    for tile_id in np.unique(tile_ids):
        positions = np.flatnonzero(tile_ids == tile_id)
        tile_coords = coordinates[positions]
        core_y = int(tile_coords[0, 0] // core_cells * core_cells * STRIDE)
        core_x = int(tile_coords[0, 1] // core_cells * core_cells * STRIDE)
        values = _tile_correlations(gray, tile_coords, lags, angle, core_y, core_x, core_cells * STRIDE)
        output[positions] = values
    return output


def _tile_correlations(
    gray: np.ndarray,
    coordinates: np.ndarray,
    lags: tuple[float, ...],
    angle: float,
    core_y: int,
    core_x: int,
    core_size: int,
) -> np.ndarray:
    halo = 112
    tile_size = core_size + 2 * halo
    global_y: np.ndarray = np.arange(core_y - halo, core_y + core_size + halo, dtype=np.int64)
    global_x: np.ndarray = np.arange(core_x - halo, core_x + core_size + halo, dtype=np.int64)
    reflected_y = _reflect(global_y, gray.shape[0])
    reflected_x = _reflect(global_x, gray.shape[1])
    a = gray[reflected_y[:, None], reflected_x[None, :]].astype(np.float64) / 255.0
    a -= a.mean()
    target_y = global_y[:, None] + np.zeros((1, tile_size), dtype=np.float64)
    target_x = global_x[None, :] + np.zeros((tile_size, 1), dtype=np.float64)
    local_y = np.minimum(coordinates[:, 0] * STRIDE + 2, gray.shape[0] - 1) - core_y + halo
    local_x = np.minimum(coordinates[:, 1] * STRIDE + 2, gray.shape[1] - 1) - core_x + halo
    result: np.ndarray = np.empty((len(coordinates), len(lags)), dtype=np.float64)
    for lag_index, lag in enumerate(lags):
        delta_y, delta_x = lag * math.sin(angle), lag * math.cos(angle)
        b = _sample_reflected_source(gray, target_y + delta_y, target_x + delta_x)
        b -= b.mean()
        lower_y = max(-32, math.ceil(-32 - delta_y))
        upper_y = min(31, math.floor(31 - delta_y))
        lower_x = max(-32, math.ceil(-32 - delta_x))
        upper_x = min(31, math.floor(31 - delta_x))
        y1, y2 = local_y + lower_y, local_y + upper_y + 1
        x1, x2 = local_x + lower_x, local_x + upper_x + 1
        integral_a, integral_b = cv2.integral(a), cv2.integral(b)
        integral_aa, integral_bb = cv2.integral(a * a), cv2.integral(b * b)
        integral_ab = cv2.integral(a * b)
        sum_a = _integral_rectangle(integral_a, y1, x1, y2, x2)
        sum_b = _integral_rectangle(integral_b, y1, x1, y2, x2)
        sum_aa = _integral_rectangle(integral_aa, y1, x1, y2, x2)
        sum_bb = _integral_rectangle(integral_bb, y1, x1, y2, x2)
        sum_ab = _integral_rectangle(integral_ab, y1, x1, y2, x2)
        count = (y2 - y1) * (x2 - x1)
        covariance = sum_ab - sum_a * sum_b / count
        raw_energy_a = sum_aa - sum_a * sum_a / count
        raw_energy_b = sum_bb - sum_b * sum_b / count
        total_scale_a = max(float(integral_aa[-1, -1]), float(integral_a[-1, -1]))
        total_scale_b = max(float(integral_bb[-1, -1]), float(integral_b[-1, -1]))
        tolerance_a = np.finfo(np.float64).eps * 64 * np.maximum(
            total_scale_a, np.maximum(sum_aa, sum_a * sum_a / count)
        )
        tolerance_b = np.finfo(np.float64).eps * 64 * np.maximum(
            total_scale_b, np.maximum(sum_bb, sum_b * sum_b / count)
        )
        energy_a = np.where(raw_energy_a > tolerance_a, raw_energy_a, 0.0)
        energy_b = np.where(raw_energy_b > tolerance_b, raw_energy_b, 0.0)
        denominator = np.sqrt(energy_a * energy_b)
        result[:, lag_index] = np.divide(
            covariance, denominator, out=np.zeros_like(covariance), where=denominator > 0
        ).clip(-1, 1)
    return result


def _sample_reflected_source(source: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    y_low, x_low = np.floor(y).astype(np.int64), np.floor(x).astype(np.int64)
    fy, fx = y - y_low, x - x_low
    height, width = source.shape
    y0, y1 = _reflect(y_low, height), _reflect(y_low + 1, height)
    x0, x1 = _reflect(x_low, width), _reflect(x_low + 1, width)
    return (source[y0, x0].astype(np.float64) * (1 - fy) * (1 - fx)
            + source[y0, x1].astype(np.float64) * (1 - fy) * fx
            + source[y1, x0].astype(np.float64) * fy * (1 - fx)
            + source[y1, x1].astype(np.float64) * fy * fx) / 255.0


def _integral_rectangle(
    integral: np.ndarray, y1: np.ndarray, x1: np.ndarray, y2: np.ndarray, x2: np.ndarray
) -> np.ndarray:
    return integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1]


def _estimate_quadrant(image: np.ndarray, quadrant: int) -> QuadrantEstimate | None:
    h, w = image.shape
    if h < 2 or w < 2:
        return None
    signal = 1.0 - image.astype(np.float64) / 255.0
    signal -= signal.mean()
    wy, wx = np.hanning(h), np.hanning(w)
    window_energy = float(np.sum(wy[:, None] ** 2 * wx[None, :] ** 2))
    weighted = signal * wy[:, None] * wx[None, :]
    if window_energy == 0 or float(np.sum(weighted * weighted)) == 0:
        return None
    power = np.abs(np.fft.fft2(weighted)) ** 2 / window_energy
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    radial = np.hypot(fx, fy)
    valid = (radial >= 1 / 32) & (radial <= 1 / 4) & (radial > 0)
    canonical = (fy > 0) | ((fy == 0) & (fx > 0))
    valid &= canonical
    candidates = np.argwhere(valid)
    if len(candidates) == 0:
        return None
    ranked = []
    for iy, ix in candidates:
        frequency = float(radial[iy, ix])
        angle = math.atan2(float(fy[iy, 0]), float(fx[0, ix])) % math.pi
        ranked.append((-float(power[iy, ix]), 1 / frequency, angle, int(iy), int(ix)))
    _, period, angle, iy, ix = min(ranked)
    if float(power.sum() - power[0, 0]) <= 0:
        return None
    support = w * abs(math.cos(angle)) + h * abs(math.sin(angle))
    if period > support / 3:
        return None
    return QuadrantEstimate(period, angle, float(power[iy, ix]), quadrant)


def _is_consensus(group: tuple[QuadrantEstimate, ...]) -> bool:
    periods = [item.period for item in group]
    return (max(periods) / min(periods) <= 1.20
            and all(_angle_distance(a.angle, b.angle) <= math.radians(15)
                    for a, b in combinations(group, 2)))


def _cluster_key(group: tuple[QuadrantEstimate, ...]) -> tuple[float, float, int]:
    strongest = max(group, key=lambda item: (item.power, -item.quadrant))
    angles = [strongest.angle + _signed_angle_delta(item.angle, strongest.angle) for item in group]
    return float(np.median([item.period for item in group])), float(np.median(angles) % math.pi), min(x.quadrant for x in group)


def _angle_distance(a: float, b: float) -> float:
    return abs((a - b + math.pi / 2) % math.pi - math.pi / 2)


def _signed_angle_delta(a: float, b: float) -> float:
    return (a - b + math.pi / 2) % math.pi - math.pi / 2


def _rho_batch(gray: np.ndarray, cy: np.ndarray, cx: np.ndarray,
               lag: float, dx: float, dy: float) -> np.ndarray:
    offsets: np.ndarray = np.arange(-32, 32, dtype=np.float64)
    oy: np.ndarray
    ox: np.ndarray
    oy, ox = np.meshgrid(offsets, offsets, indexing="ij")
    y0 = cy[:, None, None] + oy
    x0 = cx[:, None, None] + ox
    y1, x1 = y0 + lag * dy, x0 + lag * dx
    support = ((y1 >= cy[:, None, None] - 32) & (y1 <= cy[:, None, None] + 31)
               & (x1 >= cx[:, None, None] - 32) & (x1 <= cx[:, None, None] + 31))
    a = _bilinear(gray, y0, x0)
    b = _bilinear(gray, y1, x1)
    count = support.sum(axis=(1, 2)).astype(np.float64)
    safe = np.maximum(count, 1)
    mean_a = np.sum(a * support, axis=(1, 2)) / safe
    mean_b = np.sum(b * support, axis=(1, 2)) / safe
    da, db = (a - mean_a[:, None, None]) * support, (b - mean_b[:, None, None]) * support
    cov = np.sum(da * db, axis=(1, 2))
    energy_a, energy_b = np.sum(da * da, axis=(1, 2)), np.sum(db * db, axis=(1, 2))
    tolerance_a = np.finfo(np.float64).eps * 64 * np.sum(a * a * support, axis=(1, 2))
    tolerance_b = np.finfo(np.float64).eps * 64 * np.sum(b * b * support, axis=(1, 2))
    energy_a = np.where(energy_a > tolerance_a, energy_a, 0.0)
    energy_b = np.where(energy_b > tolerance_b, energy_b, 0.0)
    denom = np.sqrt(energy_a * energy_b)
    return np.divide(cov, denom, out=np.zeros_like(cov), where=(count >= 2) & (denom > 0)).clip(-1, 1)


def _bilinear(gray: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    ylo, xlo = np.floor(y).astype(np.int64), np.floor(x).astype(np.int64)
    fy, fx = y - ylo, x - xlo
    height, width = gray.shape
    y0, y1 = _reflect(ylo, height), _reflect(ylo + 1, height)
    x0, x1 = _reflect(xlo, width), _reflect(xlo + 1, width)
    return (gray[y0, x0].astype(np.float64) * (1 - fy) * (1 - fx)
            + gray[y0, x1].astype(np.float64) * (1 - fy) * fx
            + gray[y1, x0].astype(np.float64) * fy * (1 - fx)
            + gray[y1, x1].astype(np.float64) * fy * fx) / 255.0


def _reflect(index: np.ndarray, length: int) -> np.ndarray:
    if length <= 1:
        return np.zeros_like(index)
    period = 2 * length - 2
    folded = np.mod(index, period)
    return np.where(folded < length, folded, period - folded)


def _validate_gray(gray: np.ndarray) -> None:
    if not isinstance(gray, np.ndarray) or gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError("gray must be a two-dimensional uint8 array")
    if min(gray.shape) <= 0:
        raise ValueError("gray dimensions must be positive")
