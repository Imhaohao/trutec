"""Query-relative hatch spectra: angle x spacing histograms of local Fourier amplitude.

Each descriptor bins the Hann-windowed FFT amplitude of a square patch by wave angle
(12 bins over 0-180 degrees) and spatial period (8 log-spaced bins from 3 to 64 px).
Comparing whole spectra is robust to the half/double-period errors of a single
autocorrelation lag, because a hatch and its harmonics land in the same bins.
"""
from __future__ import annotations

import numpy as np

WINDOWS = (64, 128)
ANGLE_BINS = 12
PERIOD_EDGES = np.geomspace(3.0, 64.0, 9)
BATCH = 1024
FUNDAMENTAL_FRACTION = 0.5
FEATURES_PER_WINDOW = ("spec_cos", "spec_peak_log_ratio", "spec_peak_angle_diff", "spec_concentration")


def feature_names() -> list[str]:
    names = [f"{name}_w{side}" for side in WINDOWS for name in FEATURES_PER_WINDOW]
    return names + [f"query_spec_concentration_w{side}" for side in WINDOWS]


def bin_layout(side: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fy = np.fft.fftfreq(side)[:, None]
    fx = np.fft.rfftfreq(side)[None, :]
    radius = np.hypot(fy, fx)
    with np.errstate(divide="ignore"):
        period = np.where(radius > 0, 1.0 / radius, np.inf)
    angle = np.mod(np.degrees(np.arctan2(fy, fx)), 180.0)
    angle_bin = np.minimum((angle / (180.0 / ANGLE_BINS)).astype(int), ANGLE_BINS - 1)
    period_bin = np.digitize(period, PERIOD_EDGES) - 1
    valid = (period_bin >= 0) & (period_bin < len(PERIOD_EDGES) - 1) & (period <= side / 2)
    flat = np.where(valid, period_bin * ANGLE_BINS + angle_bin, -1)
    centers = np.sqrt(PERIOD_EDGES[:-1] * PERIOD_EDGES[1:])
    return flat.ravel(), valid.ravel(), centers


def descriptors(patches: np.ndarray, side: int) -> np.ndarray:
    """patches: (n, side, side) float32 already windowed. Returns L2-normalized (n, bins)."""
    valid, columns = _assignment(side)
    amplitude = np.abs(np.fft.rfft2(patches)).reshape(len(patches), -1)[:, valid].astype(np.float32)
    out = np.zeros((len(patches), len(columns)), np.float32)
    for bin_id, members in enumerate(columns):
        if len(members):
            out[:, bin_id] = amplitude[:, members].max(axis=1)
    norm = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(norm, 1e-9)


_ASSIGNMENTS: dict[int, tuple[np.ndarray, list[np.ndarray]]] = {}


def _assignment(side: int) -> tuple[np.ndarray, list[np.ndarray]]:
    """Valid-frequency mask and, per (period, angle) bin, the valid-column indices it owns."""
    if side not in _ASSIGNMENTS:
        flat, valid, _centers = bin_layout(side)
        owner = flat[valid]
        bins = (len(PERIOD_EDGES) - 1) * ANGLE_BINS
        _ASSIGNMENTS[side] = (valid, [np.flatnonzero(owner == bin_id) for bin_id in range(bins)])
    return _ASSIGNMENTS[side]


def hann(side: int) -> np.ndarray:
    window = np.hanning(side).astype(np.float32)
    return np.outer(window, window)


def window_patches(gray: np.ndarray, ys: np.ndarray, xs: np.ndarray, side: int) -> np.ndarray:
    half = side // 2
    padded = np.pad(gray, half, mode="reflect")
    taper = hann(side)
    patches = np.stack([padded[y : y + side, x : x + side] for y, x in zip(ys, xs)]).astype(np.float32)
    patches -= patches.mean(axis=(1, 2), keepdims=True)
    return patches * taper


def query_patch(gray: np.ndarray, box, side: int) -> np.ndarray:
    x0, y0, x1, y1 = (int(value) for value in box)
    height, width = min(y1 - y0, side), min(x1 - x0, side)
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    crop = gray[cy - height // 2 : cy - height // 2 + height, cx - width // 2 : cx - width // 2 + width].astype(np.float32)
    crop = (crop - crop.mean()) * np.outer(np.hanning(crop.shape[0]), np.hanning(crop.shape[1])).astype(np.float32)
    patch = np.zeros((side, side), np.float32)
    top, left = (side - crop.shape[0]) // 2, (side - crop.shape[1]) // 2
    patch[top : top + crop.shape[0], left : left + crop.shape[1]] = crop
    return patch


def peak(descriptor: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fundamental period (longest period bin holding >= half the strongest radial energy) and peak angle."""
    grid = descriptor.reshape(*descriptor.shape[:-1], len(centers), ANGLE_BINS)
    radial = grid.sum(axis=-1)
    strong = radial >= FUNDAMENTAL_FRACTION * radial.max(axis=-1, keepdims=True)
    fundamental = len(centers) - 1 - np.argmax(strong[..., ::-1], axis=-1)
    angular = grid.sum(axis=-2)
    return centers[fundamental], np.argmax(angular, axis=-1) * (180.0 / ANGLE_BINS)


def cell_features(gray: np.ndarray, box, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    columns = []
    query_concentration = []
    for side in WINDOWS:
        _flat, _valid, centers = bin_layout(side)
        query = descriptors(query_patch(gray, box, side)[None], side)[0]
        query_period, query_angle = peak(query, centers)
        local = np.concatenate([descriptors(window_patches(gray, ys[s : s + BATCH], xs[s : s + BATCH], side), side) for s in range(0, len(ys), BATCH)]) if len(ys) else np.zeros((0, len(query)), np.float32)
        period, angle = peak(local, centers)
        difference = np.abs(angle - query_angle)
        columns += [
            local @ query,
            np.abs(np.log(period / query_period)),
            np.minimum(difference, 180.0 - difference),
            local.max(axis=1) ** 2,
        ]
        query_concentration.append(float(query.max() ** 2))
    stacked = np.stack(columns, axis=1) if len(ys) else np.zeros((0, len(columns)), np.float32)
    tail = np.tile(np.array(query_concentration, np.float32), (len(ys), 1))
    return np.concatenate([stacked, tail], axis=1).astype(np.float32)
