"""Multi-scale local descriptors for one drawing, built with integral images.

Every channel is a function of a rectangle on the full sheet. Grid cells and
augmented query cells use that same function, so a pixel score does not depend
on which pixels are later sampled.
"""

from __future__ import annotations

from typing import NamedTuple

import cv2
import numpy as np

STRIDE = 8
WINDOWS = (8, 16, 32, 64)
HOG_WINDOWS = (8, 16, 32)
STRUCTURE_WINDOWS = (16, 32, 64)
COLOR_WINDOWS = (8, 16, 32)
SPECTRAL_WINDOWS = (16, 32, 64)
SPECTRAL_PERIODS = (4, 8, 12, 16, 24, 32, 48)
SPECTRAL_ANGLE_COUNT = 8
HOG_BINS = 8
DENSITY_KINDS = ("ink_mean", "ink_std", "chroma_mean", "chroma_std")
STRUCTURE_KINDS = ("coherence", "cos2", "sin2", "tensor_energy")
SPECTRAL_CLIP = 8.0
SPECTRAL_STD_EPS = 0.02


class Channel(NamedTuple):
    name: str
    window: int
    kind: str
    arg: int


class SheetFeatures(NamedTuple):
    grid_y: np.ndarray
    grid_x: np.ndarray
    grid: np.ndarray
    points: tuple[np.ndarray, ...]


def channel_table() -> tuple[Channel, ...]:
    """Fixed descriptor layout. `arg` is a color plane, HOG bin, or unused."""
    channels: list[Channel] = []
    channels.extend(_density_channels())
    channels.extend(_color_channels())
    channels.extend(_hog_channels())
    channels.extend(_gradient_energy_channels())
    channels.extend(_structure_channels())
    channels.extend(_spectral_channels())
    return tuple(channels)


def support_window(box: tuple[int, int, int, int], minimum: int = 12) -> int:
    """Largest window that still yields enough fully-inside centers."""
    x0, y0, x1, y1 = _box_tuple(box)
    for window in (64, 32, 16, 8):
        if _center_count(x0, y0, x1, y1, window, 2) >= minimum:
            return window
    return 8


def positive_centers(
    box: tuple[int, int, int, int],
    window: int,
    step: int = 2,
    cap: int = 480,
) -> tuple[np.ndarray, np.ndarray]:
    """Centers whose full `window` lies inside the box, with 2 px shifts."""
    x0, y0, x1, y1 = _box_tuple(box)
    half = window // 2
    y_first, y_last = y0 + half, y1 - half
    x_first, x_last = x0 + half, x1 - half
    if y_last < y_first or x_last < x_first:
        return _fallback_center(y0, y1, x0, x1)
    step_use = step
    ys, xs = _axis_range(y_first, y_last, step_use), _axis_range(x_first, x_last, step_use)
    while int(ys.size * xs.size) > cap and step_use < 16:
        step_use *= 2
        ys = _axis_range(y_first, y_last, step_use)
        xs = _axis_range(x_first, x_last, step_use)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    return yy.ravel().astype(np.int32), xx.ravel().astype(np.int32)


def describe_sheet(
    rgb: np.ndarray,
    point_groups: tuple[tuple[np.ndarray, np.ndarray], ...],
    stride: int = STRIDE,
) -> SheetFeatures:
    """Grid descriptors plus one feature row-set per positive center group."""
    _require_rgb(rgb)
    axes_y, axes_x = grid_axes(int(rgb.shape[0]), int(rgb.shape[1]), stride)
    table = channel_table()
    grid = np.zeros((axes_y.size, axes_x.size, len(table)), dtype=np.float32)
    points = tuple(np.zeros((ys.size, len(table)), dtype=np.float32) for ys, _xs in point_groups)
    _fill_appearance(rgb, grid, points, axes_y, axes_x, point_groups, table)
    _fill_gradients(rgb, grid, points, axes_y, axes_x, point_groups, table)
    _fill_spectral(rgb, grid, points, axes_y, axes_x, point_groups, table)
    _normalize_hog(grid, table)
    for block in points:
        _normalize_hog(block, table)
    np.nan_to_num(grid, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    for block in points:
        np.nan_to_num(block, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return SheetFeatures(axes_y, axes_x, grid, points)


def grid_axes(height: int, width: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    ys = np.arange(stride // 2, height, stride, dtype=np.int32)
    xs = np.arange(stride // 2, width, stride, dtype=np.int32)
    return ys, xs


def column_index(table: tuple[Channel, ...], kind: str, window: int, arg: int = 0) -> int:
    for index, channel in enumerate(table):
        if channel.kind == kind and channel.window == window and channel.arg == arg:
            return index
    raise KeyError((kind, window, arg))


def self_check_geometry() -> None:
    """Window rules and integral sums, on a tiny sheet (no dataset)."""
    if support_window((0, 0, 32, 32)) != 16:
        raise RuntimeError("32 px box should use the 16 px window")
    if support_window((0, 0, 192, 192)) != 64:
        raise RuntimeError("192 px box should use the 64 px window")
    if support_window((0, 0, 96, 32)) != 32:
        raise RuntimeError("96x32 box should use the 32 px window")
    ys, xs = positive_centers((10, 20, 42, 52), 16, step=2)
    if np.any(ys - 8 < 20) or np.any(ys + 8 > 52) or np.any(xs - 8 < 10) or np.any(xs + 8 > 42):
        raise RuntimeError("positive centers escaped the query box")
    sheet = np.zeros((48, 48, 3), dtype=np.uint8)
    sheet[:, :, 1] = 80
    described = describe_sheet(sheet, ((ys[:1], xs[:1]),), stride=8)
    table = channel_table()
    ink_at = column_index(table, "ink_mean", 8)
    # Flat green: ink mean is constant, including at the augmented center.
    expected = 1.0 - (0.2126 * 0 + 0.7152 * 80 + 0.0722 * 0) / 255.0
    got = float(described.grid[2, 2, ink_at])
    if abs(got - expected) > 1e-3:
        raise RuntimeError(f"ink mean {got} != {expected}")


def _density_channels() -> list[Channel]:
    channels = []
    for window in WINDOWS:
        for kind in DENSITY_KINDS:
            channels.append(Channel(f"{kind}_w{window}", window, kind, 0))
    return channels


def _color_channels() -> list[Channel]:
    names = ("red_mean", "green_mean", "blue_mean")
    return [
        Channel(f"{names[plane]}_w{window}", window, "color", plane)
        for window in COLOR_WINDOWS
        for plane in range(3)
    ]


def _hog_channels() -> list[Channel]:
    return [
        Channel(f"hog_b{bin_id}_w{window}", window, "hog", bin_id)
        for window in HOG_WINDOWS
        for bin_id in range(HOG_BINS)
    ]


def _gradient_energy_channels() -> list[Channel]:
    return [Channel(f"grad_energy_w{window}", window, "grad_energy", 0) for window in WINDOWS]


def _structure_channels() -> list[Channel]:
    return [
        Channel(f"{kind}_w{window}", window, kind, 0)
        for window in STRUCTURE_WINDOWS
        for kind in STRUCTURE_KINDS
    ]


def _spectral_channels() -> list[Channel]:
    channels = []
    for window in SPECTRAL_WINDOWS:
        for period in SPECTRAL_PERIODS:
            if period > window:
                continue
            for angle in range(SPECTRAL_ANGLE_COUNT):
                name = f"spec_p{period}_a{angle}_w{window}"
                channels.append(Channel(name, window, "spectral", period * 10 + angle))
    return channels


def _box_tuple(box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
    return x0, y0, x1, y1


def _center_count(x0: int, y0: int, x1: int, y1: int, window: int, step: int) -> int:
    half = window // 2
    y_count = _axis_count(y0 + half, y1 - half, step)
    x_count = _axis_count(x0 + half, x1 - half, step)
    return y_count * x_count


def _axis_count(first: int, last: int, step: int) -> int:
    if last < first:
        return 0
    return (last - first) // step + 1


def _axis_range(first: int, last: int, step: int) -> np.ndarray:
    return np.arange(first, last + 1, step, dtype=np.int32)


def _fallback_center(y0: int, y1: int, x0: int, x1: int) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array([(y0 + y1) // 2], dtype=np.int32),
        np.array([(x0 + x1) // 2], dtype=np.int32),
    )


def _require_rgb(rgb: np.ndarray) -> None:
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("rgb must be uint8 with shape (H, W, 3)")


def _fill_appearance(rgb, grid, points, axes_y, axes_x, groups, table) -> None:
    luma = _luminance(rgb)
    chroma = _chroma(rgb)
    _paint_mean_std(luma, grid, points, axes_y, axes_x, groups, table, "ink_mean", "ink_std", True)
    _paint_mean_std(
        chroma, grid, points, axes_y, axes_x, groups, table, "chroma_mean", "chroma_std", False
    )
    for plane in range(3):
        _paint_means(
            rgb[:, :, plane],
            grid,
            points,
            axes_y,
            axes_x,
            groups,
            table,
            "color",
            plane,
            COLOR_WINDOWS,
            255.0,
        )


def _fill_gradients(rgb, grid, points, axes_y, axes_x, groups, table) -> None:
    luma = _luminance(rgb)
    grad_x = cv2.Sobel(luma, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(luma, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    _paint_means(magnitude, grid, points, axes_y, axes_x, groups, table, "grad_energy", 0, WINDOWS, 1.0)
    _paint_hog(magnitude, grad_x, grad_y, grid, points, axes_y, axes_x, groups, table)
    _paint_structure(grad_x, grad_y, grid, points, axes_y, axes_x, groups, table)


def _fill_spectral(rgb, grid, points, axes_y, axes_x, groups, table) -> None:
    luma = _luminance(rgb)
    level = _integral(luma)
    std_at = {window: column_index(table, "ink_std", window) for window in SPECTRAL_WINDOWS}
    for period in SPECTRAL_PERIODS:
        for angle in range(SPECTRAL_ANGLE_COUNT):
            _paint_frequency(
                luma, level, period, angle, std_at, grid, points, axes_y, axes_x, groups, table
            )


def _luminance(rgb: np.ndarray) -> np.ndarray:
    luma = rgb[:, :, 0].astype(np.float32)
    luma *= np.float32(0.2126 / 255.0)
    green = rgb[:, :, 1].astype(np.float32)
    green *= np.float32(0.7152 / 255.0)
    luma += green
    blue = rgb[:, :, 2].astype(np.float32)
    blue *= np.float32(0.0722 / 255.0)
    luma += blue
    return luma


def _chroma(rgb: np.ndarray) -> np.ndarray:
    spread = rgb.max(axis=2).astype(np.float32)
    spread -= rgb.min(axis=2)
    spread *= np.float32(1.0 / 255.0)
    return spread


def _paint_mean_std(image, grid, points, axes_y, axes_x, groups, table, mean_kind, std_kind, invert):
    total, total_sq = _integral_pair(image)
    for window in WINDOWS:
        mean_i = column_index(table, mean_kind, window)
        std_i = column_index(table, std_kind, window)
        _paint_moments(grid, mean_i, std_i, total, total_sq, axes_y, axes_x, window, invert, True)
        for block, (ys, xs) in zip(points, groups):
            _paint_moments(block, mean_i, std_i, total, total_sq, ys, xs, window, invert, False)


def _paint_means(image, grid, points, axes_y, axes_x, groups, table, kind, arg, windows, divisor):
    plane = image.astype(np.float32) if image.dtype != np.float32 else image
    if divisor != 1.0:
        plane = plane * np.float32(1.0 / divisor)
    total = _integral(plane)
    for window in windows:
        index = column_index(table, kind, window, arg)
        _paint_mean(grid, index, total, axes_y, axes_x, window, True)
        for block, (ys, xs) in zip(points, groups):
            _paint_mean(block, index, total, ys, xs, window, False)


def _paint_hog(magnitude, grad_x, grad_y, grid, points, axes_y, axes_x, groups, table) -> None:
    angle = np.mod(cv2.phase(grad_x, grad_y, angleInDegrees=False), np.pi)
    bin_id = np.floor(angle * (HOG_BINS / np.pi)).astype(np.int32)
    np.minimum(bin_id, HOG_BINS - 1, out=bin_id)
    for bin_index in range(HOG_BINS):
        response = magnitude * np.float32(1.0) * (bin_id == bin_index)
        _paint_means(
            response, grid, points, axes_y, axes_x, groups, table, "hog", bin_index, HOG_WINDOWS, 1.0
        )


def _paint_structure(grad_x, grad_y, grid, points, axes_y, axes_x, groups, table) -> None:
    xx_i = _integral(grad_x * grad_x)
    yy_i = _integral(grad_y * grad_y)
    xy_i = _integral(grad_x * grad_y)
    for window in STRUCTURE_WINDOWS:
        _paint_tensor(grid, table, xx_i, yy_i, xy_i, axes_y, axes_x, window, True)
        for block, (ys, xs) in zip(points, groups):
            _paint_tensor(block, table, xx_i, yy_i, xy_i, ys, xs, window, False)


def _paint_frequency(luma, level_i, period, angle, std_at, grid, points, axes_y, axes_x, groups, table) -> None:
    theta = angle * np.pi / SPECTRAL_ANGLE_COUNT
    omega_x = 2.0 * np.pi * np.cos(theta) / period
    omega_y = 2.0 * np.pi * np.sin(theta) / period
    cos_i, sin_i = _modulated_integrals(luma, omega_x, omega_y)
    for window in SPECTRAL_WINDOWS:
        if period > window:
            continue
        index = column_index(table, "spectral", window, period * 10 + angle)
        _paint_amp(grid, index, cos_i, sin_i, level_i, axes_y, axes_x, window, omega_x, omega_y, True)
        _divide_std(grid, index, std_at[window])
        for block, (ys, xs) in zip(points, groups):
            _paint_amp(block, index, cos_i, sin_i, level_i, ys, xs, window, omega_x, omega_y, False)
            _divide_std(block, index, std_at[window])


def _integral(plane: np.ndarray) -> np.ndarray:
    return cv2.integral(np.ascontiguousarray(plane, dtype=np.float32))


def _integral_pair(plane: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.ascontiguousarray(plane, dtype=np.float32)
    squared = values * values
    return cv2.integral(values), cv2.integral(np.ascontiguousarray(squared))


def _window_edges(y_centers, x_centers, window: int, height: int, width: int, paired: bool):
    half = window // 2
    y0 = np.clip(y_centers - half, 0, height).astype(np.int32)
    x0 = np.clip(x_centers - half, 0, width).astype(np.int32)
    y1 = np.clip(y_centers - half + window, 0, height).astype(np.int32)
    x1 = np.clip(x_centers - half + window, 0, width).astype(np.int32)
    if paired:
        area = np.maximum((y1 - y0) * (x1 - x0), 1).astype(np.float64)
        return y0, x0, y1, x1, area
    y_span = np.maximum(y1 - y0, 1).astype(np.float64)
    x_span = np.maximum(x1 - x0, 1).astype(np.float64)
    return y0, x0, y1, x1, y_span[:, None] * x_span[None, :]


def _sum_grid(integral, y0, x0, y1, x1) -> np.ndarray:
    return (
        integral[y1[:, None], x1[None, :]]
        - integral[y0[:, None], x1[None, :]]
        - integral[y1[:, None], x0[None, :]]
        + integral[y0[:, None], x0[None, :]]
    )


def _sum_pairs(integral, y0, x0, y1, x1) -> np.ndarray:
    return integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]


def _sum_window(integral, y_centers, x_centers, window: int, paired: bool):
    height, width = integral.shape[0] - 1, integral.shape[1] - 1
    y0, x0, y1, x1, area = _window_edges(y_centers, x_centers, window, height, width, paired)
    if y_centers.size == 0:
        return np.zeros((0,), dtype=np.float64), area
    sums = _sum_pairs(integral, y0, x0, y1, x1) if paired else _sum_grid(integral, y0, x0, y1, x1)
    return sums, area


def _paint_moments(dest, mean_i, std_i, total, total_sq, y_centers, x_centers, window, invert, grid):
    if dest.shape[0] == 0:
        return
    sums, area = _sum_window(total, y_centers, x_centers, window, not grid)
    sumsq, _area = _sum_window(total_sq, y_centers, x_centers, window, not grid)
    mean = sums / area
    var = np.maximum(sumsq / area - mean * mean, 0.0)
    painted = (1.0 - mean) if invert else mean
    dest[..., mean_i] = painted.astype(np.float32)
    dest[..., std_i] = np.sqrt(var).astype(np.float32)


def _paint_mean(dest, index, total, y_centers, x_centers, window, grid) -> None:
    if dest.shape[0] == 0:
        return
    sums, area = _sum_window(total, y_centers, x_centers, window, not grid)
    dest[..., index] = (sums / area).astype(np.float32)


def _paint_tensor(dest, table, xx_i, yy_i, xy_i, y_centers, x_centers, window, grid) -> None:
    if dest.shape[0] == 0:
        return
    xx, area = _sum_window(xx_i, y_centers, x_centers, window, not grid)
    yy, _ = _sum_window(yy_i, y_centers, x_centers, window, not grid)
    xy, _ = _sum_window(xy_i, y_centers, x_centers, window, not grid)
    xx = xx / area
    yy = yy / area
    xy = xy / area
    trace = xx + yy
    det = xx * yy - xy * xy
    gap = np.sqrt(np.maximum(trace * trace - 4.0 * det, 0.0))
    major = 0.5 * (trace + gap)
    minor = 0.5 * (trace - gap)
    coherence = (major - minor) / (major + minor + 1e-6)
    hypot = np.hypot(xx - yy, 2.0 * xy) + 1e-6
    dest[..., column_index(table, "coherence", window)] = coherence.astype(np.float32)
    dest[..., column_index(table, "cos2", window)] = ((xx - yy) / hypot).astype(np.float32)
    dest[..., column_index(table, "sin2", window)] = ((2.0 * xy) / hypot).astype(np.float32)
    dest[..., column_index(table, "tensor_energy", window)] = trace.astype(np.float32)


def _paint_amp(dest, index, cos_i, sin_i, level_i, y_centers, x_centers, window, omega_x, omega_y, grid):
    if dest.shape[0] == 0:
        return
    paired = not grid
    sum_cos, area = _sum_window(cos_i, y_centers, x_centers, window, paired)
    sum_sin, _ = _sum_window(sin_i, y_centers, x_centers, window, paired)
    sum_level, _ = _sum_window(level_i, y_centers, x_centers, window, paired)
    carrier = _carrier(y_centers, x_centers, window, omega_x, omega_y, paired, cos_i.shape)
    mean = sum_level / area
    real = sum_cos - mean * carrier.real
    imag = sum_sin - mean * carrier.imag
    dest[..., index] = (np.hypot(real, imag) / area).astype(np.float32)


def _divide_std(dest, index, std_index) -> None:
    if dest.shape[0] == 0:
        return
    std = dest[..., std_index].astype(np.float32)
    scaled = dest[..., index] / (std + np.float32(SPECTRAL_STD_EPS))
    dest[..., index] = np.clip(scaled, 0.0, SPECTRAL_CLIP)


def _carrier(y_centers, x_centers, window, omega_x, omega_y, paired, integral_shape):
    height, width = integral_shape[0] - 1, integral_shape[1] - 1
    y0, x0, y1, x1, _area = _window_edges(y_centers, x_centers, window, height, width, paired)
    along_y = _geom_sum(omega_y, y0, y1)
    along_x = _geom_sum(omega_x, x0, x1)
    if paired:
        return along_y * along_x
    return along_y[:, None] * along_x[None, :]


def _geom_sum(omega: float, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    count = (end - start).astype(np.float64)
    origin = start.astype(np.float64)
    if abs(omega) < 1e-10:
        return count + 0j
    step = np.exp(1j * omega)
    return np.exp(1j * omega * origin) * (1.0 - np.exp(1j * omega * count)) / (1.0 - step)


def _modulated_integrals(image: np.ndarray, omega_x: float, omega_y: float):
    height, width = image.shape
    y_axis = np.arange(height, dtype=np.float32)
    x_axis = np.arange(width, dtype=np.float32)
    cos_y = np.cos(np.float32(omega_y) * y_axis)
    sin_y = np.sin(np.float32(omega_y) * y_axis)
    cos_x = np.cos(np.float32(omega_x) * x_axis)
    sin_x = np.sin(np.float32(omega_x) * x_axis)
    cos_plane = _separable_product(image, cos_y, cos_x, sin_y, sin_x, subtract=True)
    sin_plane = _separable_product(image, sin_y, cos_x, cos_y, sin_x, subtract=False)
    return _integral(cos_plane), _integral(sin_plane)


def _separable_product(image, row_a, col_a, row_b, col_b, subtract: bool) -> np.ndarray:
    first = image * row_a[:, None]
    first *= col_a[None, :]
    second = image * row_b[:, None]
    second *= col_b[None, :]
    if subtract:
        first -= second
    else:
        first += second
    return np.ascontiguousarray(first)


def _normalize_hog(dest: np.ndarray, table: tuple[Channel, ...]) -> None:
    if dest.shape[0] == 0:
        return
    for window in HOG_WINDOWS:
        columns = [column_index(table, "hog", window, bin_id) for bin_id in range(HOG_BINS)]
        block = dest[..., columns]
        total = block.sum(axis=-1, keepdims=True)
        dest[..., columns] = block / (total + np.float32(1e-6))
