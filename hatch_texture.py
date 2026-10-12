"""Label-free hatch baselines: query-crop correlation and local texture descriptors.

This module does not read manifests, annotations, or labels. Template choices
come only from the query crop. Descriptor channels stay in the order returned
by texture_feature_names for every image.
"""
import math
import warnings
from typing import NamedTuple

import cv2
import numpy as np

_INK_WINDOWS = (16, 32, 64)
_GRADIENT_WINDOW = 16
_GABOR_WAVELENGTHS = (4, 8, 16)
_GABOR_ANGLES = (0, 30, 60, 90, 120, 150)
_GABOR_POOLS = (16, 32)
_MAX_TEMPLATES = 5
_KERNEL_RADIUS_LIMIT = 15


class _Window(NamedTuple):
    score: float
    y: int
    x: int
    side: int


def normalized_template_response(
    gray: np.ndarray,
    query_box: tuple[int, int, int, int],
    max_template_side: int = 32,
) -> np.ndarray:
    """Strongest normalized cross-correlation of a few query subcrops, as float32 HxW in [0, 1].

    Scores below 0 are stored as 0. Each hit is written at the window center and
    dilated by half the template size so a periodic hatch fills its region.
    """
    _require_uint8_gray(gray)
    box = _require_box(query_box, int(gray.shape[0]), int(gray.shape[1]))
    side_limit = _require_positive_int(max_template_side, "max_template_side")
    response = np.zeros(gray.shape, dtype=np.float32)
    query = gray[box[1]:box[3], box[0]:box[2]]
    for template in _templates_from_query(query, side_limit):
        _paint_template(response, gray, template)
    np.clip(response, 0.0, 1.0, out=response)
    return response


def template_response_to_mask(response: np.ndarray, threshold: float) -> np.ndarray:
    """Threshold a template response into a boolean mask. threshold is in [0, 1]."""
    _require_response(response)
    limit = _require_unit_float(threshold, "threshold")
    return response >= limit


def texture_feature_names() -> list[str]:
    """Fixed descriptor order.

    Ink density is the mean of ``1 - gray/255`` and ink variance is the variance
    of ``gray/255``, each inside native windows 16, 32, and 64. Gradient names are
    mean squared derivatives over a native 16-pixel window: horizontal is Sobel x,
    vertical is Sobel y, diagonal is the x+y direction, and antidiagonal is x-y.
    Gabor names ``gabor_o{angle}_l{wavelength}_p{pool}`` use angles 0, 30, 60, 90,
    120, 150, native wavelengths 4, 8, 16, and pools 16 then 32. Angle 0 oscillates
    along x (vertical bars). Angle 90 oscillates along y (horizontal bars).
    """
    names: list[str] = []
    for window in _INK_WINDOWS:
        names.append(f"ink_density_w{window}")
        names.append(f"ink_variance_w{window}")
    names.extend((
        "grad_energy_horizontal",
        "grad_energy_vertical",
        "grad_energy_diagonal",
        "grad_energy_antidiagonal",
    ))
    for wavelength in _GABOR_WAVELENGTHS:
        for angle in _GABOR_ANGLES:
            for pool in _GABOR_POOLS:
                names.append(f"gabor_o{angle}_l{wavelength}_p{pool}")
    return names


def texture_features(gray: np.ndarray, stride: int = 4) -> np.ndarray:
    """Local phase-invariant descriptors on the stride grid, float32 [gh, gw, C].

    Filtering runs on a half-resolution image. Cell ``(iy, ix)`` is the native
    center ``(min(iy*stride+stride//2, H-1), min(ix*stride+stride//2, W-1))``.
    """
    height, width = _require_uint8_gray(gray)
    step = _require_positive_int(stride, "stride")
    norm = _half_luminance(gray)
    rows = _cell_axis(height, step)
    cols = _cell_axis(width, step)
    features = np.zeros((int(rows.size), int(cols.size), len(texture_feature_names())), dtype=np.float32)
    _fill_ink_channels(features, norm, rows, cols)
    _fill_gradient_channels(features, norm, rows, cols)
    _fill_gabor_channels(features, norm, rows, cols)
    return features


def query_descriptor(
    features: np.ndarray,
    query_box: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    stride: int = 4,
) -> np.ndarray:
    """Median descriptor of feature cells whose native centers lie inside query_box.

    ``image_shape`` is ``(height, width)``. A query that contains no cell center
    uses the nearest center. Distance ties resolve to the earlier row, then the
    earlier column. The result has shape ``(C,)`` and dtype float32.
    """
    step = _require_positive_int(stride, "stride")
    height, width = _require_image_shape(image_shape)
    rows, cols = _require_feature_grid(features, height, width, step)
    box = _require_box(query_box, height, width)
    return _descriptor_from_centers(features, rows, cols, box)


def descriptor_for_query(
    gray: np.ndarray,
    features: np.ndarray,
    query_box: tuple[int, int, int, int],
    stride: int = 4,
    source: str = "drawing",
) -> np.ndarray:
    """Query descriptor from drawing-grid cells or from the exact query crop.

    ``drawing`` is the legacy median of ``features`` and stays the default.
    ``crop`` runs ``texture_features`` on the exact uint8 crop, with no resampling,
    then takes the median of every crop feature cell.
    """
    choice = _require_query_source(source)
    if choice == "drawing":
        return _drawing_query_descriptor(gray, features, query_box, stride)
    return _crop_query_descriptor(gray, query_box, stride)


def descriptor_source_from_metadata(metadata: object) -> str:
    """Return the stored query descriptor source. A missing key means drawing."""
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be a dict")
    if "query_descriptor_source" not in metadata:
        return "drawing"
    return _require_query_source(metadata["query_descriptor_source"])


def _require_query_source(source: object) -> str:
    if source == "drawing" or source == "crop":
        return source
    raise ValueError("source must be 'drawing' or 'crop'")


def _drawing_query_descriptor(
    gray: np.ndarray,
    features: np.ndarray,
    query_box: tuple[int, int, int, int],
    stride: int,
) -> np.ndarray:
    height, width = _require_uint8_gray(gray)
    return query_descriptor(features, query_box, (height, width), stride)


def _crop_query_descriptor(
    gray: np.ndarray,
    query_box: tuple[int, int, int, int],
    stride: int,
) -> np.ndarray:
    height, width = _require_uint8_gray(gray)
    box = _require_box(query_box, height, width)
    step = _require_positive_int(stride, "stride")
    cells = texture_features(_exact_crop(gray, box), step)
    return _median_feature_cells(cells)


def _exact_crop(gray: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    crop = gray[y0:y1, x0:x1]
    _require_uint8_gray(crop)
    return np.ascontiguousarray(crop)


def _median_feature_cells(features: np.ndarray) -> np.ndarray:
    flat = features.reshape(-1, int(features.shape[-1]))
    median = np.median(flat, axis=0)
    return np.asarray(median, dtype=np.float32)


def _templates_from_query(query: np.ndarray, max_side: int) -> list[np.ndarray]:
    for side in _candidate_sides(int(query.shape[0]), int(query.shape[1]), max_side):
        chosen = _choose_side(query, side)
        if chosen:
            return chosen
    return []


def _candidate_sides(query_h: int, query_w: int, max_side: int) -> list[int]:
    limit = min(query_h, query_w, max_side)
    if limit < 2:
        return []
    sides = [limit]
    half = max(4, limit // 2)
    if limit >= 8 and half < limit:
        sides.append(half)
    third = max(4, limit // 3)
    if limit >= 16 and third < half:
        sides.append(third)
    return sides


def _choose_side(query: np.ndarray, side: int) -> list[np.ndarray]:
    found = _rank_windows(query, side)
    chosen: list[_Window] = []
    for window in found:
        if _far_enough(window, chosen):
            chosen.append(window)
        if len(chosen) == _MAX_TEMPLATES:
            break
    return [_crop_window(query, window) for window in chosen]


def _rank_windows(query: np.ndarray, side: int) -> list[_Window]:
    found: list[_Window] = []
    for y in _origins(int(query.shape[0]), side):
        _append_row(query, side, y, found)
    found.sort(key=_window_order)
    return found


def _append_row(query: np.ndarray, side: int, y: int, found: list[_Window]) -> None:
    for x in _origins(int(query.shape[1]), side):
        crop = query[y:y + side, x:x + side]
        if _usable_template(crop):
            found.append(_Window(_template_rank(crop), y, x, side))


def _window_order(window: _Window) -> tuple[float, int, int]:
    return (-window.score, window.y, window.x)


def _origins(span: int, window: int) -> list[int]:
    if span <= window:
        return [0]
    last = span - window
    count = _origin_count(last, window)
    raw = [int(round(i * last / (count - 1))) for i in range(count)]
    return sorted(set(raw))


def _origin_count(last: int, window: int) -> int:
    if last < max(1, window // 2):
        return 2
    if last >= window:
        return _MAX_TEMPLATES
    return 3


def _usable_template(crop: np.ndarray) -> bool:
    values = crop.astype(np.float32, copy=False)
    deviation = float(values.std())
    if deviation < 8.0:
        return False
    if float(values.mean()) > 248.0 and deviation < 20.0:
        return False
    return True


def _template_rank(crop: np.ndarray) -> float:
    return float(crop.astype(np.float32, copy=False).std())


def _far_enough(window: _Window, chosen: list[_Window]) -> bool:
    for other in chosen:
        if _close_pair(window, other):
            return False
    return True


def _close_pair(window: _Window, other: _Window) -> bool:
    gap_y = abs((window.y + window.side / 2.0) - (other.y + other.side / 2.0))
    gap_x = abs((window.x + window.side / 2.0) - (other.x + other.side / 2.0))
    return gap_y < window.side / 4.0 and gap_x < window.side / 4.0


def _crop_window(query: np.ndarray, window: _Window) -> np.ndarray:
    crop = query[window.y:window.y + window.side, window.x:window.x + window.side]
    return np.ascontiguousarray(crop)


def _paint_template(response: np.ndarray, image: np.ndarray, template: np.ndarray) -> None:
    scores = _normalized_scores(image, template)
    centers = _scores_at_centers(image.shape, template.shape, scores)
    filled = _dilate_half_template(centers, int(template.shape[0]), int(template.shape[1]))
    np.maximum(response, filled, out=response)


def _normalized_scores(image: np.ndarray, template: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        raw = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    finite = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    return np.asarray(np.clip(finite, 0.0, 1.0), dtype=np.float32)


def _scores_at_centers(
    image_shape: tuple[int, ...],
    template_shape: tuple[int, ...],
    scores: np.ndarray,
) -> np.ndarray:
    """Map each match, stored at the template's top-left, onto that window's center."""
    height, width = int(image_shape[0]), int(image_shape[1])
    template_h, template_w = int(template_shape[0]), int(template_shape[1])
    center = np.zeros((height, width), dtype=np.float32)
    y0 = template_h // 2
    x0 = template_w // 2
    center[y0:y0 + scores.shape[0], x0:x0 + scores.shape[1]] = scores
    return center


def _dilate_half_template(center: np.ndarray, template_h: int, template_w: int) -> np.ndarray:
    radius_y = template_h // 2
    radius_x = template_w // 2
    kernel = np.ones((2 * radius_y + 1, 2 * radius_x + 1), dtype=np.uint8)
    dilated = cv2.dilate(center, kernel)
    return np.asarray(dilated, dtype=np.float32)


def _half_luminance(gray: np.ndarray) -> np.ndarray:
    """Average 2x2 blocks. An odd edge is repeated so native ``x // 2`` stays in range."""
    reduced = cv2.resize(_pad_even(gray), _half_size(gray), interpolation=cv2.INTER_AREA)
    return np.asarray(reduced, dtype=np.float32) * np.float32(1.0 / 255.0)


def _half_size(gray: np.ndarray) -> tuple[int, int]:
    height, width = int(gray.shape[0]), int(gray.shape[1])
    return (max(1, (width + 1) // 2), max(1, (height + 1) // 2))


def _pad_even(gray: np.ndarray) -> np.ndarray:
    vertical = gray
    if gray.shape[0] % 2 == 1:
        vertical = np.concatenate([gray, gray[-1:]], axis=0)
    if vertical.shape[1] % 2 == 1:
        return np.concatenate([vertical, vertical[:, -1:]], axis=1)
    return vertical


def _cell_axis(length: int, stride: int) -> np.ndarray:
    count = math.ceil(length / stride)
    index = np.arange(count, dtype=np.int32)
    centers = index * np.int32(stride) + np.int32(stride // 2)
    return np.minimum(centers, np.int32(length - 1))


def _fill_ink_channels(
    features: np.ndarray,
    norm: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
) -> None:
    names = texture_feature_names()
    for window in _INK_WINDOWS:
        density = _sample_map(_box_mean(np.float32(1.0) - norm, window), rows, cols)
        features[:, :, names.index(f"ink_density_w{window}")] = density
        variance = _sample_map(_window_variance(norm, window), rows, cols)
        features[:, :, names.index(f"ink_variance_w{window}")] = variance


def _fill_gradient_channels(
    features: np.ndarray,
    norm: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
) -> None:
    dx, dy = _sobel_pair(norm)
    names = texture_feature_names()
    _store_gradient(features, names, "grad_energy_horizontal", dx * dx, rows, cols)
    _store_gradient(features, names, "grad_energy_vertical", dy * dy, rows, cols)
    diagonal = (dx + dy) * (dx + dy) * np.float32(0.5)
    _store_gradient(features, names, "grad_energy_diagonal", diagonal, rows, cols)
    antidiagonal = (dx - dy) * (dx - dy) * np.float32(0.5)
    _store_gradient(features, names, "grad_energy_antidiagonal", antidiagonal, rows, cols)


def _store_gradient(
    features: np.ndarray,
    names: list[str],
    name: str,
    energy: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
) -> None:
    features[:, :, names.index(name)] = _sample_map(_box_mean(energy, _GRADIENT_WINDOW), rows, cols)


def _sobel_pair(norm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    smooth = cv2.GaussianBlur(norm, (3, 3), 0, borderType=cv2.BORDER_REFLECT_101)
    dx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT_101)
    dy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT_101)
    return np.asarray(dx, dtype=np.float32), np.asarray(dy, dtype=np.float32)


def _fill_gabor_channels(
    features: np.ndarray,
    norm: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
) -> None:
    names = texture_feature_names()
    for wavelength in _GABOR_WAVELENGTHS:
        _fill_wavelength(features, norm, rows, cols, names, wavelength)


def _fill_wavelength(
    features: np.ndarray,
    norm: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    names: list[str],
    wavelength: int,
) -> None:
    for angle in _GABOR_ANGLES:
        energy = _quadrature_energy(norm, wavelength, angle)
        _store_gabor_pools(features, names, energy, rows, cols, wavelength, angle)


def _store_gabor_pools(
    features: np.ndarray,
    names: list[str],
    energy: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    wavelength: int,
    angle: int,
) -> None:
    for pool in _GABOR_POOLS:
        sampled = _sample_map(_box_mean(energy, pool), rows, cols)
        features[:, :, names.index(f"gabor_o{angle}_l{wavelength}_p{pool}")] = sampled


def _quadrature_energy(image: np.ndarray, wavelength: int, angle: int) -> np.ndarray:
    real_kernel, imag_kernel = _carrier_kernels(wavelength, angle)
    real = cv2.filter2D(image, cv2.CV_32F, real_kernel, borderType=cv2.BORDER_REFLECT_101)
    imag = cv2.filter2D(image, cv2.CV_32F, imag_kernel, borderType=cv2.BORDER_REFLECT_101)
    real_f = np.asarray(real, dtype=np.float32)
    imag_f = np.asarray(imag, dtype=np.float32)
    return np.sqrt(real_f * real_f + imag_f * imag_f)


def _carrier_kernels(native_wavelength: int, angle_deg: int) -> tuple[np.ndarray, np.ndarray]:
    """Quadrature pair on the half-resolution image. Native wavelength 8 is 4 pixels there."""
    wavelength = max(1.0, float(native_wavelength) / 2.0)
    sigma = max(0.8, 0.56 * wavelength)
    radius = int(min(_KERNEL_RADIUS_LIMIT, max(1, round(3.0 * sigma))))
    axis = np.arange(-radius, radius + 1, dtype=np.float32)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    theta = float(np.deg2rad(angle_deg))
    x_prime = xx * np.float32(np.cos(theta)) + yy * np.float32(np.sin(theta))
    y_prime = -xx * np.float32(np.sin(theta)) + yy * np.float32(np.cos(theta))
    gamma2 = np.float32(0.25)
    denom = np.float32(2.0 * sigma * sigma)
    envelope = np.exp(-(x_prime * x_prime + gamma2 * y_prime * y_prime) / denom)
    phase = np.float32(2.0 * np.pi / wavelength) * x_prime
    return _normalize_pair(envelope * np.cos(phase), envelope * np.sin(phase))


def _normalize_pair(real: np.ndarray, imag: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    real_zero = real - real.mean()
    imag_zero = imag - imag.mean()
    scale = np.sqrt(np.sum(real_zero * real_zero) + np.sum(imag_zero * imag_zero)) + np.float32(1e-6)
    return (real_zero / scale).astype(np.float32), (imag_zero / scale).astype(np.float32)


def _box_mean(image: np.ndarray, native_window: int) -> np.ndarray:
    ksize = max(1, int(native_window) // 2)
    pooled = cv2.boxFilter(
        image,
        cv2.CV_32F,
        (ksize, ksize),
        normalize=True,
        borderType=cv2.BORDER_REFLECT_101,
    )
    return np.asarray(pooled, dtype=np.float32)


def _window_variance(norm: np.ndarray, native_window: int) -> np.ndarray:
    mean = _box_mean(norm, native_window)
    mean_sq = _box_mean(norm * norm, native_window)
    return np.maximum(mean_sq - mean * mean, np.float32(0.0))


def _sample_map(feature_map: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    half_h, half_w = int(feature_map.shape[0]), int(feature_map.shape[1])
    ys = np.minimum(rows.astype(np.int32) // np.int32(2), np.int32(half_h - 1))
    xs = np.minimum(cols.astype(np.int32) // np.int32(2), np.int32(half_w - 1))
    return np.asarray(feature_map[np.ix_(ys, xs)], dtype=np.float32)


def _descriptor_from_centers(
    features: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    box: tuple[int, int, int, int],
) -> np.ndarray:
    x0, y0, x1, y1 = box
    row_hit = (rows >= y0) & (rows < y1)
    col_hit = (cols >= x0) & (cols < x1)
    if bool(np.any(row_hit)) and bool(np.any(col_hit)):
        return _median_block(features, row_hit, col_hit)
    return _nearest_center(features, rows, cols, box)


def _median_block(features: np.ndarray, row_hit: np.ndarray, col_hit: np.ndarray) -> np.ndarray:
    block = features[np.ix_(row_hit, col_hit)]
    flat = block.reshape(-1, block.shape[-1])
    return np.median(flat, axis=0).astype(np.float32, copy=False)


def _nearest_center(
    features: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    box: tuple[int, int, int, int],
) -> np.ndarray:
    x0, y0, x1, y1 = box
    cy = (y0 + y1 - 1) / 2.0
    cx = (x0 + x1 - 1) / 2.0
    iy, ix = _nearest_index(rows, cols, cy, cx)
    return features[iy, ix].astype(np.float32, copy=False)


def _nearest_index(rows: np.ndarray, cols: np.ndarray, cy: float, cx: float) -> tuple[int, int]:
    distance = _center_distance(rows, cols, cy, cx)
    iy_grid = np.broadcast_to(np.arange(rows.size)[:, None], distance.shape)
    ix_grid = np.broadcast_to(np.arange(cols.size)[None, :], distance.shape)
    order = np.lexsort((ix_grid.ravel(), iy_grid.ravel(), distance.ravel()))
    iy, ix = divmod(int(order[0]), int(cols.size))
    return iy, ix


def _center_distance(rows: np.ndarray, cols: np.ndarray, cy: float, cx: float) -> np.ndarray:
    dy = rows.astype(np.float64) - cy
    dx = cols.astype(np.float64) - cx
    return dy[:, None] ** 2 + dx[None, :] ** 2


def _require_feature_grid(
    features: np.ndarray,
    height: int,
    width: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    _require_feature_array(features)
    grid_h = math.ceil(height / stride)
    grid_w = math.ceil(width / stride)
    if features.shape[0] != grid_h or features.shape[1] != grid_w:
        raise ValueError("feature grid does not match image_shape and stride")
    return _cell_axis(height, stride), _cell_axis(width, stride)


def _require_feature_array(features: np.ndarray) -> None:
    if not isinstance(features, np.ndarray):
        raise TypeError("features must be a numpy array")
    if features.ndim != 3 or features.shape[2] < 1:
        raise ValueError("features must have shape grid_h x grid_w x channels")
    if not np.issubdtype(features.dtype, np.floating):
        raise TypeError("features must be floating point")
    if not bool(np.isfinite(features).all()):
        raise ValueError("features must be finite")


def _require_image_shape(image_shape: object) -> tuple[int, int]:
    if not isinstance(image_shape, tuple) or len(image_shape) != 2:
        raise ValueError("image_shape must be (height, width)")
    height = _require_positive_int(image_shape[0], "height")
    width = _require_positive_int(image_shape[1], "width")
    return height, width


def _require_uint8_gray(gray: np.ndarray) -> tuple[int, int]:
    if not isinstance(gray, np.ndarray):
        raise TypeError("gray must be a numpy array")
    if gray.dtype != np.uint8 or gray.ndim != 2 or gray.size == 0:
        raise ValueError("gray must be a non-empty uint8 array with shape HxW")
    return int(gray.shape[0]), int(gray.shape[1])


def _require_box(query_box: object, height: int, width: int) -> tuple[int, int, int, int]:
    coords = _box_ints(query_box)
    _require_bounds(coords, height, width)
    return coords


def _box_ints(query_box: object) -> tuple[int, int, int, int]:
    if not isinstance(query_box, tuple) or len(query_box) != 4:
        raise ValueError("query_box must be a tuple of four integers")
    values = tuple(_require_int(item) for item in query_box)
    return values[0], values[1], values[2], values[3]


def _require_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError("coordinates must be integers")
    return int(value)


def _require_bounds(box: tuple[int, int, int, int], height: int, width: int) -> None:
    x0, y0, x1, y1 = box
    if x0 < 0 or y0 < 0 or x1 > width or y1 > height or x0 >= x1 or y0 >= y1:
        raise ValueError("query_box is outside the image or empty")


def _require_positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    number = int(value)
    if number < 1:
        raise ValueError(f"{name} must be positive")
    return number


def _require_response(response: np.ndarray) -> None:
    if not isinstance(response, np.ndarray):
        raise TypeError("response must be a numpy array")
    if response.ndim != 2 or response.size == 0:
        raise ValueError("response must have shape HxW")
    if not np.issubdtype(response.dtype, np.floating):
        raise TypeError("response must be floating point")
    if not bool(np.isfinite(response).all()):
        raise ValueError("response must be finite")


def _require_unit_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int, np.floating, np.integer)):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number) or number < 0.0 or number > 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return number


def main() -> None:
    for name in texture_feature_names():
        print(name)


if __name__ == "__main__":
    main()
