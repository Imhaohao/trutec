"""Compare an exemplar with tile signatures and paint one score per face.

Small faces receive a single score. Large faces keep one score per tile, so a
texture change inside a leaked face does not paint the foreign pattern. Tiles
with no repeated period inherit the nearest measured tile of the same face,
which turns on the paper between hatch strokes without turning on blank paper
farther away.
"""

from __future__ import annotations

import cv2
import numpy as np

from settings import PERIODIC, PLAIN, SOLID, Settings
from signature import Exemplar, SigGrid


def score_maps(
    exemplar: Exemplar | None,
    grid: SigGrid,
    faces: np.ndarray,
    gray: np.ndarray,
    settings: Settings,
) -> dict[str, np.ndarray]:
    stats = _stats_grid(exemplar, grid)
    ncc = _ncc_grid(exemplar, grid, gray, settings)
    product = _product(stats, ncc)
    areas = np.bincount(faces.ravel())
    tile_faces = _tile_faces(faces, grid)
    return {
        "stats": _paint(stats, tile_faces, areas, settings),
        "ncc": _paint(ncc, tile_faces, areas, settings),
        "stats_times_ncc": _paint(product, tile_faces, areas, settings),
    }


def sample_maps(
    maps: dict[str, np.ndarray],
    grid: SigGrid,
    faces: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
) -> dict[str, np.ndarray]:
    iy, ix = _nearest(y, grid.center_y), _nearest(x, grid.center_x)
    tile_faces = _tile_faces(faces, grid)
    pixel_face = faces[y, x]
    same = pixel_face == tile_faces[iy, ix]
    n_faces = int(faces.max()) + 1
    sampled = {}
    for name, tile_scores in maps.items():
        fallback = _face_fallback(tile_scores, tile_faces, n_faces)
        chosen = np.where(same, tile_scores[iy, ix], fallback[pixel_face])
        sampled[name] = np.clip(chosen.astype(np.float32), 0.0, 1.0)
    return sampled


def areas_from_faces(faces: np.ndarray) -> np.ndarray:
    return np.bincount(faces.ravel())


def _paint(seeds: np.ndarray, tile_faces: np.ndarray, areas: np.ndarray, settings: Settings) -> np.ndarray:
    filled = _inherit(seeds, tile_faces, settings.inherit_tiles)
    return _flatten_small(filled, seeds, tile_faces, areas, settings.large_face_area)


def _stats_grid(exemplar: Exemplar | None, grid: SigGrid) -> np.ndarray:
    scores = np.full(grid.kind.shape, np.nan, dtype=np.float32)
    if exemplar is None:
        return scores
    if exemplar.kind == SOLID:
        return _solid_scores(exemplar, grid, scores)
    if exemplar.kind != PERIODIC:
        return scores
    periodic = grid.kind == PERIODIC
    scores[periodic] = _periodic_similarity(exemplar, grid)[periodic]
    scores[grid.kind == SOLID] = 0.0
    return scores


def _solid_scores(exemplar: Exemplar, grid: SigGrid, scores: np.ndarray) -> np.ndarray:
    solid = grid.kind == SOLID
    scores[solid] = _solid_similarity(exemplar, grid)[solid]
    scores[grid.kind == PERIODIC] = 0.0
    return scores


def _ncc_grid(
    exemplar: Exemplar | None,
    grid: SigGrid,
    gray: np.ndarray,
    settings: Settings,
) -> np.ndarray:
    scores = np.full(grid.kind.shape, np.nan, dtype=np.float32)
    if exemplar is None or exemplar.kind == PLAIN:
        return scores
    if exemplar.kind == SOLID or float(exemplar.template.std()) < 5.0:
        return _stats_grid(exemplar, grid)
    response = _template_response(gray, exemplar.template, settings.ncc_search)
    sampled = _sample_response(response, grid, exemplar.template.shape)
    measured = grid.kind != PLAIN
    scores[measured] = sampled[measured]
    return scores


def _product(stats: np.ndarray, ncc: np.ndarray) -> np.ndarray:
    both = np.isfinite(stats) & np.isfinite(ncc)
    out = np.full(stats.shape, np.nan, dtype=np.float32)
    out[both] = stats[both] * ncc[both]
    return out


def _periodic_similarity(exemplar: Exemplar, grid: SigGrid) -> np.ndarray:
    cost, matched = _matched_cost(exemplar, grid)
    penalty = (
        cost
        + 1.5 * _color_penalty(exemplar, grid)
        + 0.5 * _relative_gap(exemplar.thickness, grid.thickness)
        + 0.4 * np.abs(exemplar.density - grid.density)
    )
    ratio = np.clip(matched / max(exemplar.strength, 1e-3), 0.0, 1.0)
    return np.exp(-penalty) * ratio


def _solid_similarity(exemplar: Exemplar, grid: SigGrid) -> np.ndarray:
    penalty = 2.5 * _color_penalty(exemplar, grid) + 0.8 * np.abs(exemplar.value - grid.value)
    return np.exp(-penalty).astype(np.float32)


def _matched_cost(exemplar: Exemplar, grid: SigGrid) -> tuple[np.ndarray, np.ndarray]:
    primary, used, primary_strength = _closest_peak(exemplar.orient, exemplar.period, grid)
    if exemplar.strength2 <= 0.55 * max(exemplar.strength, 1e-3):
        return primary, primary_strength
    secondary, secondary_strength = _closest_unused(exemplar.orient2, exemplar.period2, grid, used)
    return primary + secondary, np.minimum(primary_strength, secondary_strength)


def _closest_peak(orient: float, period: float, grid: SigGrid) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    costs = _peak_costs(orient, period, grid)
    index = np.argmin(costs, axis=0)
    return np.min(costs, axis=0), index, _strength_at(grid, index)


def _closest_unused(
    orient: float, period: float, grid: SigGrid, used: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    costs = _peak_costs(orient, period, grid)
    for index in range(costs.shape[0]):
        costs[index] = np.where(used == index, 8.0, costs[index])
    choice = np.argmin(costs, axis=0)
    return np.min(costs, axis=0), _strength_at(grid, choice)


def _peak_costs(orient: float, period: float, grid: SigGrid) -> np.ndarray:
    costs = [_gated_peak_cost(orient, period, grid, index) for index in range(grid.peak_s.shape[2])]
    return np.stack(costs, axis=0)


def _gated_peak_cost(orient: float, period: float, grid: SigGrid, index: int) -> np.ndarray:
    cost = _pair_cost(orient, period, grid.peak_o[:, :, index], grid.peak_p[:, :, index])
    return np.where(grid.peak_s[:, :, index] >= 0.12, cost, 8.0)


def _strength_at(grid: SigGrid, index: np.ndarray) -> np.ndarray:
    strength = np.zeros(index.shape, dtype=np.float32)
    for peak in range(grid.peak_s.shape[2]):
        strength = np.where(index == peak, grid.peak_s[:, :, peak], strength)
    return strength


def _pair_cost(orient_a: float, period_a: float, orient_b: np.ndarray, period_b: np.ndarray) -> np.ndarray:
    orient = _angle_delta(orient_a, orient_b) / 30.0
    period = _relative_gap(period_a, period_b) / 0.35
    return orient * orient + period * period


def _color_penalty(exemplar: Exemplar, grid: SigGrid) -> np.ndarray:
    both_gray = (exemplar.sat < 0.12) & (grid.sat < 0.12)
    colored = 0.75 * (_hue_delta(exemplar.hue, grid.hue) / 90.0) + 0.25 * np.abs(exemplar.sat - grid.sat)
    gray = 0.4 * np.abs(exemplar.value - grid.value)
    return np.where(both_gray, gray, colored)


def _inherit(seeds: np.ndarray, faces: np.ndarray, radius: int) -> np.ndarray:
    out = np.zeros(seeds.shape, dtype=np.float32)
    best = np.full(seeds.shape, np.inf, dtype=np.float32)
    known = np.isfinite(seeds)
    out[known] = seeds[known]
    best[known] = 0.0
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            dist = max(abs(dy), abs(dx))
            if dist:
                _shift_fill(seeds, faces, out, best, dy, dx, float(dist))
    return out


def _shift_fill(
    seeds: np.ndarray,
    faces: np.ndarray,
    out: np.ndarray,
    best: np.ndarray,
    dy: int,
    dx: int,
    dist: float,
) -> None:
    src, dst = _shift_slices(dy, dx, seeds.shape)
    source = seeds[src]
    take = np.isfinite(source) & (faces[src] == faces[dst]) & (dist < best[dst])
    best[dst][take] = dist
    out[dst][take] = source[take]


def _flatten_small(
    filled: np.ndarray,
    seeds: np.ndarray,
    faces: np.ndarray,
    areas: np.ndarray,
    large_area: int,
) -> np.ndarray:
    out = filled.copy()
    for face_id in np.unique(faces).tolist():
        _flatten_one(out, seeds, faces, areas, int(face_id), large_area)
    return out


def _flatten_one(
    out: np.ndarray,
    seeds: np.ndarray,
    faces: np.ndarray,
    areas: np.ndarray,
    face_id: int,
    large_area: int,
) -> None:
    area = int(areas[face_id]) if 0 <= face_id < len(areas) else 0
    if area >= large_area:
        return
    mask = faces == face_id
    values = seeds[mask]
    values = values[np.isfinite(values)]
    out[mask] = float(np.median(values)) if values.size else 0.0


def _template_response(gray: np.ndarray, template: np.ndarray, search: int) -> np.ndarray:
    if template.shape[0] > gray.shape[0] or template.shape[1] > gray.shape[1]:
        return np.zeros((1, 1), dtype=np.float32)
    response = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    response = np.nan_to_num(response, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    kernel = np.ones((search, search), dtype=np.uint8)
    return cv2.dilate(np.clip(response, 0.0, 1.0), kernel)


def _sample_response(response: np.ndarray, grid: SigGrid, template_shape: tuple[int, int]) -> np.ndarray:
    top_y = np.clip(np.rint(grid.center_y - template_shape[0] / 2.0).astype(np.int32), 0, response.shape[0] - 1)
    top_x = np.clip(np.rint(grid.center_x - template_shape[1] / 2.0).astype(np.int32), 0, response.shape[1] - 1)
    return response[np.ix_(top_y, top_x)]


def _tile_faces(faces: np.ndarray, grid: SigGrid) -> np.ndarray:
    ys = np.clip(np.rint(grid.center_y).astype(np.int32), 0, faces.shape[0] - 1)
    xs = np.clip(np.rint(grid.center_x).astype(np.int32), 0, faces.shape[1] - 1)
    return faces[np.ix_(ys, xs)]


def _face_fallback(tile_scores: np.ndarray, tile_faces: np.ndarray, n_faces: int) -> np.ndarray:
    fallback = np.zeros(max(n_faces, 1), dtype=np.float32)
    if tile_faces.size:
        np.maximum.at(fallback, tile_faces.ravel(), tile_scores.ravel())
    return fallback


def _nearest(coord: np.ndarray, centers: np.ndarray) -> np.ndarray:
    out = np.empty(coord.shape[0], dtype=np.int32)
    step = 250_000
    for start in range(0, coord.shape[0], step):
        part = coord[start:start + step]
        out[start:start + part.size] = np.abs(part[:, None] - centers[None, :]).argmin(axis=1)
    return out


def _shift_slices(
    dy: int, dx: int, shape: tuple[int, int]
) -> tuple[tuple[slice, slice], tuple[slice, slice]]:
    height, width = shape
    y_src = slice(max(0, dy), min(height, height + dy))
    y_dst = slice(max(0, -dy), min(height, height - dy))
    x_src = slice(max(0, dx), min(width, width + dx))
    x_dst = slice(max(0, -dx), min(width, width - dx))
    return (y_src, x_src), (y_dst, x_dst)


def _relative_gap(left: float | np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.abs(left - right) / np.maximum(np.maximum(left, right), 1e-3)


def _angle_delta(left: float, right: np.ndarray) -> np.ndarray:
    delta = np.abs(left - right) % 180.0
    return np.minimum(delta, 180.0 - delta)


def _hue_delta(left: float, right: np.ndarray) -> np.ndarray:
    delta = np.abs(left - right) % 180.0
    return np.minimum(delta, 180.0 - delta)
