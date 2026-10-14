"""Guarded query-crop cleaners. Label-free. Rule d comes from querycrop.cleaners."""
from __future__ import annotations

import bootstrap  # noqa: F401

from typing import NamedTuple

import cv2
import numpy as np

import hatch_texture
from cleaners import (
    _is_long_line,
    _is_text_glyph,
    clean_crop_components,
    clean_descriptor_iqr,
)

PAPER_LEVEL = 210
INK_REMOVE_LIMIT = 0.5
ORIENTATION_DEGREES = 15.0
FAMILY_COUNT = 3
SPACING_CV_LIMIT = 0.35
SPAN_FRACTION = 0.5
EMPTY_INK = 10
DESCRIPTOR_STRIDE = 4
VARIANTS = ("v6", "rule_d", "g1", "g2", "g3")
GUARDED = ("g1", "g2", "g3")
CACHE_VERSION = 1


class Part(NamedTuple):
    label: int
    cx: float
    cy: float
    angle: float
    long_line: bool
    glyph: bool
    spanning: bool


def clean_crop_g1(crop: np.ndarray) -> np.ndarray:
    """Rule d's component pass, reverted when it deletes more than half the ink."""
    cleaned = clean_crop_components(crop, paper_level=PAPER_LEVEL)
    return _guard_ink(crop, cleaned)


def clean_crop_g2(crop: np.ndarray) -> np.ndarray:
    """Drop isolated long lines and glyphs. Keep spanning parts and regular line families."""
    ink = np.asarray(_ink(crop), dtype=np.uint8)
    total = int(np.count_nonzero(ink))
    if total < EMPTY_INK:
        return crop.copy()
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(ink, connectivity=8)
    if count <= 1:
        return crop.copy()
    parts = _parts(crop.shape, labels, stats, centroids, count)
    cleaned = _erase(crop, ink, labels, parts, _protected(parts))
    return _restore_if_empty(crop, cleaned, total)


def clean_crop_g3(crop: np.ndarray) -> np.ndarray:
    """G2, reverted when that pass deletes more than half the ink."""
    return _guard_ink(crop, clean_crop_g2(crop))


def crop_for(name: str, crop: np.ndarray) -> np.ndarray:
    if name == "v6":
        return crop
    if name == "rule_d":
        return clean_crop_components(crop, paper_level=PAPER_LEVEL)
    if name == "g1":
        return clean_crop_g1(crop)
    if name == "g2":
        return clean_crop_g2(crop)
    if name == "g3":
        return clean_crop_g3(crop)
    raise KeyError(name)


def descriptor_for(name: str, gray: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Stage-1 crop descriptor. v6 is the uncleaned median. Every cleaner then uses IQR."""
    if name == "v6":
        return hatch_texture.descriptor_for_query(gray, gray, box, stride=DESCRIPTOR_STRIDE, source="crop")
    x0, y0, x1, y1 = (int(value) for value in box)
    cleaned = crop_for(name, np.ascontiguousarray(gray[y0:y1, x0:x1]))
    return clean_descriptor_iqr(cleaned, stride=DESCRIPTOR_STRIDE)


def _ink(crop: np.ndarray) -> np.ndarray:
    return crop < PAPER_LEVEL


def _ink_removed_fraction(original: np.ndarray, cleaned: np.ndarray) -> float:
    before = int(np.count_nonzero(_ink(original)))
    if before == 0:
        return 0.0
    after = int(np.count_nonzero(_ink(cleaned)))
    return (before - after) / before


def _guard_ink(original: np.ndarray, cleaned: np.ndarray) -> np.ndarray:
    if _ink_removed_fraction(original, cleaned) > INK_REMOVE_LIMIT:
        return original.copy()
    return cleaned


def _restore_if_empty(original: np.ndarray, cleaned: np.ndarray, total: int) -> np.ndarray:
    remaining = int(np.count_nonzero(_ink(cleaned)))
    if remaining < EMPTY_INK and total > 50:
        return original.copy()
    return cleaned


def _parts(shape, labels, stats, centroids, count: int) -> list[Part]:
    areas = [int(stats[index, cv2.CC_STAT_AREA]) for index in range(1, count)]
    median_area = float(np.median(areas))
    total = len(areas)
    return [
        _one_part(shape, labels, stats, centroids, index, median_area, total)
        for index in range(1, count)
    ]


def _one_part(shape, labels, stats, centroids, index: int, median_area: float, total: int) -> Part:
    width = int(stats[index, cv2.CC_STAT_WIDTH])
    height = int(stats[index, cv2.CC_STAT_HEIGHT])
    area = int(stats[index, cv2.CC_STAT_AREA])
    aspect = max(width / max(height, 1.0), height / max(width, 1.0))
    crop_h, crop_w = shape
    ys, xs = np.nonzero(labels == index)
    return Part(
        label=index,
        cx=float(centroids[index, 0]),
        cy=float(centroids[index, 1]),
        angle=_axis_angle(xs, ys),
        long_line=_is_long_line(width, height, crop_w, crop_h, aspect),
        glyph=_is_text_glyph(width, height, area, aspect, median_area, total),
        spanning=width >= SPAN_FRACTION * crop_w and height >= SPAN_FRACTION * crop_h,
    )


def _protected(parts: list[Part]) -> set[int]:
    kept = {part.label for part in parts if part.spanning}
    lines = [part for part in parts if part.long_line]
    for group in _groups(lines):
        if len(group) >= FAMILY_COUNT and _regular(group):
            kept.update(part.label for part in group)
    return kept


def _erase(crop, ink, labels, parts: list[Part], protected: set[int]) -> np.ndarray:
    paper = int(np.median(crop[ink == 0])) if np.any(ink == 0) else 255
    cleaned = crop.copy()
    for part in parts:
        if part.label not in protected and (part.long_line or part.glyph):
            cleaned[labels == part.label] = paper
    return cleaned


def _groups(lines: list[Part]) -> list[list[Part]]:
    pending = list(lines)
    groups = []
    while pending:
        groups.append(_absorb(pending))
    return groups


def _absorb(pending: list[Part]) -> list[Part]:
    group = [pending.pop(0)]
    moved = True
    while moved:
        moved = False
        center = _circular_mean([part.angle for part in group])
        stay = []
        for part in pending:
            if _angle_distance(part.angle, center) <= ORIENTATION_DEGREES:
                group.append(part)
                moved = True
            else:
                stay.append(part)
        pending[:] = stay
    return group


def _regular(group: list[Part]) -> bool:
    center = _circular_mean([part.angle for part in group])
    theta = np.deg2rad(center + 90.0)
    projections = np.sort([part.cx * np.cos(theta) + part.cy * np.sin(theta) for part in group])
    gaps = np.diff(projections)
    if len(gaps) < 2:
        return False
    mean = float(np.mean(gaps))
    if mean <= 1.0:
        return False
    return float(np.std(gaps)) / mean <= SPACING_CV_LIMIT


def _axis_angle(xs: np.ndarray, ys: np.ndarray) -> float:
    if len(xs) < 2:
        return 0.0
    dx = xs.astype(np.float64)
    dy = ys.astype(np.float64)
    dx -= dx.mean()
    dy -= dy.mean()
    xx = float(np.dot(dx, dx))
    yy = float(np.dot(dy, dy))
    xy = float(np.dot(dx, dy))
    degrees = 0.5 * float(np.rad2deg(np.arctan2(2.0 * xy, xx - yy)))
    return degrees % 180.0


def _angle_distance(left: float, right: float) -> float:
    delta = abs(left - right) % 180.0
    if delta > 90.0:
        return 180.0 - delta
    return delta


def _circular_mean(angles: list[float]) -> float:
    doubled = np.deg2rad(np.asarray(angles, dtype=np.float64) * 2.0)
    mean = np.arctan2(float(np.sin(doubled).mean()), float(np.cos(doubled).mean()))
    return float(np.rad2deg(mean) / 2.0) % 180.0
