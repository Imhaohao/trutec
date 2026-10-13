"""Query crop cleaning rules for hatch descriptor extraction.

Implements three label-free query cleaning rules:
1. Rule (a) IQR outlier cell rejection: drop crop cells whose ink density or
   gradient energy is an outlier (> 1.5 IQR) versus the crop's own cells.
2. Rule (b) Autocorrelation exemplar sub-window: select the sub-window with
   strongest nonzero-period autocorrelation peak.
3. Rule (c) Connected component filtering: remove ink components matching text
   glyphs or long boundary lines.
"""
from __future__ import annotations

import bootstrap  # noqa: F401

import cv2
import numpy as np

import hatch_texture
from settings import DEFAULT, Settings
from signature import find_exemplar


def clean_descriptor_iqr(crop: np.ndarray, stride: int = 4) -> np.ndarray:
    """Rule (a): Drop crop cells beyond 1.5 IQR on ink density or gradient energy."""
    cells = hatch_texture.texture_features(crop, stride=stride)
    flat = cells.reshape(-1, cells.shape[-1])
    if len(flat) <= 2:
        return np.asarray(np.median(flat, axis=0), dtype=np.float32)

    ink_density = flat[:, 0]
    grad_energy = np.sum(flat[:, 6:10], axis=1)

    out_density = _iqr_outliers(ink_density)
    out_grad = _iqr_outliers(grad_energy)
    outliers = out_density | out_grad
    valid = ~outliers

    if np.any(valid):
        return np.asarray(np.median(flat[valid], axis=0), dtype=np.float32)
    return np.asarray(np.median(flat, axis=0), dtype=np.float32)


def clean_descriptor_autocorr(
    rgb: np.ndarray,
    gray: np.ndarray,
    query_box: tuple[int, int, int, int],
    settings: Settings = DEFAULT,
    stride: int = 4,
) -> np.ndarray:
    """Rule (b): Use cells from the strongest periodic autocorrelation sub-window."""
    exemplar = find_exemplar(rgb, gray, (query_box,), settings)
    if exemplar is not None and exemplar.kind == 1 and exemplar.strength > 0.0:
        if exemplar.template.size > 0:
            cells = hatch_texture.texture_features(exemplar.template, stride=stride)
            return np.asarray(np.median(cells.reshape(-1, cells.shape[-1]), axis=0), dtype=np.float32)

    x0, y0, x1, y1 = [int(v) for v in query_box]
    crop = gray[y0:y1, x0:x1]
    cells = hatch_texture.texture_features(crop, stride=stride)
    return np.asarray(np.median(cells.reshape(-1, cells.shape[-1]), axis=0), dtype=np.float32)


def clean_crop_components(crop: np.ndarray, paper_level: int = 210) -> np.ndarray:
    """Rule (c) helper: Remove text glyph and line components from crop."""
    ch, cw = crop.shape
    ink = (crop < paper_level).astype(np.uint8)
    total_ink = int(np.sum(ink))
    if total_ink < 10:
        return crop.copy()

    paper_val = int(np.median(crop[ink == 0])) if np.any(ink == 0) else 255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    if num_labels <= 1:
        return crop.copy()

    areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)]
    med_area = float(np.median(areas)) if areas else 0.0

    cleaned = crop.copy()
    for i in range(1, num_labels):
        if _is_removable_component(stats[i], ch, cw, med_area, len(areas)):
            cleaned[labels == i] = paper_val

    remaining_ink = int(np.sum(cleaned < paper_level))
    if remaining_ink < 10 and total_ink > 50:
        return crop.copy()
    return cleaned


def clean_descriptor_components(
    crop: np.ndarray, paper_level: int = 210, stride: int = 4
) -> np.ndarray:
    """Rule (c): Clean glyphs and long lines, then compute median feature vector."""
    cleaned = clean_crop_components(crop, paper_level=paper_level)
    cells = hatch_texture.texture_features(cleaned, stride=stride)
    return np.asarray(np.median(cells.reshape(-1, cells.shape[-1]), axis=0), dtype=np.float32)


def clean_descriptor_combined(
    rgb: np.ndarray,
    gray: np.ndarray,
    query_box: tuple[int, int, int, int],
    paper_level: int = 210,
    stride: int = 4,
) -> np.ndarray:
    """Combined rule: Clean glyphs/lines first, then reject IQR outlier cells."""
    x0, y0, x1, y1 = [int(v) for v in query_box]
    crop = gray[y0:y1, x0:x1]
    cleaned = clean_crop_components(crop, paper_level=paper_level)
    return clean_descriptor_iqr(cleaned, stride=stride)


def _iqr_outliers(values: np.ndarray) -> np.ndarray:
    q25, q75 = np.quantile(values, [0.25, 0.75])
    iqr = q75 - q25
    low = q25 - 1.5 * iqr
    high = q75 + 1.5 * iqr
    return (values < low) | (values > high)


def _is_removable_component(stat: np.ndarray, ch: int, cw: int, med_area: float, count: int) -> bool:
    x = stat[cv2.CC_STAT_LEFT]
    y = stat[cv2.CC_STAT_TOP]
    w = stat[cv2.CC_STAT_WIDTH]
    h = stat[cv2.CC_STAT_HEIGHT]
    area = stat[cv2.CC_STAT_AREA]

    if area > 0.6 * ch * cw and w > 0.8 * cw and h > 0.8 * ch:
        return False

    aspect = max(float(w) / max(float(h), 1.0), float(h) / max(float(w), 1.0))
    if _is_long_line(w, h, cw, ch, aspect):
        return True
    return _is_text_glyph(w, h, area, aspect, med_area, count)


def _is_long_line(w: int, h: int, cw: int, ch: int, aspect: float) -> bool:
    h_line = w >= 0.5 * cw and aspect >= 4.0 and h <= max(6, int(0.1 * ch))
    v_line = h >= 0.5 * ch and aspect >= 4.0 and w <= max(6, int(0.1 * cw))
    return h_line or v_line


def _is_text_glyph(
    w: int, h: int, area: int, aspect: float, med_area: float, count: int
) -> bool:
    size_ok = 8 <= h <= 45 and 6 <= w <= 45 and aspect <= 2.5 and 40 <= area <= 700
    if not size_ok:
        return False
    return area >= 3.0 * med_area or count >= 5
