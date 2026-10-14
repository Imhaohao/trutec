"""Label-free face matcher.

Consumes a drawing, the query box, and any context boxes. It does not read
labels, document ids, or query ids.
"""

from __future__ import annotations

import cv2
import numpy as np

from barriers import build_faces
from scoring import sample_maps, score_maps
from settings import Settings
from signature import SigGrid, find_exemplar, measure_sheet

_SHEETS: dict[tuple[str, Settings], tuple[SigGrid, np.ndarray]] = {}


def clear_cache() -> None:
    _SHEETS.clear()


def variant_scores(
    request: dict,
    rgb: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    settings: Settings,
) -> dict[str, np.ndarray]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    grid, faces = sheet(str(request["image"]), rgb, gray, settings)
    exemplar = find_exemplar(rgb, gray, _boxes(request), settings)
    maps = score_maps(exemplar, grid, faces, gray, settings)
    return sample_maps(maps, grid, faces, y, x)


def sheet(
    image: str,
    rgb: np.ndarray,
    gray: np.ndarray,
    settings: Settings,
) -> tuple[SigGrid, np.ndarray]:
    key = (image, settings)
    cached = _SHEETS.get(key)
    if cached is not None:
        return cached
    if len(_SHEETS) >= 16:
        _SHEETS.clear()
    built = (measure_sheet(rgb, gray, settings), build_faces(gray, settings))
    _SHEETS[key] = built
    return built


def _boxes(request: dict) -> tuple[tuple[int, int, int, int], ...]:
    query = tuple(int(v) for v in request["query_box"])
    context = tuple(tuple(int(v) for v in box) for box in request.get("context_boxes") or ())
    return (query, *context)
