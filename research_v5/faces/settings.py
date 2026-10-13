"""Frozen constants for the label-free face matcher.

Geometry knobs that were compared on the fit split are recorded in
results/fit-selection.json. Do not retune them from development scores.
"""

from __future__ import annotations

from dataclasses import dataclass


PLAIN = 0
PERIODIC = 1
SOLID = 2


@dataclass(frozen=True)
class Settings:
    black_level: int = 120
    min_length: int = 48
    lsd_thick_dist: float = 1.55
    component_min_thickness: float = 2.2
    component_max_thickness: float = 18.0
    gap_radius: int = 18
    min_run: int = 5
    spacing_min: float = 3.0
    spacing_max: float = 40.0
    spacing_cv: float = 0.40
    angle_bin_deg: float = 15.0
    tile: int = 96
    tile_stride: int = 48
    min_period: int = 4
    max_period: int = 48
    min_strength: float = 0.20
    inherit_tiles: int = 2
    large_face_area: int = 256 * 256
    paper_level: int = 235
    ncc_search: int = 21
    solid_std: float = 18.0
    solid_cover: float = 0.72
    solid_stroke: float = 0.06


DEFAULT = Settings()
