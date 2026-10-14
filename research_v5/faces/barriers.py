"""Long dark strokes, gap closing, and the faces of the complement.

Thin hatch strokes are removed in two ways: a thickness cut on connected
components, and a parallel-family cut on long line segments. Thick strokes
stay even when they are parallel. Barrier pixels are then assigned to the
nearest face so every pixel belongs to one region.
"""

from __future__ import annotations

import cv2
import numpy as np

from settings import Settings


def build_faces(gray: np.ndarray, settings: Settings) -> np.ndarray:
    dark = np.uint8(gray <= settings.black_level)
    distance = cv2.distanceTransform(dark, cv2.DIST_L2, 3)
    segments = _dark_segments(gray, settings)
    thick = _thick_segments(segments, distance, settings)
    thin = _thin_segments(segments, distance, settings)
    kept_thin = thin[~_family_drop(thin, settings)] if len(thin) else thin
    barriers = _thick_components(dark, settings)
    _paint(barriers, thick)
    _paint(barriers, kept_thin)
    closed = _close_gaps(barriers, settings.gap_radius)
    return _label_faces(closed)


def _dark_segments(gray: np.ndarray, settings: Settings) -> np.ndarray:
    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_NONE)
    found = detector.detect(gray)[0]
    if found is None:
        return np.zeros((0, 4), dtype=np.float32)
    segments = found.reshape(-1, 4).astype(np.float32)
    lengths = np.hypot(segments[:, 2] - segments[:, 0], segments[:, 3] - segments[:, 1])
    darkness = _sample_median(gray, segments)
    keep = (lengths >= settings.min_length) & (darkness <= settings.black_level)
    return segments[keep]


def _thick_segments(segments: np.ndarray, distance: np.ndarray, settings: Settings) -> np.ndarray:
    if len(segments) == 0:
        return segments
    return segments[_sample_median(distance, segments) >= settings.lsd_thick_dist]


def _thin_segments(segments: np.ndarray, distance: np.ndarray, settings: Settings) -> np.ndarray:
    if len(segments) == 0:
        return segments
    return segments[_sample_median(distance, segments) < settings.lsd_thick_dist]


def _sample_median(image: np.ndarray, segments: np.ndarray) -> np.ndarray:
    if len(segments) == 0:
        return np.zeros(0, dtype=np.float32)
    fractions = np.linspace(0.15, 0.85, 5, dtype=np.float32)
    x0, y0, x1, y1 = (segments[:, i] for i in range(4))
    xs = x0[:, None] + (x1 - x0)[:, None] * fractions
    ys = y0[:, None] + (y1 - y0)[:, None] * fractions
    xi = np.clip(np.rint(xs).astype(np.int32), 0, image.shape[1] - 1)
    yi = np.clip(np.rint(ys).astype(np.int32), 0, image.shape[0] - 1)
    return np.median(image[yi, xi], axis=1)


def _family_drop(segments: np.ndarray, settings: Settings) -> np.ndarray:
    drop = np.zeros(len(segments), dtype=bool)
    if len(segments) < settings.min_run:
        return drop
    angles = _segment_angles(segments)
    bins = np.mod(np.rint(angles / settings.angle_bin_deg), 180.0 / settings.angle_bin_deg).astype(np.int32)
    mids = _midpoints(segments)
    for bin_index in range(int(round(180.0 / settings.angle_bin_deg))):
        members = np.flatnonzero(bins == bin_index)
        _mark_family(drop, members, mids, bin_index, settings)
    return drop


def _mark_family(
    drop: np.ndarray,
    members: np.ndarray,
    mids: np.ndarray,
    bin_index: int,
    settings: Settings,
) -> None:
    if members.size < settings.min_run:
        return
    theta = np.deg2rad(bin_index * settings.angle_bin_deg)
    projection = mids[members, 0] * (-np.sin(theta)) + mids[members, 1] * np.cos(theta)
    order = np.argsort(projection)
    ordered = members[order]
    unique = _unique_lines(projection[order], settings.spacing_min)
    _drop_regular_runs(drop, ordered, projection[order], unique, settings)


def _drop_regular_runs(
    drop: np.ndarray,
    ordered: np.ndarray,
    projection: np.ndarray,
    unique: np.ndarray,
    settings: Settings,
) -> None:
    if unique.size < settings.min_run:
        return
    spacing = np.diff(projection[unique])
    start = 0
    gaps = np.concatenate([spacing, np.array([settings.spacing_max + 1.0])])
    for index, gap in enumerate(gaps.tolist()):
        if gap <= settings.spacing_max:
            continue
        _drop_one_run(drop, ordered, unique, spacing, start, index, settings)
        start = index + 1


def _drop_one_run(
    drop: np.ndarray,
    ordered: np.ndarray,
    unique: np.ndarray,
    spacing: np.ndarray,
    start: int,
    index: int,
    settings: Settings,
) -> None:
    count = index - start + 1
    if count < settings.min_run:
        return
    run_spacing = spacing[start:index]
    median = float(np.median(run_spacing))
    if median <= 1e-3:
        return
    if float(np.std(run_spacing) / median) > settings.spacing_cv:
        return
    drop[ordered[unique[start:index + 1]]] = True


def _unique_lines(projection: np.ndarray, spacing_min: float) -> np.ndarray:
    keep = [0]
    for index in range(1, projection.size):
        if projection[index] - projection[keep[-1]] >= spacing_min:
            keep.append(index)
    return np.asarray(keep, dtype=np.int32)


def _thick_components(dark: np.ndarray, settings: Settings) -> np.ndarray:
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(dark, connectivity=8)
    if count <= 1:
        return np.zeros(dark.shape, dtype=np.uint8)
    lengths = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]).astype(np.float32)
    area = stats[:, cv2.CC_STAT_AREA].astype(np.float32)
    thickness = area / np.maximum(lengths, 1.0)
    keep = (
        (lengths >= settings.min_length)
        & (thickness >= settings.component_min_thickness)
        & (thickness <= settings.component_max_thickness)
    )
    keep[0] = False
    return np.uint8(keep)[labels]


def _close_gaps(barriers: np.ndarray, gap_radius: int) -> np.ndarray:
    # Disk radius `gap_radius` bridges gaps up to about two radii.
    radius = max(1, int(gap_radius))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.morphologyEx(barriers, cv2.MORPH_CLOSE, kernel)


def _label_faces(barriers: np.ndarray) -> np.ndarray:
    free = np.uint8(barriers == 0)
    _count, labels = cv2.connectedComponents(free, connectivity=4)
    if not np.any(barriers):
        return labels.astype(np.int32)
    return _assign_barriers(labels)


def _assign_barriers(labels: np.ndarray) -> np.ndarray:
    source = np.where(labels == 0, np.uint8(255), np.uint8(0))
    _distance, nearest = cv2.distanceTransformWithLabels(
        source, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
    )
    barrier = labels == 0
    flat = nearest[barrier].astype(np.int64) - 1
    flat = np.clip(flat, 0, labels.size - 1)
    ys, xs = np.unravel_index(flat, labels.shape)
    filled = labels.copy()
    filled[barrier] = labels[ys, xs]
    return filled.astype(np.int32)


def _paint(mask: np.ndarray, segments: np.ndarray) -> None:
    for x0, y0, x1, y1 in segments.tolist():
        cv2.line(mask, (int(x0), int(y0)), (int(x1), int(y1)), 1, 2, cv2.LINE_8)


def _segment_angles(segments: np.ndarray) -> np.ndarray:
    dy = segments[:, 3] - segments[:, 1]
    dx = segments[:, 2] - segments[:, 0]
    return np.degrees(np.arctan2(dy, dx)) % 180.0


def _midpoints(segments: np.ndarray) -> np.ndarray:
    return np.column_stack((
        (segments[:, 0] + segments[:, 2]) * 0.5,
        (segments[:, 1] + segments[:, 3]) * 0.5,
    ))
