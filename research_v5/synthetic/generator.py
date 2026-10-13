"""Synthetic CAD sheet generator producing realistic hatch-matching training examples."""

import cv2
import numpy as np

from research_v5.synthetic.elements import (
    draw_crossing_lines,
    draw_dimension_line,
    draw_leader_arrow,
    draw_text_block,
    draw_title_block,
    draw_wall_gaps,
    draw_walls,
)
from research_v5.synthetic.geometry import generate_rooms
from research_v5.synthetic.patterns import PATTERN_DISPATCH, render_pattern

PATTERN_FAMILIES = [
    "parallel_lines",
    "dashed_lines",
    "crosshatch",
    "brick",
    "dots",
    "hexagons",
    "circles",
    "triangles",
    "plus_cross",
    "concrete",
    "solid_fill",
]

COLOR_PALETTES = [
    ("black", (0, 0, 0)),
    ("dark_gray", (60, 60, 60)),
    ("gray", (120, 120, 120)),
    ("red", (30, 30, 180)),
    ("blue", (180, 50, 30)),
    ("cyan", (180, 140, 20)),
    ("green", (40, 140, 40)),
    ("orange", (20, 110, 200)),
]


def sample_pattern_params(family, rng):
    """Sample random parameters for a pattern family."""
    is_colored = rng.rand() < 0.30
    if is_colored:
        color_idx = rng.randint(3, len(COLOR_PALETTES))
    else:
        color_idx = rng.randint(0, 3)
    color_name, color_bgr = COLOR_PALETTES[color_idx]

    params = {
        "family": family,
        "color_name": color_name,
        "color": color_bgr,
        "line_width": int(rng.choice([1, 2], p=[0.7, 0.3])),
    }

    if family in ("parallel_lines", "dashed_lines", "crosshatch"):
        params["angle"] = int(rng.choice([0, 15, 30, 45, 60, 75, 90, 120, 135, 150]))
        params["spacing"] = int(rng.randint(6, 36))
        if family == "dashed_lines":
            params["dash_len"] = int(rng.randint(6, 16))
            params["gap_len"] = int(rng.randint(4, 12))
        elif family == "crosshatch":
            params["cross_angle"] = int(rng.choice([60, 90]))
    elif family == "brick":
        params["course_height"] = int(rng.randint(8, 28))
        params["brick_width"] = int(params["course_height"] * rng.uniform(1.8, 3.2))
        params["angle"] = int(rng.choice([0, 45, 90]))
    elif family == "dots":
        params["spacing"] = int(rng.randint(8, 28))
        params["radius"] = int(rng.choice([1, 2, 3], p=[0.5, 0.35, 0.15]))
        params["random_stipple"] = bool(rng.rand() < 0.25)
        params["density"] = float(rng.uniform(0.002, 0.015))
        params["hex_shift"] = bool(rng.rand() < 0.5)
        params["seed"] = int(rng.randint(0, 100000))
    elif family == "hexagons":
        params["radius"] = int(rng.randint(8, 24))
        params["angle"] = int(rng.choice([0, 15, 30, 45]))
    elif family == "circles":
        params["spacing"] = int(rng.randint(12, 32))
        params["radius"] = int(rng.randint(3, params["spacing"] // 2 - 2))
        params["concentric"] = bool(rng.rand() < 0.3)
    elif family == "triangles":
        params["size"] = int(rng.randint(10, 24))
        params["spacing"] = int(params["size"] + rng.randint(4, 16))
    elif family == "plus_cross":
        params["size"] = int(rng.randint(6, 16))
        params["spacing"] = int(params["size"] + rng.randint(6, 18))
        params["diagonal"] = bool(rng.rand() < 0.5)
    elif family == "concrete":
        params["seed"] = int(rng.randint(0, 100000))
    elif family == "solid_fill":
        params["gray_level"] = int(rng.randint(180, 230))

    return params


def make_near_twin_params(base_params, rng):
    """Create a near twin trap with modified spacing/angle or color."""
    twin = dict(base_params)
    family = twin["family"]
    trap_mode = rng.choice(["spacing_angle", "color"])

    if trap_mode == "spacing_angle":
        if "spacing" in twin:
            delta_frac = rng.choice([-0.35, 0.35])
            twin["spacing"] = max(4, int(twin["spacing"] * (1.0 + delta_frac)))
        elif "course_height" in twin:
            delta_frac = rng.choice([-0.35, 0.35])
            twin["course_height"] = max(6, int(twin["course_height"] * (1.0 + delta_frac)))
            twin["brick_width"] = max(10, int(twin["brick_width"] * (1.0 + delta_frac)))
        elif "radius" in twin:
            delta_frac = rng.choice([-0.35, 0.35])
            twin["radius"] = max(2, int(twin["radius"] * (1.0 + delta_frac)))

        if "angle" in twin:
            angle_delta = int(rng.choice([-30, 30, 45, -45]))
            twin["angle"] = (twin["angle"] + angle_delta) % 180
    else:
        # Color trap
        if twin["color_name"] in ("black", "dark_gray", "gray"):
            color_idx = rng.randint(3, len(COLOR_PALETTES))
        else:
            color_idx = rng.randint(0, 2)
        twin["color_name"], twin["color"] = COLOR_PALETTES[color_idx]

    return twin


def sample_query_box_inside_mask(mask, rng, target_size=None):
    """Find a valid query bounding box completely contained inside mask."""
    # Compute distance transform to find safe deep interior
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    valid_ys, valid_xs = np.where(dist >= 16)
    if len(valid_xs) == 0:
        valid_ys, valid_xs = np.where(mask)
        if len(valid_xs) == 0:
            return None

    # Pick random interior anchor
    idx = rng.randint(0, len(valid_xs))
    cx, cy = int(valid_xs[idx]), int(valid_ys[idx])
    max_radius = int(dist[cy, cx])

    # Sample query box size (32 to 192 px)
    if target_size is not None:
        qw, qh = target_size
    else:
        side = rng.randint(32, min(193, max(33, max_radius * 2)))
        side = min(side, 192)
        qw = max(32, side)
        qh = max(32, int(qw * rng.uniform(0.8, 1.2)))

    # Align inside
    x0 = max(0, cx - qw // 2)
    y0 = max(0, cy - qh // 2)
    x1 = min(mask.shape[1], x0 + qw)
    y1 = min(mask.shape[0], y0 + qh)

    # Check fully contained
    box_mask = mask[y0:y1, x0:x1]
    if box_mask.all():
        return [int(x0), int(y0), int(x1), int(y1)]

    # Fallback search
    for r in range(min(max_radius - 2, 80), 15, -4):
        x0, y0 = cx - r, cy - r
        x1, y1 = cx + r, cy + r
        if x0 >= 0 and y0 >= 0 and x1 <= mask.shape[1] and y1 <= mask.shape[0]:
            if mask[y0:y1, x0:x1].all() and (x1 - x0 >= 32) and (y1 - y0 >= 32):
                return [int(x0), int(y0), int(x1), int(y1)]

    return None


def generate_known_mask(height, width, rng, positive_masks=None):
    """Generate known mask leaving a 0-30% unknown band (e.g. margin or partial review slice).

    Any positive labeled pixel must be marked known so label lies inside explicit known domain.
    """
    known = np.ones((height, width), dtype=bool)
    has_unknown = rng.rand() < 0.6
    if not has_unknown:
        return known

    # Mark a band or corner unknown
    unknown_frac = rng.uniform(0.05, 0.28)
    side = rng.choice(["top", "bottom", "left", "right", "corner"])
    if side == "top":
        h_un = int(height * unknown_frac)
        known[:h_un, :] = False
    elif side == "bottom":
        h_un = int(height * unknown_frac)
        known[-h_un:, :] = False
    elif side == "left":
        w_un = int(width * unknown_frac)
        known[:, :w_un] = False
    elif side == "right":
        w_un = int(width * unknown_frac)
        known[:, -w_un:] = False
    elif side == "corner":
        frac = np.sqrt(unknown_frac)
        w_un = int(width * frac)
        h_un = int(height * frac)
        known[:h_un, :w_un] = False

    # Ensure all positive pixels in any positive mask remain known
    if positive_masks is not None:
        for pm in positive_masks:
            known[pm] = True

    return known


def add_sheet_overlays(img, polygons, rng, query_box=None):
    """Add annotations, dimension lines, leader arrows, title blocks, and crossing lines."""
    h, w = img.shape[:2]

    # Title block
    tb_w, tb_h = rng.randint(220, 320), rng.randint(120, 180)
    corner = rng.choice(["br", "bl", "tr"])
    if corner == "br":
        tb_x, tb_y = w - tb_w - 40, h - tb_h - 40
    elif corner == "bl":
        tb_x, tb_y = 40, h - tb_h - 40
    else:
        tb_x, tb_y = w - tb_w - 40, 40
    draw_title_block(img, tb_x, tb_y, tb_w, tb_h)

    # Dimension lines around walls
    for poly in polygons:
        if rng.rand() < 0.4:
            n_pts = len(poly)
            e = rng.randint(0, n_pts)
            p1 = poly[e]
            p2 = poly[(e + 1) % n_pts]
            draw_dimension_line(img, p1, p2)

    # Room text labels inside polygons
    room_labels = [
        ["OFFICE 101", "FIN: CPT"],
        ["CONFERENCE", "FIN: WD"],
        ["STORAGE", "FIN: CONC"],
        ["CORRIDOR", "CLG: 9'-0\""],
        ["RESTROOM", "FIN: CT"],
        ["ELECTRICAL", "FIN: SEALED"],
        ["BREAK ROOM", "FIN: VCT"],
        ["LOBBY", "FIN: TERRAZZO"],
    ]
    for idx, poly in enumerate(polygons):
        if rng.rand() < 0.7:
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            cx, cy = int(np.mean(xs)), int(np.mean(ys))
            lbl = room_labels[idx % len(room_labels)]
            draw_text_block(img, (cx - 40, cy - 10), lbl)

    # Leader arrows
    for _ in range(rng.randint(2, 6)):
        p1 = (rng.randint(100, w - 100), rng.randint(100, h - 100))
        p2 = (p1[0] + rng.randint(-80, 80), p1[1] + rng.randint(-80, 80))
        draw_leader_arrow(img, p1, p2, text=rng.choice(["TYP.", "SIM.", "SEE DET. 4", "VERIFY"]))

    # Crossing straight lines & curves
    draw_crossing_lines(img, rng, num_lines=rng.randint(2, 5))

    # Overlaid text specifically intersecting query box if requested
    if query_box is not None and rng.rand() < 0.35:
        qx0, qy0, qx1, qy1 = query_box
        cx, cy = (qx0 + qx1) // 2, (qy0 + qy1) // 2
        draw_text_block(img, (cx - 20, cy), ["MATCH EXTG.", "TYP."], scale=0.35)
