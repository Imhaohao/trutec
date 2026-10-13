"""Floorplan and room polygon generator for synthetic CAD sheets."""

import numpy as np


def generate_rect_partition(rng, width=2048, height=2048, margin=64):
    """Generate non-overlapping rectangular rooms partitioned from a master boundary."""
    x0, y0 = margin, margin
    x1, y1 = width - margin, height - margin

    # Recursive split or grid split
    # Let's do a structured split: 2 to 4 vertical bands, each split into 1 to 4 horizontal rooms
    n_cols = rng.randint(2, 5)
    xs = sorted([x0, x1] + list(rng.randint(x0 + 100, x1 - 100, size=n_cols - 1)))
    # Ensure distinct
    xs = sorted(list(set(xs)))
    if len(xs) < 2:
        xs = [x0, x1]

    polygons = []
    for c in range(len(xs) - 1):
        col_x0, col_x1 = xs[c], xs[c + 1]
        n_rows = rng.randint(1, 4)
        ys = sorted([y0, y1] + list(rng.randint(y0 + 100, y1 - 100, size=n_rows - 1)))
        ys = sorted(list(set(ys)))
        if len(ys) < 2:
            ys = [y0, y1]
        for r in range(len(ys) - 1):
            row_y0, row_y1 = ys[r], ys[r + 1]
            if (col_x1 - col_x0 >= 80) and (row_y1 - row_y0 >= 80):
                poly = [
                    [col_x0, row_y0],
                    [col_x1, row_y0],
                    [col_x1, row_y1],
                    [col_x0, row_y1],
                ]
                polygons.append(poly)
    return polygons


def generate_l_shaped_room(rect, split_axis, split_frac, drop_corner):
    """Cut one corner from a rectangle to make an L-shape polygon."""
    x0, y0 = rect[0]
    x1, y1 = rect[2]
    w = x1 - x0
    h = y1 - y0
    if w < 120 or h < 120:
        return rect

    # split
    mid_x = int(x0 + w * split_frac)
    mid_y = int(y0 + h * split_frac)

    # 4 corners: 0=(top-left), 1=(top-right), 2=(bottom-right), 3=(bottom-left)
    if drop_corner == 0:
        return [[mid_x, y0], [x1, y0], [x1, y1], [x0, y1], [x0, mid_y], [mid_x, mid_y]]
    elif drop_corner == 1:
        return [[x0, y0], [mid_x, y0], [mid_x, mid_y], [x1, mid_y], [x1, y1], [x0, y1]]
    elif drop_corner == 2:
        return [[x0, y0], [x1, y0], [x1, mid_y], [mid_x, mid_y], [mid_x, y1], [x0, y1]]
    else:
        return [[x0, y0], [x1, y0], [x1, y1], [mid_x, y1], [mid_x, mid_y], [x0, mid_y]]


def generate_rooms(rng, width=2048, height=2048):
    """Generate 4 to 12 realistic room polygons with shared walls."""
    margin = rng.randint(40, 100)
    rects = generate_rect_partition(rng, width, height, margin)

    # Target 4 to 12 polygons
    while len(rects) > 12:
        # merge two adjacent rects
        rects.pop(rng.randint(0, len(rects)))

    polygons = []
    for rect in rects:
        # maybe convert some rect to L-shape or chamfered polygon
        if rng.rand() < 0.25 and len(polygons) < 11:
            split_frac = rng.uniform(0.3, 0.7)
            corner = rng.randint(0, 4)
            l_poly = generate_l_shaped_room(rect, 0, split_frac, corner)
            polygons.append(l_poly)
        else:
            polygons.append(rect)

    if len(polygons) < 4:
        # Subdivide the largest polygon
        poly = polygons.pop(0)
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        mid_x = (min_x + max_x) // 2
        mid_y = (min_y + max_y) // 2
        p1 = [[min_x, min_y], [mid_x, min_y], [mid_x, max_y], [min_x, max_y]]
        p2 = [[mid_x, min_y], [max_x, min_y], [max_x, max_y], [mid_x, max_y]]
        polygons.extend([p1, p2])

    return polygons[:12]
