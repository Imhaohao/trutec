"""Drawing elements for synthetic CAD sheets: boundaries, doors, dimensions, text, tables, lines."""

import cv2
import numpy as np


def draw_walls(img, polygons, wall_thickness=3, wall_color=(30, 30, 30)):
    """Draw polygon boundaries onto the image with specified thickness."""
    for poly in polygons:
        pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(
            img,
            [pts],
            isClosed=True,
            color=wall_color,
            thickness=wall_thickness,
            lineType=cv2.LINE_AA,
        )


def draw_wall_gaps(img, polygons, gaps, bg_color=(255, 255, 255)):
    """Clear small gaps (e.g. door openings) on boundary edges."""
    for p_idx, edge_idx, frac, gap_size in gaps:
        if p_idx >= len(polygons):
            continue
        poly = polygons[p_idx]
        n_pts = len(poly)
        p1 = np.array(poly[edge_idx], dtype=np.float32)
        p2 = np.array(poly[(edge_idx + 1) % n_pts], dtype=np.float32)
        edge_vec = p2 - p1
        edge_len = np.linalg.norm(edge_vec)
        if edge_len < gap_size + 10:
            continue
        center = p1 + frac * edge_vec
        unit_vec = edge_vec / edge_len
        g1 = tuple((center - (gap_size / 2.0) * unit_vec).astype(np.int32))
        g2 = tuple((center + (gap_size / 2.0) * unit_vec).astype(np.int32))
        cv2.line(img, g1, g2, bg_color, thickness=8, lineType=cv2.LINE_AA)


def draw_dimension_line(
    img, p1, p2, text=None, color=(50, 50, 50), thickness=1
):
    """Draw an architectural dimension line with ticks/arrows and measurement text."""
    p1 = np.array(p1, dtype=np.float32)
    p2 = np.array(p2, dtype=np.float32)
    vec = p2 - p1
    dist = np.linalg.norm(vec)
    if dist < 20:
        return
    unit = vec / dist
    perp = np.array([-unit[1], unit[0]], dtype=np.float32)

    # Main line
    cv2.line(
        img,
        tuple(p1.astype(np.int32)),
        tuple(p2.astype(np.int32)),
        color,
        thickness,
        cv2.LINE_AA,
    )

    # 45-degree architectural slash ticks at ends
    tick_len = 6
    for pt in (p1, p2):
        t1 = pt - (unit + perp) * (tick_len / 1.414)
        t2 = pt + (unit + perp) * (tick_len / 1.414)
        cv2.line(
            img,
            tuple(t1.astype(np.int32)),
            tuple(t2.astype(np.int32)),
            color,
            thickness + 1,
            cv2.LINE_AA,
        )

    # Dimension text centered above line
    if text is None:
        text = f"{int(dist * 0.125)}'-{int((dist * 1.5) % 12)}\""
    mid = (p1 + p2) / 2.0 + perp * 8
    cv2.putText(
        img,
        text,
        tuple(mid.astype(np.int32)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        color,
        1,
        cv2.LINE_AA,
    )


def draw_leader_arrow(
    img, start_pt, end_pt, text="TYP.", color=(40, 40, 40), thickness=1
):
    """Draw a leader line with arrow and annotation text."""
    p1 = tuple(int(x) for x in start_pt)
    p2 = tuple(int(x) for x in end_pt)
    cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)

    # Arrow head at start_pt
    vec = np.array(p2, dtype=np.float32) - np.array(p1, dtype=np.float32)
    dist = np.linalg.norm(vec)
    if dist > 5:
        unit = vec / dist
        perp = np.array([-unit[1], unit[0]], dtype=np.float32)
        arrow_p1 = np.array(p1, dtype=np.float32) + unit * 8 + perp * 4
        arrow_p2 = np.array(p1, dtype=np.float32) + unit * 8 - perp * 4
        cv2.line(
            img,
            p1,
            tuple(arrow_p1.astype(np.int32)),
            color,
            thickness,
            cv2.LINE_AA,
        )
        cv2.line(
            img,
            p1,
            tuple(arrow_p2.astype(np.int32)),
            color,
            thickness,
            cv2.LINE_AA,
        )

    # Text at end_pt
    text_pos = (p2[0] + 4, p2[1] + 4)
    cv2.putText(
        img,
        text,
        text_pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        color,
        1,
        cv2.LINE_AA,
    )


def draw_text_block(img, pos, lines, color=(30, 30, 30), scale=0.4):
    """Draw multi-line CAD notes or room label."""
    x, y = pos
    line_h = int(scale * 30) + 4
    for i, line in enumerate(lines):
        cv2.putText(
            img,
            line,
            (x, y + i * line_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            1,
            cv2.LINE_AA,
        )


def draw_title_block(img, x0, y0, width, height, color=(30, 30, 30)):
    """Draw a CAD title block / schedule table in a corner or region."""
    x1, y1 = x0 + width, y0 + height
    cv2.rectangle(img, (x0, y0), (x1, y1), color, 2, cv2.LINE_AA)

    # Horizontal divider rows
    num_rows = 4
    row_h = height // num_rows
    for r in range(1, num_rows):
        y = y0 + r * row_h
        cv2.line(img, (x0, y), (x1, y), color, 1, cv2.LINE_AA)

    # Vertical divider
    x_mid = x0 + width // 2
    cv2.line(img, (x_mid, y0), (x_mid, y1), color, 1, cv2.LINE_AA)

    labels = ["PROJECT:", "A-101", "DRAWN BY:", "SCALE: 1/4\"=1'", "DATE:", "OCT 2026", "REV:", "01"]
    for idx, lbl in enumerate(labels):
        r = idx // 2
        c = idx % 2
        lx = x0 + 6 if c == 0 else x_mid + 6
        ly = y0 + r * row_h + row_h // 2 + 4
        cv2.putText(
            img,
            lbl,
            (lx, ly),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            color,
            1,
            cv2.LINE_AA,
        )


def draw_crossing_lines(img, rng, num_lines=3, color=(80, 80, 80)):
    """Draw long straight lines or arcs crossing the sheet."""
    h, w = img.shape[:2]
    for _ in range(num_lines):
        is_curve = rng.rand() < 0.4
        if is_curve:
            center = (rng.randint(0, w), rng.randint(0, h))
            axes = (rng.randint(300, 1200), rng.randint(200, 800))
            angle = rng.randint(0, 180)
            start_a = rng.randint(0, 180)
            end_a = start_a + rng.randint(60, 180)
            cv2.ellipse(
                img,
                center,
                axes,
                angle,
                start_a,
                end_a,
                color,
                1,
                cv2.LINE_AA,
            )
        else:
            p1 = (rng.randint(0, w), rng.randint(0, h))
            p2 = (rng.randint(0, w), rng.randint(0, h))
            cv2.line(img, p1, p2, color, 1, cv2.LINE_AA)
