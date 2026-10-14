"""Pattern rendering functions for synthetic hatch matching generator."""

import cv2
import numpy as np


def create_solid_fill(params, height, width):
    val = params.get("gray_level", 200)
    img = np.full((height, width, 3), val, dtype=np.uint8)
    return img


def create_parallel_lines(params, height, width):
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    angle_deg = params.get("angle", 45)
    spacing = max(4, params.get("spacing", 12))
    line_width = max(1, params.get("line_width", 1))
    color = params.get("color", (0, 0, 0))

    diag = int(np.hypot(width, height)) + 100
    pattern_size = diag * 2
    temp = np.full((pattern_size, pattern_size, 3), 255, dtype=np.uint8)

    for x in range(0, pattern_size, spacing):
        cv2.line(temp, (x, 0), (x, pattern_size), color, line_width, cv2.LINE_AA)

    rot_mat = cv2.getRotationMatrix2D(
        (pattern_size // 2, pattern_size // 2), angle_deg, 1.0
    )
    rotated = cv2.warpAffine(
        temp,
        rot_mat,
        (pattern_size, pattern_size),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )

    cx, cy = pattern_size // 2, pattern_size // 2
    x0, y0 = cx - width // 2, cy - height // 2
    img = rotated[y0 : y0 + height, x0 : x0 + width]
    return img


def create_dashed_lines(params, height, width):
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    angle_deg = params.get("angle", 45)
    spacing = max(4, params.get("spacing", 12))
    dash_len = max(4, params.get("dash_len", 8))
    gap_len = max(3, params.get("gap_len", 6))
    line_width = max(1, params.get("line_width", 1))
    color = params.get("color", (0, 0, 0))

    diag = int(np.hypot(width, height)) + 100
    pattern_size = diag * 2
    temp = np.full((pattern_size, pattern_size, 3), 255, dtype=np.uint8)

    cycle = dash_len + gap_len
    for x in range(0, pattern_size, spacing):
        offset = (x // spacing * 7) % cycle
        y = -offset
        while y < pattern_size:
            y_end = min(y + dash_len, pattern_size)
            if y_end > 0:
                cv2.line(
                    temp,
                    (x, max(0, y)),
                    (x, y_end),
                    color,
                    line_width,
                    cv2.LINE_AA,
                )
            y += cycle

    rot_mat = cv2.getRotationMatrix2D(
        (pattern_size // 2, pattern_size // 2), angle_deg, 1.0
    )
    rotated = cv2.warpAffine(
        temp,
        rot_mat,
        (pattern_size, pattern_size),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )

    cx, cy = pattern_size // 2, pattern_size // 2
    x0, y0 = cx - width // 2, cy - height // 2
    img = rotated[y0 : y0 + height, x0 : x0 + width]
    return img


def create_crosshatch(params, height, width):
    p1 = dict(params)
    p2 = dict(params)
    ang = params.get("angle", 45)
    cross_angle = params.get("cross_angle", 90)
    p2["angle"] = (ang + cross_angle) % 180

    im1 = create_parallel_lines(p1, height, width)
    im2 = create_parallel_lines(p2, height, width)
    return np.minimum(im1, im2)


def create_brick(params, height, width):
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    course_h = max(6, params.get("course_height", 12))
    brick_w = max(10, params.get("brick_width", course_h * 2))
    line_width = max(1, params.get("line_width", 1))
    color = params.get("color", (0, 0, 0))

    diag = int(np.hypot(width, height)) + 100
    pattern_size = diag * 2
    temp = np.full((pattern_size, pattern_size, 3), 255, dtype=np.uint8)

    row = 0
    for y in range(0, pattern_size, course_h):
        cv2.line(temp, (0, y), (pattern_size, y), color, line_width, cv2.LINE_AA)
        x_shift = (brick_w // 2) if (row % 2 == 1) else 0
        for x in range(x_shift, pattern_size, brick_w):
            cv2.line(
                temp, (x, y), (x, y + course_h), color, line_width, cv2.LINE_AA
            )
        row += 1

    angle_deg = params.get("angle", 0)
    if angle_deg != 0:
        rot_mat = cv2.getRotationMatrix2D(
            (pattern_size // 2, pattern_size // 2), angle_deg, 1.0
        )
        temp = cv2.warpAffine(
            temp,
            rot_mat,
            (pattern_size, pattern_size),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )

    cx, cy = pattern_size // 2, pattern_size // 2
    x0, y0 = cx - width // 2, cy - height // 2
    return temp[y0 : y0 + height, x0 : x0 + width]


def create_dots(params, height, width):
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    spacing = max(6, params.get("spacing", 12))
    radius = max(1, params.get("radius", 1))
    color = params.get("color", (0, 0, 0))
    is_random = params.get("random_stipple", False)

    if is_random:
        density = params.get("density", 0.005)
        num_dots = int(width * height * density)
        rng = np.random.RandomState(params.get("seed", 42))
        xs = rng.randint(0, width, size=num_dots)
        ys = rng.randint(0, height, size=num_dots)
        for x, y in zip(xs, ys):
            cv2.circle(img, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)
    else:
        shift_row = params.get("hex_shift", False)
        for r_idx, y in enumerate(range(spacing // 2, height, spacing)):
            x_offset = (spacing // 2) if (shift_row and r_idx % 2 == 1) else 0
            for x in range(x_offset + spacing // 2, width, spacing):
                cv2.circle(img, (x, y), radius, color, -1, cv2.LINE_AA)
    return img


def create_hexagons(params, height, width):
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    r = max(6, params.get("radius", 12))
    line_width = max(1, params.get("line_width", 1))
    color = params.get("color", (0, 0, 0))

    diag = int(np.hypot(width, height)) + 100
    pattern_size = diag * 2
    temp = np.full((pattern_size, pattern_size, 3), 255, dtype=np.uint8)

    h_step = int(np.sqrt(3) * r)
    v_step = int(1.5 * r)
    angles = np.array([np.pi / 6 + i * np.pi / 3 for i in range(6)])
    unit_hex_x = (r * np.cos(angles)).astype(np.int32)
    unit_hex_y = (r * np.sin(angles)).astype(np.int32)

    row = 0
    for y in range(r, pattern_size + r, v_step if v_step > 0 else 10):
        x_shift = (h_step // 2) if (row % 2 == 1) else 0
        for x in range(x_shift, pattern_size + h_step, h_step if h_step > 0 else 10):
            pts = np.stack([unit_hex_x + x, unit_hex_y + y], axis=-1)
            cv2.polylines(temp, [pts], isClosed=True, color=color, thickness=line_width, lineType=cv2.LINE_AA)
        row += 1

    angle_deg = params.get("angle", 0)
    if angle_deg != 0:
        rot_mat = cv2.getRotationMatrix2D(
            (pattern_size // 2, pattern_size // 2), angle_deg, 1.0
        )
        temp = cv2.warpAffine(
            temp,
            rot_mat,
            (pattern_size, pattern_size),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )

    cx, cy = pattern_size // 2, pattern_size // 2
    x0, y0 = cx - width // 2, cy - height // 2
    return temp[y0 : y0 + height, x0 : x0 + width]


def create_circles(params, height, width):
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    spacing = max(8, params.get("spacing", 16))
    radius = max(2, params.get("radius", 5))
    line_width = max(1, params.get("line_width", 1))
    color = params.get("color", (0, 0, 0))
    concentric = params.get("concentric", False)

    for y in range(spacing // 2, height, spacing):
        for x in range(spacing // 2, width, spacing):
            cv2.circle(img, (x, y), radius, color, line_width, cv2.LINE_AA)
            if concentric and radius > 3:
                cv2.circle(
                    img, (x, y), radius // 2, color, line_width, cv2.LINE_AA
                )
    return img


def create_triangles(params, height, width):
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    size = max(8, params.get("size", 16))
    spacing = max(size + 4, params.get("spacing", 20))
    line_width = max(1, params.get("line_width", 1))
    color = params.get("color", (0, 0, 0))

    h = int(size * np.sqrt(3) / 2)
    for y in range(spacing // 2, height, spacing):
        for x in range(spacing // 2, width, spacing):
            pts = np.array(
                [[x, y - h // 2], [x - size // 2, y + h // 2], [x + size // 2, y + h // 2]],
                dtype=np.int32,
            )
            cv2.polylines(
                img, [pts], isClosed=True, color=color, thickness=line_width, lineType=cv2.LINE_AA
            )
    return img


def create_plus_cross(params, height, width):
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    size = max(4, params.get("size", 8))
    spacing = max(size + 4, params.get("spacing", 16))
    line_width = max(1, params.get("line_width", 1))
    color = params.get("color", (0, 0, 0))
    is_x = params.get("diagonal", False)

    for y in range(spacing // 2, height, spacing):
        for x in range(spacing // 2, width, spacing):
            arm = size // 2
            if is_x:
                cv2.line(
                    img,
                    (x - arm, y - arm),
                    (x + arm, y + arm),
                    color,
                    line_width,
                    cv2.LINE_AA,
                )
                cv2.line(
                    img,
                    (x - arm, y + arm),
                    (x + arm, y - arm),
                    color,
                    line_width,
                    cv2.LINE_AA,
                )
            else:
                cv2.line(
                    img,
                    (x - arm, y),
                    (x + arm, y),
                    color,
                    line_width,
                    cv2.LINE_AA,
                )
                cv2.line(
                    img,
                    (x, y - arm),
                    (x, y + arm),
                    color,
                    line_width,
                    cv2.LINE_AA,
                )
    return img


def create_concrete(params, height, width):
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    num_marks = params.get("num_marks", int(width * height * 0.003))
    color = params.get("color", (60, 60, 60))
    seed = params.get("seed", 123)
    rng = np.random.RandomState(seed)

    xs = rng.randint(0, width, size=num_marks)
    ys = rng.randint(0, height, size=num_marks)

    for x, y in zip(xs, ys):
        mark_type = rng.randint(0, 3)
        if mark_type == 0:
            r = rng.randint(1, 3)
            cv2.circle(img, (int(x), int(y)), r, color, -1, cv2.LINE_AA)
        elif mark_type == 1:
            dx = rng.randint(-3, 4)
            dy = rng.randint(-3, 4)
            cv2.line(
                img,
                (int(x), int(y)),
                (int(x + dx), int(y + dy)),
                color,
                1,
                cv2.LINE_AA,
            )
        else:
            s = rng.randint(2, 5)
            pts = np.array(
                [[x, y], [x + s, y + rng.randint(-2, 3)], [x + s // 2, y + s]],
                dtype=np.int32,
            )
            cv2.polylines(
                img, [pts], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA
            )
    return img


PATTERN_DISPATCH = {
    "parallel_lines": create_parallel_lines,
    "dashed_lines": create_dashed_lines,
    "crosshatch": create_crosshatch,
    "brick": create_brick,
    "dots": create_dots,
    "hexagons": create_hexagons,
    "circles": create_circles,
    "triangles": create_triangles,
    "plus_cross": create_plus_cross,
    "concrete": create_concrete,
    "solid_fill": create_solid_fill,
}


def render_pattern(family, params, height=2048, width=2048):
    func = PATTERN_DISPATCH.get(family, create_parallel_lines)
    return func(params, height, width)
