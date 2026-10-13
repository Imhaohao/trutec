"""Sheet assembly pipeline producing images, masks, and manifest metadata."""

import json
import os
import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

from research_v5.synthetic.elements import draw_wall_gaps, draw_walls
from research_v5.synthetic.generator import (
    PATTERN_FAMILIES,
    add_sheet_overlays,
    generate_known_mask,
    make_near_twin_params,
    sample_pattern_params,
    sample_query_box_inside_mask,
)
from research_v5.synthetic.geometry import generate_rooms
from research_v5.synthetic.patterns import render_pattern


def build_room_masks(polygons, height=2048, width=2048):
    """Render individual binary masks for each polygon."""
    masks = []
    for poly in polygons:
        m = np.zeros((height, width), dtype=np.uint8)
        pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(m, [pts], 255)
        masks.append(m > 0)
    return masks


def assign_patterns_to_rooms(num_rooms, rng):
    """Assign hatch patterns and trap patterns across rooms."""
    num_hatched = rng.randint(3, min(9, num_rooms + 1))
    hatched_indices = rng.choice(num_rooms, size=num_hatched, replace=False)

    # Pick 1 to 3 patterns that will serve as query targets
    target_pattern = sample_pattern_params(rng.choice(PATTERN_FAMILIES), rng)
    assignments = {}

    # Assign target pattern to 1 or 2 rooms
    target_rooms = [hatched_indices[0]]
    if len(hatched_indices) > 3 and rng.rand() < 0.35:
        target_rooms.append(hatched_indices[1])

    for r_idx in target_rooms:
        assignments[r_idx] = ("target", target_pattern)

    # Assign near twin / color trap on >= 50% of sheets
    has_trap = rng.rand() < 0.85
    trap_room = None
    remaining = [i for i in hatched_indices if i not in target_rooms]

    if has_trap and remaining:
        trap_room = remaining.pop(0)
        twin_params = make_near_twin_params(target_pattern, rng)
        assignments[trap_room] = ("trap", twin_params)

    # Fill other hatched rooms with distinct random patterns
    for r_idx in remaining:
        other_fam = rng.choice(PATTERN_FAMILIES)
        other_params = sample_pattern_params(other_fam, rng)
        assignments[r_idx] = ("other", other_params)

    return assignments, target_rooms, trap_room


def assemble_sheet_image(polygons, room_masks, assignments, rng, width=2048, height=2048):
    """Assemble background, hatch patterns, walls, door gaps, and overlays."""
    sheet_img = np.full((height, width, 3), 255, dtype=np.uint8)

    # Render each room's hatch pattern inside its polygon mask
    for r_idx, (_, params) in assignments.items():
        pattern_img = render_pattern(params["family"], params, height=height, width=width)
        mask = room_masks[r_idx]
        sheet_img[mask] = pattern_img[mask]

    # Draw walls
    wall_thick = rng.randint(2, 6)
    draw_walls(sheet_img, polygons, wall_thickness=wall_thick)

    # Draw door gaps
    gaps = []
    for p_idx, poly in enumerate(polygons):
        if rng.rand() < 0.4:
            e = rng.randint(0, len(poly))
            gaps.append((p_idx, e, rng.uniform(0.3, 0.7), rng.randint(12, 36)))
    if gaps:
        draw_wall_gaps(sheet_img, polygons, gaps)

    return sheet_img


def generate_single_sheet(sheet_idx, rng, data_dir):
    """Generate one sheet with 1 to 3 queries, saving image, masks, and metadata."""
    width, height = 2048, 2048
    polygons = generate_rooms(rng, width=width, height=height)
    room_masks = build_room_masks(polygons, height=height, width=width)

    assignments, target_rooms, trap_room = assign_patterns_to_rooms(len(polygons), rng)
    target_params = assignments[target_rooms[0]][1]

    # Combine positive mask (all rooms sharing the exact target pattern)
    positive_mask = np.zeros((height, width), dtype=bool)
    for r_idx in target_rooms:
        positive_mask |= room_masks[r_idx]

    # Pick query box inside one of the target rooms
    target_r_idx = rng.choice(target_rooms)
    query_box = sample_query_box_inside_mask(room_masks[target_r_idx], rng)
    if query_box is None:
        # Fallback query box
        return None

    # Check if there are other candidate queries
    other_rooms = [r for r, (role, _) in assignments.items() if role == "other"]
    extra_rooms = []
    if other_rooms and rng.rand() < 0.4:
        extra_r = rng.choice(other_rooms)
        extra_qbox = sample_query_box_inside_mask(room_masks[extra_r], rng)
        if extra_qbox is not None:
            extra_rooms.append((extra_r, extra_qbox))

    # Known mask with 0-30% unknown band, ensuring all positive masks are inside known
    all_pos_masks = [positive_mask] + [room_masks[er[0]] for er in extra_rooms]
    known_mask = generate_known_mask(height, width, rng, positive_masks=all_pos_masks)

    # Assemble sheet image
    sheet_img = assemble_sheet_image(polygons, room_masks, assignments, rng, width=width, height=height)

    # Add overlays ON TOP of everything (overlays on positive areas are part of the image, mask remains positive)
    add_sheet_overlays(sheet_img, polygons, rng, query_box=query_box)

    # Generate document and query IDs
    doc_id = f"syn-doc-{sheet_idx:04d}"
    ex_id = f"syn-q-{sheet_idx:04d}-0"

    # Save image and masks
    img_rel = f"images/{doc_id}.png"
    pos_mask_rel = f"masks/{ex_id}_positive.png"
    known_mask_rel = f"masks/{ex_id}_known.png"

    img_path = data_dir / img_rel
    pos_mask_path = data_dir / pos_mask_rel
    known_mask_path = data_dir / known_mask_rel

    Image.fromarray(sheet_img).save(img_path, format="PNG")
    Image.fromarray((positive_mask.astype(np.uint8) * 255), mode="L").save(pos_mask_path, format="PNG")
    Image.fromarray((known_mask.astype(np.uint8) * 255), mode="L").save(known_mask_path, format="PNG")

    examples = [
        {
            "id": ex_id,
            "document_id": doc_id,
            "kind": "generated_synthetic",
            "image": img_rel,
            "width": width,
            "height": height,
            "query_box": query_box,
            "context_boxes": [],
            "labels": {
                "positive_mask": pos_mask_rel,
                "known_mask": known_mask_rel,
            },
        }
    ]

    # Additional query if there are extra rooms sampled
    for extra_r, extra_qbox in extra_rooms:
        extra_ex_id = f"syn-q-{sheet_idx:04d}-1"
        extra_pos_mask = room_masks[extra_r]
        extra_pos_rel = f"masks/{extra_ex_id}_positive.png"
        extra_known_rel = f"masks/{extra_ex_id}_known.png"

        Image.fromarray((extra_pos_mask.astype(np.uint8) * 255), mode="L").save(
            data_dir / extra_pos_rel, format="PNG"
        )
        Image.fromarray((known_mask.astype(np.uint8) * 255), mode="L").save(
            data_dir / extra_known_rel, format="PNG"
        )

        examples.append(
            {
                "id": extra_ex_id,
                "document_id": doc_id,
                "kind": "generated_synthetic",
                "image": img_rel,
                "width": width,
                "height": height,
                "query_box": extra_qbox,
                "context_boxes": [],
                "labels": {
                    "positive_mask": extra_pos_rel,
                    "known_mask": extra_known_rel,
                },
            }
        )

    stats = {
        "pattern": target_params["family"],
        "color": target_params["color_name"],
        "has_trap": trap_room is not None,
        "query_size": (query_box[2] - query_box[0], query_box[3] - query_box[1]),
        "pos_frac": float(positive_mask.sum()) / (width * height),
    }

    return examples, sheet_img, positive_mask, stats
