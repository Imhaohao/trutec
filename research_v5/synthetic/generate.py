"""Entrypoint script to generate 600 synthetic sheets, manifest, stats, and previews."""

import json
import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np

from research_v5.synthetic.assembler import generate_single_sheet


def generate_all_sheets(
    num_sheets=600, seed=42, output_dir="research_v5/synthetic"
):
    out_path = Path(output_dir)
    data_dir = out_path / "data"
    img_dir = data_dir / "images"
    mask_dir = data_dir / "masks"
    prev_dir = out_path / "previews"

    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    prev_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(seed)
    all_examples = []
    stats_records = []

    start_time = time.time()
    preview_count = 0
    sheet_idx = 0

    while sheet_idx < num_sheets:
        seed_i = int(rng.randint(0, 1000000000))
        sheet_rng = np.random.RandomState(seed_i)

        res = generate_single_sheet(sheet_idx, sheet_rng, data_dir)
        if res is None:
            continue

        examples, sheet_img, pos_mask, stat = res
        all_examples.extend(examples)
        stats_records.append(stat)

        # Generate preview for first 12 sheets
        if preview_count < 12:
            prev_img = sheet_img.copy()
            # Tint positive area green
            overlay = prev_img.copy()
            overlay[pos_mask] = np.clip(
                overlay[pos_mask].astype(np.int32) * 0.5 + np.array([0, 180, 0]) * 0.5,
                0,
                255,
            ).astype(np.uint8)
            cv2.addWeighted(overlay, 0.6, prev_img, 0.4, 0, prev_img)

            # Draw query box in red with thick border
            for ex in examples:
                qx0, qy0, qx1, qy1 = ex["query_box"]
                cv2.rectangle(
                    prev_img, (qx0, qy0), (qx1, qy1), (0, 0, 255), 4, cv2.LINE_AA
                )
                cv2.putText(
                    prev_img,
                    f"QUERY: {ex['id']}",
                    (qx0, max(20, qy0 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            prev_path = prev_dir / f"preview_{preview_count:02d}.png"
            cv2.imwrite(str(prev_path), cv2.cvtColor(prev_img, cv2.COLOR_RGB2BGR))
            preview_count += 1

        sheet_idx += 1
        if sheet_idx % 50 == 0:
            elapsed = time.time() - start_time
            print(
                f"Generated {sheet_idx}/{num_sheets} sheets ({len(all_examples)} examples) in {elapsed:.1f}s"
            )

    elapsed_total = time.time() - start_time

    # Save manifest
    manifest_data = {
        "schema": "hatch-matching-challenge/v1",
        "examples": all_examples,
    }
    manifest_file = data_dir / "manifest.json"
    manifest_file.write_text(
        json.dumps(manifest_data, indent=2), encoding="utf-8"
    )

    # Compute aggregate stats
    pattern_counts = {}
    color_counts = {}
    trap_count = 0
    q_widths = []
    q_heights = []
    pos_fracs = []

    for s in stats_records:
        pattern_counts[s["pattern"]] = pattern_counts.get(s["pattern"], 0) + 1
        color_counts[s["color"]] = color_counts.get(s["color"], 0) + 1
        if s["has_trap"]:
            trap_count += 1
        q_widths.append(s["query_size"][0])
        q_heights.append(s["query_size"][1])
        pos_fracs.append(s["pos_frac"])

    stats_summary = {
        "sheets": num_sheets,
        "examples": len(all_examples),
        "generation_time_seconds": round(elapsed_total, 2),
        "patterns": pattern_counts,
        "colors": color_counts,
        "traps": {
            "total_with_traps": trap_count,
            "trap_rate": round(trap_count / num_sheets, 4),
        },
        "query_sizes": {
            "min_width": int(min(q_widths)),
            "max_width": int(max(q_widths)),
            "mean_width": round(float(np.mean(q_widths)), 1),
            "min_height": int(min(q_heights)),
            "max_height": int(max(q_heights)),
            "mean_height": round(float(np.mean(q_heights)), 1),
        },
        "positive_fraction": {
            "min": round(float(min(pos_fracs)), 4),
            "max": round(float(max(pos_fracs)), 4),
            "mean": round(float(np.mean(pos_fracs)), 4),
        },
    }

    stats_file = out_path / "stats.json"
    stats_file.write_text(json.dumps(stats_summary, indent=2), encoding="utf-8")
    print(f"Stats written to {stats_file}")

    return manifest_file, len(all_examples), num_sheets, seed


def main():
    manifest_file, n_ex, n_sheets, seed = generate_all_sheets(
        num_sheets=600, seed=42
    )
    print(
        f"Done generating {n_sheets} sheets ({n_ex} examples). Manifest: {manifest_file}"
    )


if __name__ == "__main__":
    main()
