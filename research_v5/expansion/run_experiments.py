"""Run anchor-and-grow and query expansion experiments on development split."""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research_v5"))
sys.path.insert(0, str(ROOT / "research_v5/expansion"))

from color.color_features import (  # noqa: E402
    apply_gate,
    build_image_color_cache,
    compute_sample_mismatch,
    extract_robust_query_signature,
)
import devharness  # noqa: E402
import hatch_texture  # noqa: E402
from inference_texture import predict_probability_grid, verify_model_metadata  # noqa: E402
from pipeline import (  # noqa: E402
    compute_anchor_density,
    expand_query_descriptor,
    fuse_prob_and_density,
    grow_connected_components,
    morphological_closing_grid,
    morphological_dilation_grid,
    sample_grid_to_coordinates,
)
from texture_learning import load_texture_model  # noqa: E402

HERE = Path(__file__).resolve().parent
FINDINGS = HERE / "findings.txt"


def get_model_path() -> Path:
    candidates = [
        ROOT / "models/texture-hgb-crop-v1.joblib",
        ROOT / "models/models/texture-hgb-crop-v1.joblib",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("Could not find texture-hgb-crop-v1.joblib")


def init_findings() -> None:
    FINDINGS.write_text(
        "Anchor-and-Grow with Test-Time Query Expansion on 55-query development split.\n"
        "Key design: Positive labels include text and dimension lines printed on top of hatch.\n"
        "Anchor selection isolates clean hatch; query expansion sharpens descriptor;\n"
        "growth fills interior gaps and letters while suppressing isolated text.\n\n",
        encoding="utf-8",
    )


def append_finding(name: str, settings_desc: str, result: dict) -> None:
    selections = result["selections"]
    best_iou = selections["best_macro_iou"]
    p_at_r98 = selections["best_precision_at_recall_ge_0_98"]
    iou_at_r98 = selections["best_macro_iou_at_recall_ge_0_98"]

    def fmt_point(pt: dict | None) -> str:
        if not pt:
            return "none"
        return (
            f"iou={pt['document_macro_iou']:.6f} "
            f"precision={pt['global_precision']:.6f} "
            f"recall={pt['global_recall']:.6f} "
            f"threshold={pt['threshold']:.2f} "
            f"fpr={pt['reviewed_negative_fpr']:.6f}"
        )

    text = (
        f"variant: {name}\n"
        f"settings: {settings_desc}\n"
        f"best_macro_iou: {fmt_point(best_iou)}\n"
        f"best_precision_at_recall_ge_0.98: {fmt_point(p_at_r98)}\n"
        f"best_macro_iou_at_recall_ge_0.98: {fmt_point(iou_at_r98)}\n"
        f"all_goals_met: {selections['all_goals_met']}\n\n"
    )
    with FINDINGS.open("a", encoding="utf-8") as handle:
        handle.write(text)


def process_image_queries(
    model: object,
    gray: np.ndarray,
    rgb: np.ndarray,
    queries: list,
    stride: int,
    source: str,
) -> dict[str, dict[str, np.ndarray]]:
    """Compute all variant scores for queries belonging to a single drawing."""
    height, width = gray.shape
    features = hatch_texture.texture_features(gray, stride=stride)
    color_cache = build_image_color_cache(rgb, window_sizes=(16, 32))

    local_scores: dict[str, dict[str, np.ndarray]] = defaultdict(dict)

    for query in queries:
        query_box = query.query_box
        y_coords = query.y
        x_coords = query.x

        # Initial descriptor & probability grid (Iteration 0)
        q_desc_0 = hatch_texture.descriptor_for_query(
            gray, features, query_box, stride, source
        )
        p_grid_0 = predict_probability_grid(model, features, q_desc_0)

        # Iteration 1 Query Expansion
        q_desc_1 = expand_query_descriptor(
            features, p_grid_0, query_box, (height, width), stride=stride, anchor_quantile=0.50
        )
        p_grid_1 = predict_probability_grid(model, features, q_desc_1)

        # Iteration 2 Query Expansion
        q_desc_2 = expand_query_descriptor(
            features, p_grid_1, query_box, (height, width), stride=stride, anchor_quantile=0.50
        )
        p_grid_2 = predict_probability_grid(model, features, q_desc_2)

        # Iteration 3 Query Expansion
        q_desc_3 = expand_query_descriptor(
            features, p_grid_2, query_box, (height, width), stride=stride, anchor_quantile=0.50
        )
        p_grid_3 = predict_probability_grid(model, features, q_desc_3)

        # Choose expanded grid for growth (Iter 1 / Iter 2)
        p_exp = p_grid_1

        # Query crop size
        qw = query_box[2] - query_box[0]
        qh = query_box[3] - query_box[1]
        q_min_dim = min(qw, qh)
        scaled_window = max(24, q_min_dim // 2)

        # Anchors & Growth Maps
        density_w24 = compute_anchor_density(p_exp, anchor_threshold=0.50, window_pixels=24, stride=stride)
        density_w48 = compute_anchor_density(p_exp, anchor_threshold=0.50, window_pixels=48, stride=stride)
        density_w72 = compute_anchor_density(p_exp, anchor_threshold=0.50, window_pixels=72, stride=stride)
        density_scaled = compute_anchor_density(
            p_exp, anchor_threshold=0.50, window_pixels=scaled_window, stride=stride
        )

        morph_close_w48 = morphological_closing_grid(p_exp, window_pixels=48, stride=stride)
        morph_close_w72 = morphological_closing_grid(p_exp, window_pixels=72, stride=stride)

        morph_dilate_w48 = morphological_dilation_grid(p_exp, window_pixels=48, stride=stride)
        morph_dilate_w72 = morphological_dilation_grid(p_exp, window_pixels=72, stride=stride)

        connected_w48 = grow_connected_components(
            p_exp, density_w48, density_threshold=0.20, min_area_cells=15
        )

        fused_p_density = fuse_prob_and_density(p_exp, density_w48, alpha=0.5)

        # Color Gating
        query_sig = extract_robust_query_signature(rgb, query_box)
        mismatch = compute_sample_mismatch(query_sig, color_cache[16], y_coords, x_coords)

        # Grid sampling for each variant
        grid_maps = {
            "baseline_hgb": p_grid_0,
            "expansion_iter1": p_grid_1,
            "expansion_iter2": p_grid_2,
            "expansion_iter3": p_grid_3,
            "anchor_density_w24": density_w24,
            "anchor_density_w48": density_w48,
            "anchor_density_w72": density_w72,
            "anchor_density_query_scaled": density_scaled,
            "morph_close_w48": morph_close_w48,
            "morph_close_w72": morph_close_w72,
            "grow_max_dilate_w48": morph_dilate_w48,
            "grow_max_dilate_w72": morph_dilate_w72,
            "grow_connected_density": connected_w48,
            "grow_density_prob_fusion": fused_p_density,
        }

        for var_name, grid in grid_maps.items():
            sampled = sample_grid_to_coordinates(
                grid, height, width, stride, y_coords, x_coords
            )
            local_scores[var_name][query.id] = sampled

        # Add color-gated variant
        iter1_sampled = local_scores["expansion_iter1"][query.id]
        local_scores["expansion_color_gate"][query.id] = apply_gate(
            iter1_sampled, mismatch, gate_type="linear", strength=0.6
        )

    del features
    return local_scores


def get_variant_descriptions() -> dict[str, str]:
    return {
        "baseline_hgb": "Exact crop-descriptor HistGradientBoosting full-drawing probability grid.",
        "expansion_iter1": (
            "Query expansion 1 iteration: re-estimate query descriptor as median of "
            "top-50% anchor cells in query crop, then rescore full drawing."
        ),
        "expansion_iter2": (
            "Query expansion 2 iterations: iterative self-training refinement of query descriptor."
        ),
        "expansion_iter3": (
            "Query expansion 3 iterations: 3-step test-time descriptor refinement."
        ),
        "expansion_color_gate": (
            "Query expansion (iter 1) gated by neighborhood chroma/hue mismatch (window 16, strength 0.6)."
        ),
        "anchor_density_w24": (
            "Anchor density (threshold 0.50) in a 24px box window over expanded grid."
        ),
        "anchor_density_w48": (
            "Anchor density (threshold 0.50) in a 48px box window over expanded grid."
        ),
        "anchor_density_w72": (
            "Anchor density (threshold 0.50) in a 72px box window over expanded grid."
        ),
        "anchor_density_query_scaled": (
            "Anchor density with window scaled to query crop size (min_dim / 2)."
        ),
        "morph_close_w48": (
            "Grayscale morphological closing (48px window) on expanded probability grid to bridge text gaps."
        ),
        "morph_close_w72": (
            "Grayscale morphological closing (72px window) on expanded probability grid."
        ),
        "grow_max_dilate_w48": (
            "Grayscale dilation / max filter (48px window) allowing pixels to inherit nearby anchor scores."
        ),
        "grow_max_dilate_w72": (
            "Grayscale dilation / max filter (72px window) over expanded probability grid."
        ),
        "grow_connected_density": (
            "Connected-component hole-filled growing of anchor density (threshold 0.20, min area 15 cells)."
        ),
        "grow_density_prob_fusion": (
            "Linear fusion (alpha 0.5) of expanded probability grid and 48px anchor density map."
        ),
    }


def main() -> None:
    init_findings()
    model_path = get_model_path()
    model, metadata = load_texture_model(model_path)
    verify_model_metadata(metadata)

    stride = int(metadata.get("stride", 4))
    source = hatch_texture.descriptor_source_from_metadata(metadata)

    all_queries = devharness.development_queries()
    grouped_queries: dict[str, list] = defaultdict(list)
    for q in all_queries:
        grouped_queries[q.image].append(q)

    total_scores: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    descriptions = get_variant_descriptions()

    print(f"Loaded model from {model_path}. Processing {len(all_queries)} queries in {len(grouped_queries)} images...")
    start_time = time.perf_counter()

    for img_idx, (image_rel, q_list) in enumerate(grouped_queries.items(), start=1):
        img_start = time.perf_counter()
        gray = devharness.load_gray(image_rel)
        rgb = devharness.load_rgb(image_rel)
        print(f"[{img_idx}/{len(grouped_queries)}] Processing {image_rel} ({gray.shape[1]}x{gray.shape[0]}, {len(q_list)} queries)...")

        img_scores = process_image_queries(model, gray, rgb, q_list, stride, source)
        for var_name, query_dict in img_scores.items():
            total_scores[var_name].update(query_dict)

        print(f"  Done in {time.perf_counter() - img_start:.2f}s")
        del gray, rgb

    print(f"\nAll drawing passes completed in {time.perf_counter() - start_time:.2f}s.")
    print("Evaluating harness sweeps for all variants...\n")

    summary_records = []
    for var_name, scores_dict in total_scores.items():
        desc = descriptions.get(var_name, var_name)
        json_path = HERE / f"{var_name}.json"
        report_res = devharness.report(var_name, scores_dict, json_path)
        append_finding(var_name, desc, report_res)

        selections = report_res["selections"]
        best_iou = selections["best_macro_iou"]
        p_at_r98 = selections["best_precision_at_recall_ge_0_98"]
        iou_at_r98 = selections["best_macro_iou_at_recall_ge_0_98"]

        summary_records.append({
            "name": var_name,
            "best_macro_iou": best_iou["document_macro_iou"] if best_iou else None,
            "best_macro_iou_threshold": best_iou["threshold"] if best_iou else None,
            "precision_at_best_iou": best_iou["global_precision"] if best_iou else None,
            "recall_at_best_iou": best_iou["global_recall"] if best_iou else None,
            "p_at_r98": p_at_r98["global_precision"] if p_at_r98 else None,
            "p_at_r98_threshold": p_at_r98["threshold"] if p_at_r98 else None,
            "iou_at_r98": iou_at_r98["document_macro_iou"] if iou_at_r98 else None,
            "all_goals_met": selections["all_goals_met"],
        })

        best_iou_val = f"{best_iou['document_macro_iou']:.4f}" if best_iou else "N/A"
        p_r98_val = f"{p_at_r98['global_precision']:.4f}" if p_at_r98 else "N/A"
        print(f"{var_name:<30} | Best Macro IoU: {best_iou_val} | P(R>=0.98): {p_r98_val}")

    # Save summary JSON
    summary_path = HERE / "summary.json"
    summary_path.write_text(json.dumps(summary_records, indent=2), encoding="utf-8")
    print(f"\nSaved findings to {FINDINGS} and summary to {summary_path}")


if __name__ == "__main__":
    main()
