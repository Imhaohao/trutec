"""Experiment runner and evaluation suite for drawing-level pattern competition matcher.

Evaluates standalone competition variants and their fusion with existing HGB baseline
across all 55 development queries using the exact sparse evaluation harness.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research_v5.competition.matcher import (  # noqa: E402
    CompetitionMatcher,
    MatcherConfig,
    _normalize_vectors,
    discover_prototypes,
    extract_texture_grid,
    sample_score_coordinates,
)
from research_v5.devharness import (  # noqa: E402
    HGB_CACHE,
    cached_scores,
    development_queries,
    load_rgb,
    report,
    request_view,
    _rows,
    DEVELOPMENT_MANIFEST,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "results"
FINDINGS_PATH = Path(__file__).resolve().parent / "findings.txt"


def evaluate_matcher_config(
    matcher: CompetitionMatcher,
    name: str,
    feature_cache: dict[str, tuple],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Score all 55 development queries for a given matcher configuration."""
    queries = development_queries()
    manifest_rows = {row["id"]: row for row in _rows(DEVELOPMENT_MANIFEST)}

    drawing_contexts: dict[str, tuple] = {}
    for img_name, (feats, ink_grid, ys, xs) in feature_cache.items():
        gh, gw, ch = feats.shape
        flat_feats = feats.reshape(-1, ch)
        flat_ink = ink_grid.reshape(-1)
        protos, mean, std, non_blank = discover_prototypes(flat_feats, flat_ink, matcher.config)
        norm_all = _normalize_vectors(flat_feats, mean, std)
        drawing_contexts[img_name] = (feats, ink_grid, ys, xs, protos, mean, std, norm_all)

    scores: dict[str, np.ndarray] = {}
    for q in queries:
        drawing_ctx = drawing_contexts[q.image]
        score_grid = matcher.score_query(request_view(manifest_rows[q.id]), drawing_ctx)
        scores[q.id] = sample_score_coordinates(score_grid, q.y, q.x, matcher.config.stride)

    json_path = OUTPUT_DIR / f"{name}.json"
    result = report(name, scores, json_path)
    return scores, result


def evaluate_fusion(
    comp_scores: dict[str, np.ndarray],
    hgb_scores: dict[str, np.ndarray],
    fusion_name: str,
    weight_comp: float = 0.5,
    mode: str = "linear",
    gate_threshold: float = 0.25,
) -> dict[str, Any]:
    """Combine competition scores with existing HGB baseline."""
    fused: dict[str, np.ndarray] = {}
    for q_id, s_comp in comp_scores.items():
        s_hgb = hgb_scores[q_id]
        if mode == "linear":
            w = weight_comp
            fused[q_id] = np.clip(w * s_comp + (1.0 - w) * s_hgb, 0.0, 1.0)
        elif mode == "geometric":
            fused[q_id] = np.clip(np.sqrt(s_comp * s_hgb), 0.0, 1.0)
        elif mode == "harmonic":
            fused[q_id] = (2.0 * s_comp * s_hgb) / (s_comp + s_hgb + 1e-4)
        elif mode == "gated":
            th = gate_threshold
            gate = np.where(s_comp > th, 1.0, s_comp / th)
            fused[q_id] = s_hgb * gate
        elif mode == "gated_power2":
            th = gate_threshold
            gate = np.where(s_comp > th, 1.0, (s_comp / th) ** 2)
            fused[q_id] = s_hgb * gate
        else:
            raise ValueError(f"Unknown fusion mode: {mode}")

    json_path = OUTPUT_DIR / f"{fusion_name}.json"
    return report(fusion_name, fused, json_path)


def format_summary_line(name: str, res: dict[str, Any]) -> str:
    """Format one-line summary of results for findings.txt."""
    sel = res["selections"]
    best_iou = sel["best_macro_iou"]
    best_rec = sel["best_precision_at_recall_ge_0_98"]

    iou_val = best_iou["document_macro_iou"] if best_iou else 0.0
    iou_rec = best_iou["global_recall"] if best_iou else 0.0
    iou_prec = best_iou["global_precision"] if best_iou else 0.0
    iou_th = best_iou["threshold"] if best_iou else 0.0

    rec_prec = best_rec["global_precision"] if best_rec else 0.0
    rec_iou = best_rec["document_macro_iou"] if best_rec else 0.0
    rec_th = best_rec["threshold"] if best_rec else 0.0

    return (
        f"{name:<35} | "
        f"Macro IoU: {iou_val:.4f} (th={iou_th:.2f}, R={iou_rec:.4f}, P={iou_prec:.4f}) | "
        f"Prec@R>=0.98: {rec_prec:.4f} (IoU={rec_iou:.4f}, th={rec_th:.2f})"
    )


def run_experiment_suite() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    hgb_scores = cached_scores(HGB_CACHE)
    hgb_report = report("baseline_hgb", hgb_scores, OUTPUT_DIR / "baseline_hgb.json")

    print("Baseline HGB:")
    print(format_summary_line("baseline_hgb", hgb_report))

    findings_lines: list[str] = [
        "HATCH MATCHING RESEARCH - DRAWING-LEVEL PATTERN COMPETITION",
        "===========================================================",
        f"Baseline HGB: {format_summary_line('baseline_hgb', hgb_report)}",
        "",
        "STANDALONE COMPETITION VARIANTS",
        "--------------------------------",
    ]

    print("\nPre-extracting rich texture feature grids for all 12 unique drawing images (stride=8)...")
    t0 = time.time()
    feature_cache: dict[str, tuple] = {}
    queries = development_queries()
    for q in queries:
        if q.image not in feature_cache:
            rgb = load_rgb(q.image)
            feature_cache[q.image] = extract_texture_grid(rgb, stride=8)
    print(f"Feature extraction done in {time.time() - t0:.2f}s")

    # Comprehensive Experiment Configurations
    configs: list[tuple[str, MatcherConfig]] = [
        ("comp_k8_base", MatcherConfig(n_clusters=8)),
        ("comp_k16_base", MatcherConfig(n_clusters=16)),
        ("comp_k20_base", MatcherConfig(n_clusters=20)),
        ("comp_k24_base", MatcherConfig(n_clusters=24)),
        ("comp_k32_base", MatcherConfig(n_clusters=32)),
        ("comp_k20_blur7", MatcherConfig(n_clusters=20, spatial_blur_radius=7, spatial_blur_sigma=1.8)),
        ("comp_k20_blur9", MatcherConfig(n_clusters=20, spatial_blur_radius=9, spatial_blur_sigma=2.2)),
        ("comp_k20_blur11", MatcherConfig(n_clusters=20, spatial_blur_radius=11, spatial_blur_sigma=2.8)),
        ("comp_k20_blur13", MatcherConfig(n_clusters=20, spatial_blur_radius=13, spatial_blur_sigma=3.2)),
        ("comp_k20_sigmoid_g8", MatcherConfig(n_clusters=20, score_mapping_mode="sigmoid", sigmoid_gain=8.0)),
        ("comp_k20_sigmoid_g12", MatcherConfig(n_clusters=20, score_mapping_mode="sigmoid", sigmoid_gain=12.0)),
        ("comp_k20_soft_t4", MatcherConfig(n_clusters=20, use_softmax_margin=True, softmax_temperature=4.0)),
        ("comp_k20_tight", MatcherConfig(n_clusters=20, competitor_margin_threshold=0.08)),
        ("comp_k20_wide", MatcherConfig(n_clusters=20, competitor_margin_threshold=0.20)),
    ]

    evaluated_comp_scores: dict[str, dict[str, np.ndarray]] = {}

    for name, cfg in configs:
        t_start = time.time()
        matcher = CompetitionMatcher(cfg)
        scores, res = evaluate_matcher_config(matcher, name, feature_cache)
        evaluated_comp_scores[name] = scores
        line = format_summary_line(name, res)
        print(f"[{time.time() - t_start:.2f}s] {line}")
        findings_lines.append(line)

    findings_lines.extend([
        "",
        "FUSION WITH HGB BASELINE",
        "------------------------",
    ])

    best_comps = ["comp_k20_blur9", "comp_k20_blur11", "comp_k20_blur13", "comp_k20_base"]

    for comp_name in best_comps:
        if comp_name not in evaluated_comp_scores:
            continue
        c_scores = evaluated_comp_scores[comp_name]
        print(f"\n--- Fusion sweeps with {comp_name} ---")

        fusion_variants = [
            (f"fuse_{comp_name}_w0.20", c_scores, 0.20, "linear", 0.25),
            (f"fuse_{comp_name}_w0.35", c_scores, 0.35, "linear", 0.25),
            (f"fuse_{comp_name}_w0.45", c_scores, 0.45, "linear", 0.25),
            (f"fuse_{comp_name}_geom", c_scores, 0.50, "geometric", 0.25),
            (f"fuse_{comp_name}_harmonic", c_scores, 0.50, "harmonic", 0.25),
            (f"fuse_{comp_name}_gate_th0.20", c_scores, 0.50, "gated", 0.20),
            (f"fuse_{comp_name}_gate_th0.25", c_scores, 0.50, "gated", 0.25),
            (f"fuse_{comp_name}_gate_th0.30", c_scores, 0.50, "gated", 0.30),
            (f"fuse_{comp_name}_gate_th0.35", c_scores, 0.50, "gated", 0.35),
            (f"fuse_{comp_name}_gate_pow2_th0.25", c_scores, 0.50, "gated_power2", 0.25),
            (f"fuse_{comp_name}_gate_pow2_th0.30", c_scores, 0.50, "gated_power2", 0.30),
        ]

        for fname, scores_c, w, mode, g_th in fusion_variants:
            fres = evaluate_fusion(
                scores_c,
                hgb_scores,
                fname,
                weight_comp=w,
                mode=mode,
                gate_threshold=g_th,
            )
            fline = format_summary_line(fname, fres)
            print(fline)
            findings_lines.append(fline)

    FINDINGS_PATH.write_text("\n".join(findings_lines) + "\n", encoding="utf-8")
    print(f"\nWrote findings to {FINDINGS_PATH}")


def main() -> None:
    run_experiment_suite()


if __name__ == "__main__":
    main()
