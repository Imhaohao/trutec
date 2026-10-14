"""Evaluate query cleaning rules on the development split."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Callable

import bootstrap  # noqa: F401

import joblib
import numpy as np

import devharness
from bootstrap import CACHE, HERE, STACKER, log, require
from cleaners import (
    clean_descriptor_autocorr,
    clean_descriptor_combined,
    clean_descriptor_components,
    clean_descriptor_iqr,
)
from color.color_features import (
    apply_gate,
    build_image_color_cache,
    compute_sample_mismatch,
    extract_robust_query_signature,
)
from evaluate_dev import assert_development_unseen, load_original
from volume import score_drawing_queries

RULES: dict[str, Callable] = {
    "rule_a_iqr": lambda q, rgb, gray: clean_descriptor_iqr(
        gray[q.query_box[1] : q.query_box[3], q.query_box[0] : q.query_box[2]]
    ),
    "rule_b_autocorr": lambda q, rgb, gray: clean_descriptor_autocorr(rgb, gray, q.query_box),
    "rule_c_components": lambda q, rgb, gray: clean_descriptor_components(
        gray[q.query_box[1] : q.query_box[3], q.query_box[0] : q.query_box[2]]
    ),
    "rule_d_combined": lambda q, rgb, gray: clean_descriptor_combined(rgb, gray, q.query_box),
}


def score_rule(rule_name: str, force: bool = False) -> tuple[dict[str, np.ndarray], dict]:
    """Score one query cleaning rule on development split."""
    require(rule_name in RULES, f"unknown rule: {rule_name}")
    desc_fn = RULES[rule_name]

    stage1, metadata = load_original()
    assert_development_unseen(metadata)
    stage2 = joblib.load(STACKER / "cache" / "stage2_model.joblib")["model"]

    queries = devharness.development_queries()
    rule_cache = CACHE / rule_name
    rule_cache.mkdir(parents=True, exist_ok=True)

    scores: dict[str, np.ndarray] = {}
    if not force and _all_cached(queries, rule_cache):
        log(f"loading cached scores for {rule_name}")
        for q in queries:
            scores[q.id] = np.load(rule_cache / f"{q.id}.npy")
    else:
        log(f"computing scores for {rule_name}")
        grouped = defaultdict(list)
        for q in queries:
            grouped[q.image].append(q)

        for image, members in grouped.items():
            spec = {
                "id": members[0].id,
                "image": image,
                "width": members[0].width,
                "height": members[0].height,
            }
            drawing_scores = score_drawing_queries(spec, members, stage1, stage2, desc_fn)
            for q_id, arr in drawing_scores.items():
                scores[q_id] = arr
                np.save(rule_cache / f"{q_id}.npy", arr.astype(np.float32))
            log(f"scored {len(members)} queries on {image}")

    report_path = HERE / f"dev_report_{rule_name}.json"
    rep = devharness.report(rule_name, scores, report_path)
    return scores, rep


def load_v6_scores() -> dict[str, np.ndarray]:
    """Load baseline v6 development scores (stacker with color gate)."""
    queries = devharness.development_queries()
    raw = {q.id: np.load(STACKER / "cache" / "dev_scores" / f"{q.id}.npy") for q in queries}
    grouped = defaultdict(list)
    for q in queries:
        grouped[q.image].append(q)

    gated = {}
    for image, members in grouped.items():
        rgb = devharness.load_rgb(image)
        color_maps = build_image_color_cache(rgb, window_sizes=(16,))[16]
        for q in members:
            sig = extract_robust_query_signature(rgb, q.query_box)
            mismatch = compute_sample_mismatch(sig, color_maps, q.y, q.x)
            gated[q.id] = apply_gate(raw[q.id], mismatch, gate_type="linear", strength=0.6)
    return gated


def query_iou_at_threshold(decision: np.ndarray, positive: np.ndarray) -> float:
    """Compute IoU for a single query."""
    tp = int(np.count_nonzero(decision & positive))
    fp = int(np.count_nonzero(decision & ~positive))
    fn = int(np.count_nonzero(~decision & positive))
    denom = tp + fp + fn
    return float(tp / denom) if denom > 0 else 1.0


def compare_query_ious(
    v6_scores: dict[str, np.ndarray],
    rule_scores: dict[str, np.ndarray],
    threshold: float = 0.14,
) -> list[dict]:
    """Compare per-query IoUs between v6 and rule at given threshold."""
    queries = devharness.development_queries()
    diffs = []
    for q in queries:
        pos = q.positive.astype(bool)
        iou_v6 = query_iou_at_threshold(v6_scores[q.id] >= threshold, pos)
        iou_rule = query_iou_at_threshold(rule_scores[q.id] >= threshold, pos)
        diff = iou_rule - iou_v6
        diffs.append(
            {
                "id": q.id,
                "document_id": q.document_id,
                "iou_v6": iou_v6,
                "iou_rule": iou_rule,
                "diff": diff,
                "abs_diff": abs(diff),
            }
        )
    diffs.sort(key=lambda item: item["abs_diff"], reverse=True)
    return diffs


def _all_cached(queries, cache_dir: Path) -> bool:
    for q in queries:
        if not (cache_dir / f"{q.id}.npy").is_file():
            return False
    return True
