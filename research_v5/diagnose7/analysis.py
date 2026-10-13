"""False positive analysis for v7 on the development split.

Computes confusion matrices, spatial boundary bleed, query cleaning effects,
score distributions, and failure mode categorization.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bootstrap  # noqa: E402,F401

from collections import defaultdict  # noqa: E402
from dataclasses import dataclass  # noqa: E402

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

from bootstrap import DATA_ROOT, STRUCTURE_MODEL, V7_DEV_CACHE, guard_memory, log
from devharness import development_queries
from guards import _ink, _parts, clean_crop_g2


@dataclass
class QueryStat:
    id: str
    document_id: str
    image: str
    width: int
    height: int
    query_box: tuple[int, int, int, int]
    tp: int
    fp: int
    fn: int
    tn: int
    iou: float
    recall: float
    precision: float
    fpr: float
    scores: np.ndarray
    decision: np.ndarray
    positive: np.ndarray
    y: np.ndarray
    x: np.ndarray


def load_or_compute_v7_scores() -> dict[str, np.ndarray]:
    """Load cached v7 dev scores or recompute with pipeline.py if missing."""
    queries = development_queries()
    all_cached = all((V7_DEV_CACHE / f"{q.id}.npz").is_file() for q in queries)
    if all_cached:
        log(f"loading v7 scores from cache {V7_DEV_CACHE}")
        return {q.id: np.load(V7_DEV_CACHE / f"{q.id}.npz")["scores"] for q in queries}

    log("cache missing, recomputing v7 dev scores via pipeline.py")
    return _recompute_scores(queries)


def _recompute_scores(queries) -> dict[str, np.ndarray]:
    from evaluate_dev import load_original
    from inference_v7 import load_stage2
    from pipeline import cleaner_by_name, map_request, prepare_drawing, release_drawing

    stage1, _ = load_original()
    stage2 = load_stage2(STRUCTURE_MODEL)
    cleaner = cleaner_by_name("g2")

    grouped: dict[str, list] = defaultdict(list)
    for q in queries:
        grouped[q.image].append(q)

    scores = {}
    for image, img_queries in grouped.items():
        requests = [_request_dict(q) for q in img_queries]
        context = prepare_drawing(requests, DATA_ROOT)
        try:
            for index, (q, req) in enumerate(zip(img_queries, requests)):
                gated = map_request(context, index, req, stage1, stage2, cleaner)
                scores[q.id] = np.asarray(gated[q.y, q.x], dtype=np.float32)
                del gated
        finally:
            release_drawing(context)
        guard_memory()
    return scores


def _request_dict(q) -> dict:
    return {
        "id": q.id,
        "image": q.image,
        "width": q.width,
        "height": q.height,
        "query_box": list(q.query_box),
        "context_boxes": [list(box) for box in q.context_boxes],
    }


def compute_query_and_doc_stats(scores: dict[str, np.ndarray], threshold: float = 0.25):
    """Compute per-query and per-document confusion matrices and metrics."""
    queries = development_queries()
    query_stats: list[QueryStat] = []
    doc_groups = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "ious": []})

    for q in queries:
        stat = _single_query_stat(q, scores[q.id], threshold)
        query_stats.append(stat)

        d = doc_groups[q.document_id]
        d["tp"] += stat.tp
        d["fp"] += stat.fp
        d["fn"] += stat.fn
        d["tn"] += stat.tn
        d["ious"].append(stat.iou)

    query_stats.sort(key=lambda s: s.fp, reverse=True)
    doc_stats = _aggregate_doc_stats(doc_groups)
    return query_stats, doc_stats


def _single_query_stat(q, sc: np.ndarray, threshold: float) -> QueryStat:
    dec = sc >= threshold
    pos = q.positive
    tp = int(np.count_nonzero(dec & pos))
    fp = int(np.count_nonzero(dec & ~pos))
    fn = int(np.count_nonzero(~dec & pos))
    tn = int(np.count_nonzero(~dec & ~pos))

    denom_iou = tp + fp + fn
    iou = float(tp / denom_iou) if denom_iou > 0 else 1.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    return QueryStat(
        id=q.id,
        document_id=q.document_id,
        image=q.image,
        width=q.width,
        height=q.height,
        query_box=q.query_box,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        iou=iou,
        recall=recall,
        precision=precision,
        fpr=fpr,
        scores=sc,
        decision=dec,
        positive=pos,
        y=q.y,
        x=q.x,
    )


def _aggregate_doc_stats(doc_groups: dict) -> list[dict]:
    doc_stats = []
    for doc_id, d in doc_groups.items():
        total_p = d["tp"] + d["fn"]
        total_pred = d["tp"] + d["fp"]
        total_neg = d["fp"] + d["tn"]
        doc_stats.append(
            {
                "document_id": doc_id,
                "queries_count": len(d["ious"]),
                "tp": d["tp"],
                "fp": d["fp"],
                "fn": d["fn"],
                "tn": d["tn"],
                "macro_iou": float(np.mean(d["ious"])),
                "precision": float(d["tp"] / total_pred) if total_pred > 0 else 0.0,
                "recall": float(d["tp"] / total_p) if total_p > 0 else 0.0,
                "fpr": float(d["fp"] / total_neg) if total_neg > 0 else 0.0,
            }
        )
    doc_stats.sort(key=lambda d: d["fp"], reverse=True)
    return doc_stats


def compute_concentration(query_stats: list[QueryStat], doc_stats: list[dict]) -> dict:
    """Analyze how concentrated false positives are across queries and documents."""
    total_fp = sum(s.fp for s in query_stats)
    total_tp = sum(s.tp for s in query_stats)

    top_q = {
        1: sum(s.fp for s in query_stats[:1]),
        3: sum(s.fp for s in query_stats[:3]),
        5: sum(s.fp for s in query_stats[:5]),
        10: sum(s.fp for s in query_stats[:10]),
    }
    top_d = {
        1: sum(d["fp"] for d in doc_stats[:1]),
        2: sum(d["fp"] for d in doc_stats[:2]),
        3: sum(d["fp"] for d in doc_stats[:3]),
    }

    zero_fp_docs = sum(1 for d in doc_stats if d["fp"] == 0)
    zero_fp_queries = sum(1 for s in query_stats if s.fp == 0)

    return {
        "total_fp": total_fp,
        "total_tp": total_tp,
        "top_queries_fp": {k: {"count": v, "share": float(v / total_fp)} for k, v in top_q.items()},
        "top_docs_fp": {k: {"count": v, "share": float(v / total_fp)} for k, v in top_d.items()},
        "zero_fp_docs_count": zero_fp_docs,
        "zero_fp_queries_count": zero_fp_queries,
    }


def analyze_spatial_distances(query_stats: list[QueryStat], top_k: int = 10, distance_px: float = 24.0) -> list[dict]:
    """Measure boundary bleed (<= 24 px from TP) vs far FP (> 24 px)."""
    results = []
    for stat in query_stats[:top_k]:
        tp_mask = stat.decision & stat.positive
        fp_mask = stat.decision & ~stat.positive

        if not np.any(fp_mask):
            results.append(_empty_distance_record(stat))
            continue

        dists = _compute_fp_distances(stat, tp_mask, fp_mask)
        near = int(np.count_nonzero(dists <= distance_px))
        far = int(np.count_nonzero(dists > distance_px))
        results.append(
            {
                "query_id": stat.id,
                "document_id": stat.document_id,
                "fp": stat.fp,
                "tp": stat.tp,
                "boundary_bleed_near_24px": near,
                "boundary_bleed_fraction": float(near / stat.fp) if stat.fp > 0 else 0.0,
                "far_fp_over_24px": far,
                "far_fp_fraction": float(far / stat.fp) if stat.fp > 0 else 0.0,
            }
        )
    return results


def _compute_fp_distances(stat: QueryStat, tp_mask: np.ndarray, fp_mask: np.ndarray) -> np.ndarray:
    if not np.any(tp_mask):
        return np.full(stat.fp, np.inf)

    grid_tp = np.zeros((stat.height, stat.width), dtype=bool)
    grid_tp[stat.y[tp_mask], stat.x[tp_mask]] = True
    dist_map = distance_transform_edt(~grid_tp)
    return dist_map[stat.y[fp_mask], stat.x[fp_mask]]


def _empty_distance_record(stat: QueryStat) -> dict:
    return {
        "query_id": stat.id,
        "document_id": stat.document_id,
        "fp": stat.fp,
        "tp": stat.tp,
        "boundary_bleed_near_24px": 0,
        "boundary_bleed_fraction": 0.0,
        "far_fp_over_24px": 0,
        "far_fp_fraction": 0.0,
    }


def analyze_query_cleaning(query_stats: list[QueryStat], top_k: int = 10, data_root: Path = DATA_ROOT) -> list[dict]:
    """Inspect query crop cleaning (g2) for ink and glyph removal."""
    results = []
    for stat in query_stats[:top_k]:
        with Image.open(data_root / stat.image) as img:
            gray = np.asarray(img.convert("L"))

        x0, y0, x1, y1 = stat.query_box
        orig = np.ascontiguousarray(gray[y0:y1, x0:x1])
        cleaned = clean_crop_g2(orig)

        ink_before = int(np.count_nonzero(_ink(orig)))
        ink_after = int(np.count_nonzero(_ink(cleaned)))
        removed_ink = ink_before - ink_after

        ink_bin = np.asarray(_ink(orig), dtype=np.uint8)
        cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(ink_bin, connectivity=8)
        parts = _parts(orig.shape, labels, stats, centroids, cnt) if cnt > 1 else []

        glyphs = [p for p in parts if p.glyph]
        long_lines = [p for p in parts if p.long_line]

        results.append(
            {
                "query_id": stat.id,
                "crop_shape": list(orig.shape),
                "ink_before": ink_before,
                "ink_after": ink_after,
                "ink_removed": removed_ink,
                "components_total": len(parts),
                "glyphs_in_crop": len(glyphs),
                "long_lines_in_crop": len(long_lines),
                "glyph_removed_flag": len(glyphs) > 0 and removed_ink > 0,
            }
        )
    return results


def analyze_score_distributions(query_stats: list[QueryStat], top_k: int = 10) -> list[dict]:
    """Compute median and mean stage-2 scores of FP vs TP pixels."""
    results = []
    for stat in query_stats[:top_k]:
        tp_mask = stat.decision & stat.positive
        fp_mask = stat.decision & ~stat.positive

        fp_sc = stat.scores[fp_mask]
        tp_sc = stat.scores[tp_mask]

        med_fp = float(np.median(fp_sc)) if len(fp_sc) > 0 else 0.0
        mean_fp = float(np.mean(fp_sc)) if len(fp_sc) > 0 else 0.0
        med_tp = float(np.median(tp_sc)) if len(tp_sc) > 0 else 0.0
        mean_tp = float(np.mean(tp_sc)) if len(tp_sc) > 0 else 0.0

        results.append(
            {
                "query_id": stat.id,
                "document_id": stat.document_id,
                "median_fp_score": med_fp,
                "median_tp_score": med_tp,
                "mean_fp_score": mean_fp,
                "mean_tp_score": mean_tp,
            }
        )
    return results


def categorize_all_false_positives(query_stats: list[QueryStat], data_root: Path = DATA_ROOT) -> dict:
    """Categorize every single FP pixel into exact failure categories."""
    doc_queries = defaultdict(list)
    for q in query_stats:
        doc_queries[q.document_id].append(q)

    images = _load_unique_images(query_stats, data_root)
    categories = {
        "competing_hatch_swatches": 0,
        "blank_paper_texture": 0,
        "linework_and_geometry": 0,
        "text_glyphs_annotations": 0,
        "boundary_bleed_near_tp": 0,
    }

    for stat in query_stats:
        _categorize_single_query(stat, doc_queries[stat.document_id], images[stat.image], categories)

    return categories


def _load_unique_images(query_stats: list[QueryStat], data_root: Path) -> dict:
    images = {}
    for stat in query_stats:
        if stat.image not in images:
            with Image.open(data_root / stat.image) as img:
                gray = np.asarray(img.convert("L"))
                ink_map = (gray < 210).astype(np.uint8)
                cnt, labels, stats, _ = cv2.connectedComponentsWithStats(ink_map, connectivity=8)
                images[stat.image] = {"gray": gray, "labels": labels, "stats": stats}
    return images


def _categorize_single_query(stat: QueryStat, siblings: list[QueryStat], img_data: dict, categories: dict) -> None:
    fp_mask = stat.decision & ~stat.positive
    if not np.any(fp_mask):
        return

    fp_y = stat.y[fp_mask]
    fp_x = stat.x[fp_mask]

    # Map of all other positive regions in the document
    other_pos = set()
    for sib in siblings:
        if sib.id != stat.id:
            for py, px in zip(sib.y[sib.positive], sib.x[sib.positive]):
                other_pos.add((py, px))

    # Distance to true positive
    tp_mask = stat.decision & stat.positive
    dists = _compute_fp_distances(stat, tp_mask, fp_mask)

    gray = img_data["gray"]
    labels = img_data["labels"]
    stats = img_data["stats"]

    for idx, (py, px) in enumerate(zip(fp_y, fp_x)):
        if (py, px) in other_pos:
            categories["competing_hatch_swatches"] += 1
        elif dists[idx] <= 24.0:
            categories["boundary_bleed_near_tp"] += 1
        elif gray[py, px] >= 210:
            categories["blank_paper_texture"] += 1
        else:
            lbl = labels[py, px]
            if lbl > 0 and _is_glyph_component(stats[lbl]):
                categories["text_glyphs_annotations"] += 1
            else:
                categories["linework_and_geometry"] += 1


def _is_glyph_component(stat_row: np.ndarray) -> bool:
    w = stat_row[cv2.CC_STAT_WIDTH]
    h = stat_row[cv2.CC_STAT_HEIGHT]
    area = stat_row[cv2.CC_STAT_AREA]
    aspect = max(w / max(h, 1), h / max(w, 1))
    return bool(w <= 45 and h <= 45 and area <= 700 and aspect <= 2.5)
