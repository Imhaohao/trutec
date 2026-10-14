"""Document-macro IoU on sampled cells.

Each sampled cell stands for (known cells of its class) / (sampled cells of
its class). Query IoU is computed from those weights. Document-macro IoU
averages query IoUs inside each document, then averages documents. Pooled
recall and precision sum the weighted counts. Thresholds match the
development harness: 0, 0.01, ..., 1.
"""
from __future__ import annotations

import numpy as np

THRESHOLDS = np.linspace(0.0, 1.0, 101)
RECALL_FLOOR = 0.95


def operating_point(queries: list[dict], floor: float = RECALL_FLOOR, thresholds=THRESHOLDS) -> dict | None:
    eligible = []
    for threshold in thresholds:
        summary = summarize(queries, float(threshold))
        if _meets_floor(summary, floor):
            eligible.append(summary)
    if not eligible:
        return None
    return max(eligible, key=_rank_key)


def summarize(queries: list[dict], threshold: float) -> dict:
    rows = [_query_row(query, threshold) for query in queries]
    totals = _totals(rows)
    precision = _ratio(totals["tp"], totals["tp"] + totals["fp"])
    return {
        "threshold": float(threshold),
        "macro_iou": _macro(rows),
        "precision": precision,
        "recall": _ratio(totals["tp"], totals["tp"] + totals["fn"]),
        "fpr": _ratio(totals["fp"], totals["fp"] + totals["tn"]),
        "tp": totals["tp"],
        "fp": totals["fp"],
        "fn": totals["fn"],
        "tn": totals["tn"],
        "queries": len(rows),
        "documents": len({row["document_id"] for row in rows if row["iou"] is not None}),
    }


def prefer(candidate: dict, incumbent: dict) -> bool:
    return _rank_key(candidate) > _rank_key(incumbent)


def _query_row(query: dict, threshold: float) -> dict:
    tp, fp, fn, tn = _counts(query["labels"], query["scores"], query["pop"], threshold)
    positive = tp + fn
    iou = tp / (tp + fp + fn) if positive else None
    return {"document_id": query["document_id"], "iou": iou, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _counts(labels, scores, pop, threshold: float):
    labels = np.asarray(labels)
    chosen = np.asarray(scores) >= threshold
    positive = labels == 1
    weights = np.asarray(pop, dtype=np.float64)
    if len(labels) == 0:
        return 0.0, 0.0, 0.0, 0.0
    tp = float(weights[chosen & positive].sum())
    fn = float(weights[~chosen & positive].sum())
    fp = float(weights[chosen & ~positive].sum())
    tn = float(weights[~chosen & ~positive].sum())
    return tp, fp, fn, tn


def _macro(rows: list[dict]):
    grouped: dict[str, list[float]] = {}
    for row in rows:
        if row["iou"] is None:
            continue
        grouped.setdefault(row["document_id"], []).append(float(row["iou"]))
    if not grouped:
        return None
    document_means = [float(np.mean(values)) for values in grouped.values()]
    return float(np.mean(document_means))


def _totals(rows: list[dict]) -> dict:
    keys = ("tp", "fp", "fn", "tn")
    return {key: float(sum(row[key] for row in rows)) for key in keys}


def _ratio(numerator: float, denominator: float):
    if denominator <= 0.0:
        return None
    return numerator / denominator


def _meets_floor(summary: dict, floor: float) -> bool:
    recall = summary["recall"]
    macro = summary["macro_iou"]
    if recall is None or macro is None:
        return False
    return float(recall) >= floor


def _rank_key(summary: dict) -> tuple:
    precision = summary["precision"]
    if precision is None:
        precision = -1.0
    return (float(summary["macro_iou"]), float(precision))
