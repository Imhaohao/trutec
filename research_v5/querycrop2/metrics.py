"""Threshold sweeps with the official count metrics. One query is loaded at a time."""
from __future__ import annotations

import bootstrap  # noqa: F401

import numpy as np

from evaluate import aggregate, metrics_from_counts
from guards import GUARDED

THRESHOLDS = np.linspace(0.0, 1.0, 101)
SUMMARY_KEYS = (
    "document_macro_iou",
    "global_recall",
    "global_precision",
    "reviewed_negative_fpr",
    "tp",
    "fp",
    "fn",
    "tn",
)


def sweep_cached(jobs: list[dict], variant: str) -> list[dict]:
    packed = [_query_counts(job, variant) for job in jobs]
    return [_point(packed, index) for index in range(len(THRESHOLDS))]


def best_macro_iou(points: list[dict]) -> dict | None:
    if not points:
        return None
    return max(points, key=lambda point: point["document_macro_iou"] or 0.0)


def best_at_recall(points: list[dict], recall: float) -> dict | None:
    eligible = [point for point in points if (point["global_recall"] or 0.0) >= recall]
    if not eligible:
        return None
    return max(eligible, key=_recall_rank)


def best_precision_at_recall(points: list[dict], recall: float) -> dict | None:
    eligible = [point for point in points if (point["global_recall"] or 0.0) >= recall]
    if not eligible:
        return None
    return max(eligible, key=lambda point: (point["global_precision"] or 0.0, point["document_macro_iou"] or 0.0))


def choose_guard(reports: dict[str, dict]) -> tuple[str, dict]:
    """Highest fit macro IoU at recall >= 0.95. Precision breaks ties. Guards only."""
    winner = None
    for name in GUARDED:
        point = best_at_recall(reports[name]["points"], 0.95)
        if point is None:
            continue
        rank = _recall_rank(point)
        if winner is None or rank > winner[0]:
            winner = (rank, name, point)
    if winner is None:
        raise RuntimeError("no guarded variant reached recall 0.95 on the fit split")
    return winner[1], winner[2]


def query_iou(path, variant: str, threshold: float = 0.14) -> float:
    scores, positive = _load_pair(path, variant)
    decision = scores >= threshold
    tp = int(np.count_nonzero(decision & positive))
    fp = int(np.count_nonzero(decision & ~positive))
    fn = int(np.count_nonzero(~decision & positive))
    denom = tp + fp + fn
    return float(tp / denom) if denom else 1.0


def _query_counts(job: dict, variant: str):
    scores, positive = _load_pair(job["cache_path"], variant)
    kept = scores[positive]
    dropped = scores[~positive]
    del scores
    true_positive = np.fromiter((_above(kept, threshold) for threshold in THRESHOLDS), dtype=np.int64)
    false_positive = np.fromiter((_above(dropped, threshold) for threshold in THRESHOLDS), dtype=np.int64)
    return (
        job["document_id"],
        true_positive,
        false_positive,
        len(kept) - true_positive,
        len(dropped) - false_positive,
    )


def _above(values: np.ndarray, threshold: float) -> int:
    return int(np.count_nonzero(values >= threshold))


def _load_pair(path, variant: str):
    with np.load(path) as stored:
        scores = np.asarray(stored[variant], dtype=np.float32)
        positive = np.asarray(stored["positive"]).astype(bool)
    if len(scores) != len(positive):
        raise RuntimeError(f"score length mismatch in {path}")
    return scores, positive


def _point(packed, index: int) -> dict:
    rows = [_metric_row(item, index) for item in packed]
    summary = aggregate(rows)
    negatives = summary["fp"] + summary["tn"]
    compact = {key: summary[key] for key in SUMMARY_KEYS if key != "reviewed_negative_fpr"}
    compact["reviewed_negative_fpr"] = summary["fp"] / negatives if negatives else None
    return {"threshold": float(THRESHOLDS[index]), **compact}


def _metric_row(item, index: int) -> dict:
    document, true_positive, false_positive, false_negative, true_negative = item
    counts = (
        int(true_positive[index]),
        int(false_positive[index]),
        int(false_negative[index]),
        int(true_negative[index]),
    )
    return {"document_id": document, "metrics": metrics_from_counts(*counts)}


def _recall_rank(point: dict) -> tuple[float, float]:
    return (point["document_macro_iou"] or 0.0, point["global_precision"] or 0.0)
