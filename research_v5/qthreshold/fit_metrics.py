"""Apply cutoffs to cached fit reviewed scores and aggregate with the official metric."""
from __future__ import annotations

import math

import limits  # noqa: F401

from cutoffs import confusion_counts
from limits import log
from evaluate import aggregate, metrics_from_counts
from store import read_pos_neg

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


def score_many(ids, documents, columns: dict[str, list]) -> dict[str, dict]:
    rows = {name: [] for name in columns}
    for index, query_id in enumerate(ids):
        if index % 25 == 0:
            log(f"fit scoring query {index} of {len(ids)}")
        positive, negative = read_pos_neg(query_id)
        for name, values in columns.items():
            rows[name].append(_metric_row(documents[index], positive, negative, float(values[index])))
        del positive, negative
    return {name: compact_summary(aggregate(built)) for name, built in rows.items()}


def compact_summary(summary: dict) -> dict:
    negatives = summary["fp"] + summary["tn"]
    payload = {key: summary.get(key) for key in SUMMARY_KEYS if key != "reviewed_negative_fpr"}
    payload["reviewed_negative_fpr"] = summary["fp"] / negatives if negatives else None
    return payload


def _metric_row(document_id: str, positive, negative, threshold: float) -> dict:
    if not math.isfinite(threshold):
        raise RuntimeError(f"non-finite cutoff for {document_id}")
    counts = confusion_counts(positive, negative, threshold)
    return {"document_id": document_id, "metrics": metrics_from_counts(*counts)}
