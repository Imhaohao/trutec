"""Cutoff targets from positive and negative scores the caller already extracted.

The IoU cutoff is the largest grid point that maximizes TP/(TP+FP+FN).
The recall cutoff is the largest grid point whose recall stays at or above 0.98.
A query with no positive scores has no target.
"""
from __future__ import annotations

import math

import limits  # noqa: F401

import numpy as np

from experiment import RECALL_FLOOR, TARGET_STEPS


def threshold_grid(steps: int = TARGET_STEPS) -> np.ndarray:
    return np.linspace(0.0, 1.0, steps)


def best_cutoffs(pos, neg, steps: int = TARGET_STEPS) -> tuple[float, float]:
    positive = np.sort(np.asarray(pos, dtype=np.float64))
    if positive.size == 0:
        return math.nan, math.nan
    negative = np.sort(np.asarray(neg, dtype=np.float64))
    thresholds = threshold_grid(steps)
    _true_positive, _false_positive, recall, iou = _curves(positive, negative, thresholds)
    return float(thresholds[_last_max(iou)]), float(thresholds[_recall_index(recall)])


def confusion_counts(pos, neg, threshold: float) -> tuple[int, int, int, int]:
    positive = np.asarray(pos)
    negative = np.asarray(neg)
    true_positive = int(np.count_nonzero(positive >= threshold))
    false_positive = int(np.count_nonzero(negative >= threshold))
    false_negative = int(positive.size - true_positive)
    true_negative = int(negative.size - false_positive)
    return true_positive, false_positive, false_negative, true_negative


def _curves(positive: np.ndarray, negative: np.ndarray, thresholds: np.ndarray):
    true_positive = positive.size - np.searchsorted(positive, thresholds, side="left")
    false_positive = negative.size - np.searchsorted(negative, thresholds, side="left")
    false_negative = positive.size - true_positive
    denominator = true_positive + false_positive + false_negative
    iou = np.zeros(len(thresholds), dtype=np.float64)
    np.divide(true_positive, denominator, out=iou, where=denominator > 0)
    recall = true_positive / float(positive.size)
    return true_positive, false_positive, recall, iou


def _last_max(values: np.ndarray) -> int:
    return int(np.flatnonzero(values == np.max(values))[-1])


def _recall_index(recall: np.ndarray) -> int:
    eligible = np.flatnonzero(recall >= RECALL_FLOOR)
    if eligible.size:
        return int(eligible[-1])
    return _last_max(recall)
