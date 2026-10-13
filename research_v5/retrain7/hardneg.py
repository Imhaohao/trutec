"""Document-balanced hard-negative boost with a configurable multiplier.

Same rule as research_v5/structure/weights.py: hard negatives are reviewed
negative cells whose stage-1 probability is at least the cutoff. Their weight
is multiplied, then each document's negative weights are scaled back to 0.5.
"""
from __future__ import annotations

import numpy as np


def apply_hard_negative(weights, labels, stage1, documents, cutoff, hard_weight: float):
    adjusted = np.array(weights, dtype=np.float64, copy=True)
    if cutoff is None:
        return adjusted.astype(np.float32)
    for document in np.unique(documents):
        _boost_document(
            adjusted, labels, stage1, documents, int(document), float(cutoff), float(hard_weight)
        )
    return adjusted.astype(np.float32)


def hard_negative_fraction(labels, stage1, cutoff) -> float:
    negative = np.asarray(labels) == 0
    if cutoff is None or not np.any(negative):
        return 0.0
    return float(np.mean(np.asarray(stage1)[negative] >= cutoff))


def _boost_document(weights, labels, stage1, documents, document: int, cutoff: float, hard_weight: float) -> None:
    negative = (documents == document) & (np.asarray(labels) == 0)
    if not np.any(negative):
        return
    hard = negative & (np.asarray(stage1) >= cutoff)
    if not np.any(hard):
        return
    weights[hard] *= hard_weight
    mass = float(weights[negative].sum())
    if mass <= 0.0:
        return
    weights[negative] *= 0.5 / mass
