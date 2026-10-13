"""Document-balanced weights plus an optional hard-negative boost.

Hard negatives are reviewed negative cells whose stage-1 probability is at
least the cutoff. Their weight is multiplied by 3, then the negative weights
inside each document are scaled back to 0.5 so documents stay balanced and
the boost stays a 3x ratio against easy negatives.
"""
from __future__ import annotations

import numpy as np

HARD_WEIGHT = 3.0
SETTINGS = (
    {"name": "structure", "cutoff": None},
    {"name": "hardneg_0.50", "cutoff": 0.50},
    {"name": "hardneg_0.75", "cutoff": 0.75},
)


def apply_hard_negative(weights, labels, stage1, documents, cutoff):
    adjusted = np.array(weights, dtype=np.float64, copy=True)
    if cutoff is None:
        return adjusted.astype(np.float32)
    for document in np.unique(documents):
        _boost_document(adjusted, labels, stage1, documents, int(document), float(cutoff))
    return adjusted.astype(np.float32)


def hard_negative_fraction(labels, stage1, cutoff) -> float:
    negative = np.asarray(labels) == 0
    if cutoff is None or not np.any(negative):
        return 0.0
    return float(np.mean(np.asarray(stage1)[negative] >= cutoff))


def _boost_document(weights, labels, stage1, documents, document: int, cutoff: float) -> None:
    negative = (documents == document) & (labels == 0)
    if not np.any(negative):
        return
    hard = negative & (np.asarray(stage1) >= cutoff)
    if not np.any(hard):
        return
    weights[hard] *= HARD_WEIGHT
    mass = float(weights[negative].sum())
    if mass <= 0.0:
        return
    weights[negative] *= 0.5 / mass
