"""Sibling-competition score rules.

A rival list is the other queries on the same drawing that passed the
label-free competitor test. An empty rival list returns the original
probability array unchanged.
"""

from __future__ import annotations

import numpy as np


def hard_winner(own: np.ndarray, rivals: list[np.ndarray]) -> np.ndarray:
    """R1: keep p_A where it beats every competitor, else multiply by 0.25."""
    if not rivals:
        return own
    rival_max = _rival_max(own, rivals)
    won = own >= rival_max
    mixed = np.where(won, own, own * np.float32(0.25))
    return mixed.astype(np.float32, copy=False)


def soft_margin(own: np.ndarray, rivals: list[np.ndarray], temperature: float) -> np.ndarray:
    """R2: p_A * min(1, 2 * sigmoid((p_A - max rival) / t))."""
    if not rivals:
        return own
    margin = (own.astype(np.float64) - _rival_max(own, rivals)) / float(temperature)
    multiplier = np.minimum(1.0, 2.0 / (1.0 + np.exp(-margin)))
    return (own.astype(np.float64) * multiplier).astype(np.float32)


def softmax_share(own: np.ndarray, rivals: list[np.ndarray], temperature: float) -> np.ndarray:
    """R3: p_A times A's softmax share against its competitors."""
    if not rivals:
        return own
    stack = np.vstack([own, *rivals]).astype(np.float64)
    scaled = stack / float(temperature)
    scaled -= np.max(scaled, axis=0, keepdims=True)
    weights = np.exp(scaled)
    share = weights[0] / np.sum(weights, axis=0)
    return (own.astype(np.float64) * share).astype(np.float32)


def assert_unchanged_without_rivals(scores: dict[str, np.ndarray], baseline: dict[str, np.ndarray], rivals: dict[str, list[str]]) -> None:
    for query_id, rival_ids in rivals.items():
        if rival_ids:
            continue
        if not np.array_equal(scores[query_id], baseline[query_id]):
            raise AssertionError(f"{query_id} changed despite having no competitor")


def _rival_max(own: np.ndarray, rivals: list[np.ndarray]) -> np.ndarray:
    acc = np.array(rivals[0], dtype=np.float32, copy=True)
    for rival in rivals[1:]:
        if rival.shape != own.shape:
            raise ValueError("competitor probability length does not match the query")
        np.maximum(acc, rival, out=acc)
    return acc


def _self_check() -> None:
    own = np.array([0.2, 0.8, 0.5], dtype=np.float32)
    for rule in (
        lambda values: hard_winner(values, []),
        lambda values: soft_margin(values, [], 0.05),
        lambda values: softmax_share(values, [], 0.1),
    ):
        if not np.array_equal(rule(own), own):
            raise AssertionError("empty rival list must return the original array")
    stronger = np.array([0.9, 0.1, 0.5], dtype=np.float32)
    winner = hard_winner(own, [stronger])
    if not np.isclose(winner[0], 0.05) or not np.isclose(winner[1], 0.8) or not np.isclose(winner[2], 0.5):
        raise AssertionError(f"hard winner mismatch: {winner}")
    tied = soft_margin(own, [own.copy()], 0.15)
    if not np.allclose(tied, own):
        raise AssertionError(f"tied soft margin must keep p_A, got {tied}")
    shared = softmax_share(np.array([0.4], dtype=np.float32), [np.array([0.4], dtype=np.float32)], 0.1)
    if not np.isclose(shared[0], 0.2):
        raise AssertionError(f"equal softmax share must be 1/2, got {shared}")


if __name__ == "__main__":
    _self_check()
    print("rules ok")
