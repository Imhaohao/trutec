"""Anchor-to-cell InfoNCE plus class-balanced BCE. Unknown cells are dropped first."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def contrastive_loss(
    anchor: torch.Tensor,
    cells: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, dict]:
    """labels use 1 positive, 0 negative, and -1 unknown. Unknown rows are not read."""
    known = labels >= 0
    if int(known.sum()) == 0:
        zero = anchor.sum() * 0
        return zero, _empty_stats()
    anchor_n = F.normalize(anchor, dim=0)
    cells_n = F.normalize(cells[known], dim=-1)
    positive = labels[known] == 1
    cosine = cells_n @ anchor_n
    prob = _probabilities(cosine)
    nce = info_nce(anchor_n, cells_n, positive, temperature)
    bce = balanced_bce(prob, positive)
    total = nce + bce
    return total, _stats(total, nce, bce, prob, positive)


def info_nce(
    anchor: torch.Tensor,
    cells: torch.Tensor,
    positive: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    logits = (cells @ anchor) / temperature
    if not bool(positive.any()):
        return logits.sum() * 0
    chosen = logits[positive]
    return -(chosen - torch.logsumexp(logits, dim=0)).mean()


def balanced_bce(prob: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
    terms = []
    if bool(positive.any()):
        terms.append(_bce_class(prob[positive], 1.0))
    if bool((~positive).any()):
        terms.append(_bce_class(prob[~positive], 0.0))
    if not terms:
        return prob.sum() * 0
    return sum(terms) / len(terms)


def _bce_class(prob: torch.Tensor, target: float) -> torch.Tensor:
    filled = torch.full_like(prob, target)
    return F.binary_cross_entropy(prob, filled)


def _probabilities(cosine: torch.Tensor) -> torch.Tensor:
    return torch.clamp((cosine + 1) * 0.5, 1e-4, 1 - 1e-4)


def _mean_score(prob: torch.Tensor, mask: torch.Tensor) -> float | None:
    if not bool(mask.any()):
        return None
    return float(prob[mask].detach().mean())


def _stats(total, nce, bce, prob, positive) -> dict:
    return {
        "loss": float(total.detach()),
        "nce": float(nce.detach()),
        "bce": float(bce.detach()),
        "n_pos": int(positive.sum()),
        "n_neg": int((~positive).sum()),
        "pos_score": _mean_score(prob, positive),
        "neg_score": _mean_score(prob, ~positive),
    }


def _empty_stats() -> dict:
    return {
        "loss": 0.0,
        "nce": 0.0,
        "bce": 0.0,
        "n_pos": 0,
        "n_neg": 0,
        "pos_score": None,
        "neg_score": None,
    }
