"""Masked BCE plus soft Dice. Unknown output cells contribute nothing."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from .constants import KNOWN_FRACTION

EPS = 1e-6


def masked_loss(
    logits: torch.Tensor,
    known_frac: torch.Tensor,
    pos_frac: torch.Tensor,
    threshold: float = KNOWN_FRACTION,
) -> tuple[torch.Tensor, dict]:
    """known_frac and pos_frac are area averages on the logit grid.

    A cell is supervised only when at least `threshold` of its native pixels
    are reviewed-known. Its target is the positive area fraction.
    """
    if tuple(logits.shape[-2:]) != tuple(known_frac.shape[-2:]):
        raise ValueError("logits and mask grid differ")
    known = known_frac >= threshold
    bce = _masked_bce(logits, pos_frac, known)
    dice = _soft_dice(logits, pos_frac, known)
    total = bce + dice
    return total, _stats(total, bce, dice, pos_frac, known)


def _masked_bce(logits: torch.Tensor, target: torch.Tensor, known: torch.Tensor) -> torch.Tensor:
    if not bool(known.any()):
        return logits.sum() * 0
    return F.binary_cross_entropy_with_logits(logits[:, 0][known], target[known])


def _soft_dice(logits: torch.Tensor, target: torch.Tensor, known: torch.Tensor) -> torch.Tensor:
    prob = torch.sigmoid(logits[:, 0])
    mask = known.to(dtype=prob.dtype)
    pred = prob * mask
    filled = target * mask
    overlap = (pred * filled).sum()
    denom = pred.sum() + filled.sum()
    return 1 - (2 * overlap + EPS) / (denom + EPS)


def _mean_target(target: torch.Tensor, known: torch.Tensor) -> float | None:
    if not bool(known.any()):
        return None
    return float(target[known].detach().mean())


def _stats(total, bce, dice, target, known) -> dict:
    return {
        "loss": float(total.detach()),
        "bce": float(bce.detach()),
        "dice": float(dice.detach()),
        "known_cells": int(known.sum().detach()),
        "pos_fraction": _mean_target(target, known),
    }
