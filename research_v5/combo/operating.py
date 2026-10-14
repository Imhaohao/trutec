"""Development operating points. The threshold rule is the v1/v2/v6 rule."""
from __future__ import annotations


def recall_point(points: list[dict], floor: float) -> dict | None:
    """Max document-macro IoU among thresholds with pooled recall >= floor.

    Ties take the point with fewer reviewed false positives.
    """
    eligible = [point for point in points if (point.get("global_recall") or 0.0) >= floor]
    if not eligible:
        return None
    return max(eligible, key=_macro_then_fewer_fp)


def _macro_then_fewer_fp(point: dict) -> tuple:
    return (point.get("document_macro_iou") or 0.0, -(point.get("fp") or 0))
