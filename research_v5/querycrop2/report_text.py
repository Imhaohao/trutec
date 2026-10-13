"""Findings text. Every number is copied from a report dict produced by this run."""
from __future__ import annotations

import bootstrap  # noqa: F401

from guards import (
    DESCRIPTOR_STRIDE,
    EMPTY_INK,
    FAMILY_COUNT,
    INK_REMOVE_LIMIT,
    ORIENTATION_DEGREES,
    PAPER_LEVEL,
    SPACING_CV_LIMIT,
    SPAN_FRACTION,
)
from metrics import best_at_recall, best_macro_iou, best_precision_at_recall

ORDER = ("v6", "rule_d", "g1", "g2", "g3")
TARGET = "q-42b50c596dd17d8f"


def render(fit_reports: dict, dev_reports: dict, chosen: str, diffs: list[dict], runtime: dict) -> str:
    lines = [_rules(), "", _table("FIT SPLIT (selection)", fit_reports), ""]
    lines.append(f"Selected variant: {chosen}")
    lines.append(
        "Selection used only the fit split, and only g1, g2, and g3. "
        "The winning point is the highest document-macro IoU among thresholds with pooled recall >= 0.95. "
        "Equal IoU is broken by pooled precision at that same point. "
        "Development scores were computed after this choice."
    )
    lines.append("")
    lines.append(_table("DEVELOPMENT SPLIT (frozen v6 models, after selection)", dev_reports))
    lines.append("")
    lines.extend(_target(diffs))
    lines.append("")
    lines.extend(_top(diffs))
    lines.append("")
    lines.append(
        f"Runtime seconds={runtime['seconds']:.1f} peak_gb={runtime['peak_gb']:.2f} "
        f"fit_queries={runtime['fit_queries']} dev_queries={runtime['dev_queries']}"
    )
    if runtime.get("v6_gap") is not None:
        lines.append(f"Max absolute gap between recomputed v6 development scores and gated stacker cache: {runtime['v6_gap']:.6g}")
    if runtime.get("rule_d_gap") is not None:
        lines.append(
            "Max absolute gap between recomputed rule d development scores and "
            f"querycrop cache: {runtime['rule_d_gap']:.6g}"
        )
    lines.append("")
    lines.append(_failure_note(diffs, chosen))
    return "\n".join(lines) + "\n"


def _rules() -> str:
    return "\n".join(
        [
            "Guarded query cleaning (research_v5/querycrop2/)",
            "Selection was made on the fit split before any development score from this run was read.",
            "",
            "Shared constants",
            f"- Ink pixel: grayscale < {PAPER_LEVEL}.",
            f"- Descriptor stride {DESCRIPTOR_STRIDE}. Stage-1 local evidence stride 8.",
            "- Color gate: window 16, linear, strength 0.6, applied after stride-8 cell lookup.",
            "- Fit models: stage-1 research_v5/stacker/cache/folds/fold_k.joblib and stage-2",
            "  research_v5/autocontext/cache/stage2_folds/fold_k.joblib, where k is the query document's",
            "  fold in research_v5/stacker/folds.json. That stage-1 model must not list the document in",
            "  training_documents. Development uses the frozen crop model and",
            "  research_v5/stacker/cache/stage2_model.joblib.",
            "- Scored pixels: fit reviewed-known pixels after query and context exclusion; development",
            "  pixels from devharness. Aggregation is evaluate.metrics_from_counts and evaluate.aggregate.",
            "",
            "Rule d, imported from research_v5/querycrop/cleaners.py:",
            "- Remove a connected component that is a long line or a text glyph, then take the IQR cell median.",
            "- Long line: one side spans at least half the crop, aspect >= 4, and the thin side is",
            "  at most max(6, 10% of that crop side).",
            "- Glyph: height 8-45, width 6-45, aspect <= 2.5, area 40-700, and either the area is",
            "  at least 3 times the crop's median component area or the crop has at least 5 components.",
            f"- If fewer than {EMPTY_INK} ink pixels remain and the crop started with more than 50, restore it.",
            "",
            "G1: rule d's component pass, but if it removes more than "
            f"{INK_REMOVE_LIMIT:.0%} of the ink pixels, keep the uncleaned crop. Then IQR.",
            "G2: same glyph rule. A component is spanning when its box covers at least "
            f"{SPAN_FRACTION:.0%} of both crop sides, and every spanning component is kept.",
            f"A long line is kept when it belongs to a family of at least {FAMILY_COUNT} long lines whose",
            f"undirected orientations agree within {ORIENTATION_DEGREES:.0f} degrees and whose spacing is regular.",
            "Spacing is regular when the mean gap between centroids, measured along the normal to the",
            f"family orientation, is greater than 1 pixel and the population standard deviation / mean is <= {SPACING_CV_LIMIT}.",
            "Isolated long lines are removed. Then the same empty-crop restore as rule d. Then IQR.",
            "G3: G2, then the G1 ink fallback, then IQR.",
            "v6: uncleaned crop median, no component pass and no IQR.",
        ]
    )


def _table(title: str, reports: dict) -> str:
    header = f"{'variant':<10} {'best macro IoU':<42} {'IoU at recall>=0.95':<42} {'precision at recall>=0.98'}"
    rows = [title, header]
    for name in ORDER:
        points = reports[name]["points"]
        rows.append(
            f"{name:<10} {_fmt(best_macro_iou(points)):<42} "
            f"{_fmt(best_at_recall(points, 0.95)):<42} {_fmt(best_precision_at_recall(points, 0.98))}"
        )
    return "\n".join(rows)


def _fmt(point) -> str:
    if point is None:
        return "none"
    return (
        f"{point['document_macro_iou']:.4f} th={point['threshold']:.2f} "
        f"rec={point['global_recall']:.4f} pr={point['global_precision']:.4f}"
    )


def _target(diffs: list[dict]) -> list[str]:
    row = next(item for item in diffs if item["id"] == TARGET)
    repaired = _repaired(row)
    return [
        f"{TARGET} at threshold 0.14: v6 IoU {row['iou_v6']:.4f}, rule d IoU {row['iou_rule_d']:.4f}, "
        f"selected IoU {row['iou_selected']:.4f}.",
        f"Repaired: {repaired}. Repaired means the selected IoU is within 0.05 of v6, or closer to v6 than to rule d.",
    ]


def _repaired(row: dict) -> str:
    selected = row["iou_selected"]
    v6 = row["iou_v6"]
    rule_d = row["iou_rule_d"]
    if selected >= v6 - 0.05:
        return "yes"
    if abs(selected - v6) < abs(selected - rule_d):
        return "yes"
    return "no"


def _top(diffs: list[dict]) -> list[str]:
    lines = ["Ten development queries with the largest absolute IoU change versus v6 at threshold 0.14:"]
    lines.append(f"{'query':<22} {'document':<22} {'v6':>8} {'selected':>8} {'delta':>8}")
    for row in diffs[:10]:
        lines.append(
            f"{row['id']:<22} {row['document_id']:<22} {row['iou_v6']:8.4f} "
            f"{row['iou_selected']:8.4f} {row['delta']:8.4f}"
        )
    return lines


def _failure_note(diffs: list[dict], chosen: str) -> str:
    row = next(item for item in diffs if item["id"] == TARGET)
    if _repaired(row) == "yes":
        return (
            f"What failed: {TARGET} is repaired by {chosen} at threshold 0.14, "
            f"from rule d IoU {row['iou_rule_d']:.4f} to {row['iou_selected']:.4f} "
            f"(v6 was {row['iou_v6']:.4f}). Remaining misses are the other nine large IoU changes above."
        )
    return (
        f"What failed: {chosen} does not repair {TARGET} at threshold 0.14 "
        f"(v6 {row['iou_v6']:.4f}, rule d {row['iou_rule_d']:.4f}, selected {row['iou_selected']:.4f}). "
        "The vertical stroke in that crop is one isolated long line beside a diagonal component, "
        "so G2 still deletes it, and the deleted ink is under half the crop so G1 does not fall back."
    )
