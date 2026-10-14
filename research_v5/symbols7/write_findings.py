"""Write findings.txt from the JSON this run saved."""
from __future__ import annotations

import time

import guard

guard.activate()

from guard import CV_PATH, DEV_REPORT, FINDINGS_PATH, SUMMARY_PATH, V7_REPORT, log, peak_gb, read_json, require
from operating import recall_point


def write_findings(started: float, dev_ran: bool) -> None:
    cv = _required(CV_PATH)
    summary = read_json(SUMMARY_PATH)
    dev = read_json(DEV_REPORT) if dev_ran else None
    v7 = _required(V7_REPORT)
    text = _render(summary, cv, dev, v7, time.perf_counter() - started, dev_ran)
    FINDINGS_PATH.write_text(text, encoding="utf-8")
    log(f"wrote {FINDINGS_PATH.name}")


def _render(summary, cv, dev, v7, seconds: float, dev_ran: bool) -> str:
    lines = _setting_lines(summary, cv)
    lines.extend(_cv_lines(cv))
    lines.extend(_gap_lines(cv))
    if dev_ran:
        lines.extend(_dev_lines(dev, v7))
    else:
        lines.extend(["", "Development was not run. The time budget ran out after cross-validation."])
    lines.extend(_failure_lines(dev))
    lines.extend(["", f"wall_seconds={seconds:.1f}", f"peak_rss_gb={peak_gb():.2f}", "JSON: cell_summary.json, cv_control.json, cv_results.json, dev_report.json"])
    return "\n".join(lines) + "\n"


def _setting_lines(summary, cv) -> list[str]:
    recipe = cv["recipe"]
    slowest = None if summary is None else summary.get("slowest_window_seconds")
    return [
        "Overlay-symbol channels on the g2-cleaned stage-2 cells. Fit-split choice; development is one measurement.",
        "",
        "Cleaner: g2. Base cells are retrain7's cleaned-descriptor stage-1 plus frozen structure channels.",
        "Sampling matches retrain7: fit-manifest order, seed 42 plus that index, 2000 cells per class.",
        "Symbol channels, per stride-8 cell, windows 64, 128 and 256 px:",
        "closed-ink fraction (outer contour encloses a hole, bounding-box diameter 20 to 120 px),",
        "count of outer contours with circularity > 0.7 (bounding-box center inside the window;",
        "boxes at least 4 times longer than wide are skipped because circularity cannot exceed 0.7),",
        "large-ink fraction (component area > 400 px).",
        "The same three numbers are computed on the g2-cleaned query crop (ink is gray < 210).",
        "Drawing ink matches extract_ink_and_chroma: HSV value < 235 or channel spread > 20.",
        "Absolute differences between each window and the query are appended. 21 channels total.",
        f"Slowest drawing for the window features: {_num(slowest)} seconds.",
        "",
        "Stage-2 recipe, not re-searched: "
        f"learning_rate {recipe['learning_rate']}, max_leaf_nodes {recipe['max_leaf_nodes']}, "
        f"l2 {recipe['l2_regularization']}, max_iter {recipe['max_iter']}, "
        f"min_samples_leaf {recipe['min_samples_leaf']}, early_stopping false, seed {cv['seed']}.",
        "Hard negatives: reviewed negative cells with cleaned stage-1 cell_score >= 0.75.",
        "Their weight is multiplied by 3, then each document's negatives are scaled back to total weight 0.5.",
        "CV trains on every fit query outside the held-out document fold, generated CAD included.",
        "CV scores only real fit queries. Each sampled cell is weighted by known-cells/sampled-cells of its class.",
        "The fold score is document-macro IoU at the pooled-recall >= 0.95 threshold. Ties take higher precision.",
        "If macro IoU and precision both tie, the lower threshold remains. That score is on stride-8 cells, not native pixels.",
        "",
        "Development, when run, uses the symbol model (165 channels) inside the v7 pipeline:",
        "g2 descriptor, stage-1 crop model, symbol channels concatenated after the structure channels,",
        "then the w16 linear color gate at strength 0.6. Thresholds are not chosen on development.",
        "",
    ]


def _cv_lines(cv) -> list[str]:
    lines = [f"Selected variant: {cv['selected']}", "Fit CV means (real held-out queries, not development):"]
    for row in cv["ranked"]:
        lines.append(_cv_row(row))
    lines.append("")
    return lines


def _cv_row(row: dict) -> str:
    folds = ", ".join(_fold_text(fold) for fold in row["folds"])
    return (
        f"  {row['name']}: mean_macro={row['mean_macro_iou']:.6f} mean_precision={row['mean_precision']:.6f} "
        f"hard_fraction_real={row['hard_negative_fraction_real']:.4f} "
        f"hard_fraction_all={row['hard_negative_fraction_all']:.4f} "
        f"rows={row['rows']} per_class={row['per_class']} hard_weight={row['hard_weight']} "
        f"features={row['features']} folds=[{folds}]"
    )


def _fold_text(fold: dict) -> str:
    return (
        f"fold {fold['fold']} threshold={fold['threshold']} macro={fold['macro_iou']:.6f} "
        f"precision={fold['precision']:.6f} recall={fold['recall']:.6f}"
    )


def _gap_lines(cv) -> list[str]:
    gaps = cv["control_gaps"]
    lines = [
        "Control versus retrain7 cleaned_hardneg_0.75 (absolute gaps): "
        f"mean_macro={gaps['mean_macro']:.8f} mean_precision={gaps['mean_precision']:.8f}"
    ]
    for key in sorted(gaps):
        if key.startswith("fold"):
            lines.append(f"  {key}={gaps[key]:.8f}")
    return lines


def _dev_lines(dev, v7) -> list[str]:
    require(dev is not None, "missing development report")
    ours_95 = dev["recall_ge_0_95"]
    ours_98 = dev["recall_ge_0_98"]
    v7_95 = recall_point(v7["points"], 0.95)
    v7_98 = recall_point(v7["points"], 0.98)
    v7_best = v7["selections"]["best_macro_iou"]
    lines = ["", "Development, 55 queries. v7 numbers are read from combo/dev_report_g2.json with the same threshold rule."]
    lines.append(_point_line("v7 best macro IoU", v7_best))
    lines.append(_point_line("v7 recall>=0.95", v7_95))
    lines.append(_point_line("v7 recall>=0.98", v7_98))
    lines.append(_point_line("symbols best macro IoU", dev["selections"]["best_macro_iou"]))
    lines.append(_point_line("symbols recall>=0.95", ours_95))
    lines.append(_point_line("symbols recall>=0.98", ours_98))
    lines.extend(_delta_lines(dev, v7_best, v7_95, v7_98))
    lines.extend(_fp_lines(dev["document_fp"]))
    return lines


def _delta_lines(dev, v7_best, v7_95, v7_98) -> list[str]:
    best = dev["selections"]["best_macro_iou"]
    ours_95 = dev["recall_ge_0_95"]
    ours_98 = dev["recall_ge_0_98"]
    return [
        f"best macro IoU minus v7: {_delta(best, v7_best, 'document_macro_iou')}",
        f"recall>=0.95 macro IoU minus v7: {_delta(ours_95, v7_95, 'document_macro_iou')}",
        f"recall>=0.95 precision minus v7: {_delta(ours_95, v7_95, 'global_precision')}",
        f"recall>=0.95 FPR minus v7: {_delta(ours_95, v7_95, 'reviewed_negative_fpr')}",
        f"recall>=0.98 macro IoU minus v7: {_delta(ours_98, v7_98, 'document_macro_iou')}",
    ]


def _fp_lines(block: dict) -> list[str]:
    return [
        f"False positives on {block['document_id']}:",
        f"  v7 at its recall>=0.95 threshold {block['v7_threshold']}: fp={block['v7_fp']}",
        f"  symbols at its recall>=0.95 threshold {block['ours_threshold']}: fp={block['ours_fp']}",
        f"  symbols at v7's threshold {block['v7_threshold']}: fp={block['ours_fp_at_v7_threshold']}",
    ]


def _failure_lines(dev) -> list[str]:
    lines = ["", "Failed:"]
    if dev is None:
        lines.append("development scores were not measured")
        return lines
    lines.extend(_goal_lines(dev))
    return lines


def _goal_lines(dev) -> list[str]:
    best = dev["selections"]["best_macro_iou"]
    at_98 = dev["selections"]["best_precision_at_recall_ge_0_98"]
    point = dev["recall_ge_0_95"]
    lines = []
    if best["document_macro_iou"] < 0.95:
        lines.append(f"best document-macro IoU is {_num(best['document_macro_iou'])} at threshold {best['threshold']}, below 0.95")
    if at_98 is None or (at_98["global_precision"] or 0.0) < 0.98:
        lines.append(f"best precision at recall>=0.98 is {_num(None if at_98 is None else at_98['global_precision'])}, below 0.98")
    if point is None or (point["reviewed_negative_fpr"] or 1.0) > 0.05:
        lines.append(f"at the recall>=0.95 rule point FPR is {_num(None if point is None else point['reviewed_negative_fpr'])}")
    lines.append(f"Harness all_goals_met: {dev['selections']['all_goals_met']}")
    return lines


def _point_line(label: str, point: dict) -> str:
    return (
        f"{label} threshold={point['threshold']} macro={_num(point['document_macro_iou'])} "
        f"recall={_num(point['global_recall'])} precision={_num(point['global_precision'])} "
        f"fpr={_num(point['reviewed_negative_fpr'])} fp={int(point['fp'])}"
    )


def _delta(ours, reference, key: str) -> str:
    if ours is None or reference is None or ours.get(key) is None or reference.get(key) is None:
        return "unavailable"
    return f"{float(ours[key]) - float(reference[key]):+.6f}"


def _num(value) -> str:
    if value is None:
        return "none"
    return f"{float(value):.6f}"


def _required(path):
    payload = read_json(path)
    require(payload is not None, f"missing {path.name}")
    return payload
