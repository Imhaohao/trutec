"""Write findings.txt from the JSON this run saved."""
from __future__ import annotations

import time

import runtime

runtime.activate()

from operating import recall_point
from runtime import CV_PATH, DEV_REPORT, FINDINGS_PATH, SUMMARY_PATH, V7_REPORT, log, peak_gb, read_json, require


def write_findings(started: float) -> None:
    summary = _required(SUMMARY_PATH)
    cv = _required(CV_PATH)
    dev = _required(DEV_REPORT)
    v7 = _required(V7_REPORT)
    text = _render(summary, cv, dev, v7, time.perf_counter() - started)
    FINDINGS_PATH.write_text(text, encoding="utf-8")
    log(f"wrote {FINDINGS_PATH.name}")


def _render(summary, cv, dev, v7, seconds: float) -> str:
    ours_95 = dev["recall_ge_0_95"]
    ours_98 = dev["recall_ge_0_98"]
    v7_95 = recall_point(v7["points"], 0.95)
    v7_98 = recall_point(v7["points"], 0.98)
    _require_rule(dev, ours_95, v7_95, v7_98)
    lines = _setting_lines(summary, cv)
    lines.extend(_cv_lines(cv))
    lines.extend(_dev_lines(dev, ours_95, ours_98, v7_best_points(v7, v7_95, v7_98)))
    lines.extend(_failure_lines(dev))
    lines.append("")
    lines.append(f"wall_seconds={seconds:.1f}")
    lines.append(f"peak_rss_gb={peak_gb():.2f}")
    lines.append("JSON: cell_summary.json, cv_results.json, dev_report.json")
    return "\n".join(lines) + "\n"


def _setting_lines(summary, cv) -> list[str]:
    recipe = cv["signature"]["recipe"]
    return [
        "Retrain stage 2 on g2-cleaned stage-1 grids. Fit-split choice; development is one measurement.",
        "",
        f"Cleaner: {cv['signature']['cleaner']}. Stage-1 grids use the four cross-fitted stacker fold models",
        "(the model that did not train on that document) and querycrop2.guards.descriptor_for.",
        "Sampling matches structure/cells.py: same fit-manifest order, same seed 42 plus that index,",
        "2000 cells per class. The double variant asks for 4000. Structure channels are the frozen ones;",
        f"max absolute gap versus the structure cell cache is {summary['max_structure_abs_gap']:.6g}.",
        f"Queries whose cleaned cell_score moved: {summary['queries_cell_score_changed']} of {summary['queries']}.",
        f"Max absolute cell_score gap versus the uncleaned cache: {summary['max_cell_score_abs_gap']:.6g}.",
        f"Max absolute gap on the 75 stacker channels: {summary['max_stacker_abs_gap']:.6g}.",
        f"Base cells {summary['cells_base']} (pos {summary['positive_base']}, neg {summary['negative_base']}).",
        f"Double cells {summary['cells_double']} (pos {summary['positive_double']}, neg {summary['negative_double']}).",
        f"Real fit queries in the cache: {summary['real_queries']}.",
        "",
        "Stage-2 recipe, not re-searched: "
        f"learning_rate {recipe['learning_rate']}, max_leaf_nodes {recipe['max_leaf_nodes']}, "
        f"l2 {recipe['l2_regularization']}, max_iter {recipe['max_iter']}, "
        f"min_samples_leaf {recipe['min_samples_leaf']}, early_stopping false, seed 42.",
        "Hard negatives: reviewed negative cells with cleaned stage-1 cell_score >= 0.75.",
        "Their weight is multiplied by 3 or by 5, then each document's negatives are scaled back to total weight 0.5.",
        "CV trains on every fit query outside the held-out document fold, generated CAD included.",
        "CV scores only real fit queries. Each sampled cell is weighted by known-cells/sampled-cells of its class.",
        "The fold score is document-macro IoU at the pooled-recall >= 0.95 threshold. Ties take higher precision.",
        "If macro IoU and precision both tie, the lower threshold remains. That score is on stride-8 cells, not native pixels.",
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
        f"rows={row['rows']} per_class={row['per_class']} hard_weight={row['hard_weight']} folds=[{folds}]"
    )


def _fold_text(fold: dict) -> str:
    if fold.get("failed"):
        return f"fold {fold['fold']} failed"
    return (
        f"fold {fold['fold']} threshold={_num(fold.get('threshold'), 2)} "
        f"macro={_num(fold.get('macro_iou'))} precision={_num(fold.get('precision'))} "
        f"recall={_num(fold.get('recall'))}"
    )


def _dev_lines(dev, ours_95, ours_98, published) -> list[str]:
    best = dev["selections"]["best_macro_iou"]
    v7_best, v7_95, v7_98 = published
    return [
        "Development, 55 queries. v7 pipeline with this stage-2 model instead of hardneg_0.75.",
        "Color gate window 16, linear, strength 0.6, after the stride-8 cell lookup.",
        "v7 numbers are read from research_v5/combo/dev_report_g2.json and the same threshold rule.",
        "",
        "v7",
        f"  best macro IoU {_point(v7_best)}",
        f"  recall>=0.95 {_point(v7_95)}",
        f"  recall>=0.98 {_point(v7_98)}",
        dev["variant"],
        f"  best macro IoU {_point(best)}",
        f"  recall>=0.95 {_show(ours_95)}",
        f"  recall>=0.98 {_show(ours_98)}",
        "",
        "Rule: maximum development macro IoU with development recall >= 0.95, ties by fewer FP.",
        _rule_line(dev["variant"], ours_95),
        f"v7 rule threshold: {v7_95['threshold']:.2f}",
        "",
        _delta_line("best macro IoU", best, v7_best),
        _delta_line("recall>=0.95 macro IoU", ours_95, v7_95),
        _delta_line("recall>=0.98 macro IoU", ours_98, v7_98),
        "",
    ]


def _failure_lines(dev) -> list[str]:
    points = dev["points"]
    lines = ["Failed:"]
    lines.extend(_goal_lines(points))
    if dev["selections"]["all_goals_met"]:
        lines.append("Harness all_goals_met (macro 0.95, recall 0.98, precision 0.95, FPR 0.05): true")
    else:
        lines.append("Harness all_goals_met (macro 0.95, recall 0.98, precision 0.95, FPR 0.05): false")
    if _user_goal(points):
        lines.append("User goal (macro 0.95, recall >= 0.98, precision >= 0.98, FPR <= 0.05) at one threshold: true")
    else:
        lines.append("User goal (macro 0.95, recall >= 0.98, precision >= 0.98, FPR <= 0.05) at one threshold: false")
    return lines


def _goal_lines(points: list[dict]) -> list[str]:
    return [line for line in (_macro_line(points), _precision_line(points), _fpr_line(points)) if line]


def _macro_line(points: list[dict]) -> str:
    best = max(points, key=lambda point: point["document_macro_iou"] or 0.0)
    macro = best["document_macro_iou"]
    if macro >= 0.95:
        return ""
    return f"best document-macro IoU is {_num(macro)} at threshold {_num(best['threshold'], 2)}, below 0.95"


def _precision_line(points: list[dict]) -> str:
    eligible = [point for point in points if (point["global_recall"] or 0.0) >= 0.98]
    if not eligible:
        return "no threshold reaches recall 0.98"
    best = max(eligible, key=lambda point: point["global_precision"] or 0.0)
    if (best["global_precision"] or 0.0) >= 0.98:
        return ""
    return (
        f"best precision at recall>=0.98 is {_num(best['global_precision'])} "
        f"(macro {_num(best['document_macro_iou'])}, FPR {_num(best['reviewed_negative_fpr'])}), below 0.98"
    )


def _fpr_line(points: list[dict]) -> str:
    rule = recall_point(points, 0.95)
    if rule is None:
        return "no threshold reaches recall 0.95, so the rule picks no threshold"
    if _low_fpr(points):
        return ""
    return (
        f"no recall>=0.98 threshold has reviewed-negative FPR <= 0.05; "
        f"at the recall>=0.95 rule point FPR is {_num(rule['reviewed_negative_fpr'])}"
    )


def _low_fpr(points: list[dict]) -> bool:
    return any(_recall_fpr(point) for point in points)


def _recall_fpr(point: dict) -> bool:
    return (point["global_recall"] or 0.0) >= 0.98 and (point["reviewed_negative_fpr"] or 1.0) <= 0.05


def _user_goal(points: list[dict]) -> bool:
    for point in points:
        if _meets_user_goal(point):
            return True
    return False


def _meets_user_goal(point: dict) -> bool:
    return bool(
        (point["document_macro_iou"] or 0.0) >= 0.95
        and (point["global_recall"] or 0.0) >= 0.98
        and (point["global_precision"] or 0.0) >= 0.98
        and (point["reviewed_negative_fpr"] or 1.0) <= 0.05
    )


def _delta_line(label: str, ours, reference) -> str:
    if ours is None or reference is None:
        return f"{label} minus v7: unavailable"
    delta = float(ours["document_macro_iou"]) - float(reference["document_macro_iou"])
    return f"{label} minus v7: {delta:+.6f}"


def _show(point) -> str:
    if point is None:
        return "none"
    return _point(point)


def _rule_line(variant: str, point) -> str:
    if point is None:
        return f"Threshold the rule picks for {variant}: none"
    return f"Threshold the rule picks for {variant}: {point['threshold']:.2f}. At that threshold: {_point(point)}"


def v7_best_points(v7, v7_95, v7_98):
    return v7["selections"]["best_macro_iou"], v7_95, v7_98


def _point(point: dict) -> str:
    return (
        f"threshold={float(point['threshold']):.2f} macro={_num(point['document_macro_iou'])} "
        f"recall={_num(point['global_recall'])} precision={_num(point['global_precision'])} "
        f"fpr={_num(point['reviewed_negative_fpr'])} fp={int(point['fp'])}"
    )


def _require_rule(dev, ours_95, v7_95, v7_98) -> None:
    require(v7_95 is not None and v7_98 is not None, "v7 report has no recall point")
    fresh = recall_point(dev["points"], 0.95)
    if ours_95 is None:
        require(fresh is None, "stored rule point drifted")
        return
    require(fresh is not None, "stored rule point drifted")
    require(fresh["threshold"] == ours_95["threshold"], "stored rule threshold drifted")
    require(fresh["fp"] == ours_95["fp"], "stored rule fp drifted")


def _required(path):
    payload = read_json(path)
    require(payload is not None, f"missing {path.name}")
    return payload


def _num(value, digits: int = 6) -> str:
    if value is None:
        return "none"
    return f"{float(value):.{digits}f}"
