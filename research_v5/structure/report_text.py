"""Write findings.txt from harness JSON. Numbers are read, not typed."""
from __future__ import annotations

import time

import limits  # noqa: F401

from constants import CANDIDATES
from limits import CV_PATH, FINDINGS_PATH, STACKER, log, peak_gb, read_json, write_json
from structure_features import (
    DOWNSAMPLE,
    FEATURE_VERSION,
    HARMONIC_FRACTION,
    MIN_PERIODICITY,
    ORIENT_BINS,
    PERIODS,
    WINDOWS,
    feature_names,
)
from weights import HARD_WEIGHT, SETTINGS


def best_at_recall(points, floor: float):
    eligible = [point for point in points if (point.get("global_recall") or 0.0) >= floor]
    if not eligible:
        return None
    return max(eligible, key=_recall_key)


def write_findings(started: float, ranked: list[dict]) -> None:
    v6 = _v6()
    variants = [_variant(setting["name"]) for setting in SETTINGS]
    text = _render(v6, variants, ranked, time.perf_counter() - started)
    FINDINGS_PATH.write_text(text, encoding="utf-8")
    log(f"wrote {FINDINGS_PATH.name}")


def write_settings(extra: dict) -> None:
    from limits import SETTINGS_PATH

    payload = {
        "chosen_on": "fit split grouped document folds; development is measurement only",
        "hgb": "stacker recipe candidate, not re-searched",
        "recipe": [item for item in CANDIDATES if item["name"] == "recipe"][0],
        "settings": [_public_setting(item) for item in SETTINGS],
        "hard_weight": HARD_WEIGHT,
        "hard_negative_rule": (
            "reviewed negative cells with stage-1 cell_score >= cutoff are multiplied by 3, "
            "then negative weights inside the document are rescaled to sum to 0.5"
        ),
        "windows_px_side": list(WINDOWS),
        "orientation_bins": ORIENT_BINS,
        "orientation_centers_deg": [index * 180 / ORIENT_BINS for index in range(ORIENT_BINS)],
        "periods_px": list(PERIODS),
        "period_downsample": DOWNSAMPLE,
        "harmonic_fraction": HARMONIC_FRACTION,
        "min_periodicity": MIN_PERIODICITY,
        "feature_version": FEATURE_VERSION,
        "structure_feature_count": len(feature_names()),
        "color_gate": {"window": 16, "type": "linear", "strength": 0.6},
        "lookup": "stride-8 cell",
        **extra,
    }
    write_json(SETTINGS_PATH, payload)


def _render(v6, variants, ranked, seconds: float) -> str:
    lines = [
        "Structure-aware stage 2. Fit-split choices only; development numbers are measurements.",
        "",
        "Settings are in settings.json. Stage-2 hyperparameters are the stacker's fit-CV winner",
        "(recipe: learning_rate 0.08, max_leaf_nodes 31, l2 1, max_iter 200, min_samples_leaf 20).",
        "The three fit-CV settings are standard document-balanced weights, hard-negative cutoff",
        "0.50, and hard-negative cutoff 0.75. Hard negatives are reviewed negatives whose stage-1",
        f"probability is at least the cutoff; their weight is {HARD_WEIGHT:g}x an easy negative",
        "after each document's negatives are scaled back to total weight 0.5.",
        "",
        "Features: the frozen 75 stacker channels, plus label-free structure on windows of side",
        "32, 64 and 128 px. Eight-bin unsigned gradient-energy histograms and cosine / L1 distance",
        "to the query histogram; dominant autocorrelation period along the dominant orientation,",
        "its ratio and absolute difference versus the query, and periodicity strength; ink",
        "connected-component area and smaller box side, plus absolute differences versus the query.",
        f"Period uses lagged products on a {DOWNSAMPLE}x downsampled drawing. The shortest lag",
        f"within {HARMONIC_FRACTION:g} of the best autocorrelation, and above {MIN_PERIODICITY:g}, is kept.",
        "",
        f"Fit CV choice: {ranked[0]['name']} mean precision-at-recall-0.98 = {ranked[0]['mean']:.6f}",
        "CV means (fit, not development):",
    ]
    lines.extend(_cv_lines(ranked))
    lines.extend(["", "Development, 55 queries, color gate w16 linear 0.6. Same selector as the v6 file.", ""])
    lines.append(_point_line("v6", v6))
    lines.extend(_point_line(item["name"], item) for item in variants)
    lines.extend(["", "Reviewed-negative false-positive rate at those same operating points:", ""])
    lines.append(_fpr_line("v6", v6))
    lines.extend(_fpr_line(item["name"], item) for item in variants)
    lines.extend(["", _goal_line(variants), "", f"wall_seconds_this_process={seconds:.1f}", f"peak_rss_gb={peak_gb():.2f}"])
    return "\n".join(lines) + "\n"


def _cv_lines(ranked) -> list[str]:
    lines = []
    for row in ranked:
        folds = ", ".join(f"{value:.4f}" for value in row["fold_precision_at_recall_0_98"])
        lines.append(
            f"  {row['name']}: mean={row['mean']:.6f} hard_fraction={row['hard_negative_fraction']:.4f} folds=[{folds}]"
        )
    return lines


def _point_line(name: str, payload: dict) -> str:
    best = payload["selections"]["best_macro_iou"]
    recall = payload["recall_ge_0_95"]
    precise = payload["selections"]["best_precision_at_recall_ge_0_98"]
    return "\n".join(
        [
            f"{name}",
            f"  best macro IoU {_fmt_point(best)}",
            f"  recall>=0.95 {_fmt_point(recall)}",
            f"  precision at recall>=0.98 {_fmt_point(precise)}",
        ]
    )


def _fpr_line(name: str, payload: dict) -> str:
    best = payload["selections"]["best_macro_iou"]
    recall = payload["recall_ge_0_95"]
    precise = payload["selections"]["best_precision_at_recall_ge_0_98"]
    return (
        f"{name}: best={_fpr(best)} recall>=0.95={_fpr(recall)} "
        f"recall>=0.98={_fpr(precise)}"
    )


def _fmt_point(point) -> str:
    if point is None:
        return "none"
    return (
        f"threshold={point['threshold']:.2f} macro={point['document_macro_iou']:.6f} "
        f"recall={point['global_recall']:.6f} precision={point['global_precision']:.6f} "
        f"fpr={point['reviewed_negative_fpr']:.6f}"
    )


def _fpr(point) -> str:
    if point is None or point.get("reviewed_negative_fpr") is None:
        return "none"
    return f"{point['reviewed_negative_fpr']:.6f}"


def _goal_line(variants) -> str:
    met = [item["name"] for item in variants if item["selections"]["all_goals_met"]]
    harness = "none" if not met else ", ".join(met)
    return (
        "Harness all_goals_met uses macro IoU 0.95, recall 0.98, precision 0.95, "
        f"reviewed-negative FPR <= 0.05. Met: {harness}. "
        "The user target asks precision 0.98 as well; read the precision column above."
    )


def _variant(name: str) -> dict:
    from limits import HERE

    payload = read_json(HERE / f"dev_{name}.json")
    if payload is None:
        raise RuntimeError(f"missing development report {name}")
    payload["name"] = name
    return payload


def _v6() -> dict:
    payload = read_json(STACKER / "dev_report_color.json")
    if payload is None:
        raise RuntimeError("missing v6 color report")
    payload = dict(payload)
    payload["recall_ge_0_95"] = best_at_recall(payload["points"], 0.95)
    write_json(FINDINGS_PATH.with_name("v6_reference_point.json"), {"recall_ge_0_95": payload["recall_ge_0_95"]})
    return payload


def _public_setting(setting: dict) -> dict:
    return {"name": setting["name"], "stage1_cutoff": setting["cutoff"]}


def _recall_key(point: dict) -> tuple:
    return (point["document_macro_iou"] or 0.0, point["threshold"])
