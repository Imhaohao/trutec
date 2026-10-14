"""Write findings.txt from JSON this run saved. Numbers are read, not typed."""
from __future__ import annotations

import bootstrap  # noqa: F401

import json
import re

from bootstrap import FINDINGS_PATH, QUERYCROP2_FINDINGS, STRUCTURE_REPORT, V6_REPORT, log
from operating import recall_point

SCORE_RUNTIME = FINDINGS_PATH.parent / "cache" / "score_runtime.json"
IDENTITY_RUNTIME = FINDINGS_PATH.parent / "cache" / "identity_runtime.json"
CHECK_RUNTIME = FINDINGS_PATH.parent / "equivalence.json"


def main() -> None:
    text = render()
    FINDINGS_PATH.write_text(text, encoding="utf-8")
    log(f"wrote {FINDINGS_PATH}")


def render() -> str:
    score = _read(SCORE_RUNTIME)
    check = _read(CHECK_RUNTIME)
    cleaner = score["cleaner"]
    lines = [
        "Candidate v7. Development numbers below were measured by this run.",
        "v6 and published structure rows were read from their saved development sweeps.",
        "",
        f"Cleaner: {cleaner}",
        f"Selection: {score['selection_note']}",
        _confirm_selection(cleaner),
        "",
        "Pipeline",
        "- Stage 1: original crop HistGradientBoosting, descriptor stride 4.",
        "- The cleaner replaces only that query descriptor. identity is the uncleaned crop median.",
        "- Stage 2: research_v5/structure/cache/models/hardneg_0.75.joblib.",
        "- Stage-2 channels: the 75 stacker channels, then the structure channels from",
        "  research_v5/structure/develop.py (windows 32/64/128, 8 orientation bins,",
        "  autocorrelation period on a 4x downsampled drawing, mark area and side).",
        "  Structure references use the original query box, not the cleaned crop.",
        "- Color gate: window 16, linear, strength 0.6, after the stride-8 cell lookup.",
        "- Threshold rule: maximum development document-macro IoU among thresholds with",
        "  pooled development recall >= 0.95. Ties choose fewer reviewed false positives.",
        "",
        _block("v6 published", _read(V6_REPORT)),
        _block("structure hardneg_0.75 published", _read(STRUCTURE_REPORT)),
        _block("identity + structure (this run)", _read(_report(score, "identity"))),
        _block(f"{cleaner} + structure (this run)", _read(_report(score, cleaner))),
        "",
        _threshold(cleaner, score),
        "",
        _gap(score),
        "",
        _equivalence(check),
        "",
        _runtime(score, check),
    ]
    return "\n".join(lines) + "\n"


def _block(title: str, payload: dict) -> str:
    points = payload["points"]
    precise = payload["selections"]["best_precision_at_recall_ge_0_98"]
    lines = [
        title,
        "  best macro IoU " + _fmt(payload["selections"]["best_macro_iou"]),
        "  recall>=0.95 " + _fmt(recall_point(points, 0.95)),
        "  recall>=0.98 " + _fmt(recall_point(points, 0.98)),
        "  precision at recall>=0.98 " + _fmt(precise),
    ]
    return "\n".join(lines)


def _threshold(cleaner: str, score: dict) -> str:
    payload = _read(_report(score, cleaner))
    point = recall_point(payload["points"], 0.95)
    if point is None:
        return f"Selected threshold for {cleaner}: none. No threshold reached recall >= 0.95."
    return (
        f"Selected threshold for {cleaner}: {point['threshold']:.2f}\n"
        "Rule: maximum development macro IoU with development recall >= 0.95, ties by fewer FP.\n"
        f"At that threshold: {_fmt(point)}"
    )


def _confirm_selection(cleaner: str) -> str:
    if not QUERYCROP2_FINDINGS.is_file():
        return (
            "querycrop2/findings.txt was absent when this file was written. "
            "The cleaner name was read from querycrop2/chosen.json, whose split field is fit."
        )
    text = QUERYCROP2_FINDINGS.read_text(encoding="utf-8")
    fit = "fit split" in text.lower() or "selected on fit" in text.lower()
    match = re.search(r"Selected variant:\s*(\S+)", text)
    named = match.group(1) if match else None
    if fit and named == cleaner:
        return f"querycrop2/findings.txt states Selected variant: {named} and that selection used the fit split."
    return f"querycrop2/findings.txt variant={named} fit_stated={fit}, which does not match cleaner {cleaner}."


def _gap(score: dict) -> str:
    gap = score.get("identity_structure_max_abs_gap")
    return (
        "Max absolute gap between this run's identity+structure development scores and "
        f"research_v5/structure/cache/dev_gated/hardneg_0.75: {gap}"
    )


def _equivalence(check: dict | None) -> str:
    if check is None:
        return "Native equivalence: not run."
    mismatches = check.get("threshold_count_mismatches") or []
    shown = mismatches[:8]
    return "\n".join(
        [
            f"Native equivalence identical_counts={check.get('identical_counts')}",
            f"queries={check.get('queries')} reviewed_pixels={check.get('reviewed_pixels')}",
            f"max_abs_gap={check.get('max_abs_gap')}",
            f"threshold_count_mismatches={shown}",
            f"check_seconds={check.get('seconds')} check_peak_gb={check.get('peak_gb')}",
        ]
    )


def _runtime(score: dict, check: dict | None) -> str:
    failed = _failed(score, check)
    identity = _identity_runtime(score)
    check_seconds = 0.0 if check is None or check.get("seconds") is None else float(check["seconds"])
    score_seconds = float(score["seconds"]) + identity[0]
    peaks = [float(score["peak_gb"]), identity[1]]
    if check is not None and check.get("peak_gb") is not None:
        peaks.append(float(check["peak_gb"]))
    return "\n".join(
        [
            f"score_seconds={score_seconds:.1f} score_peak_gb={max(float(score['peak_gb']), identity[1]):.2f}",
            f"total_seconds={score_seconds + check_seconds:.1f} peak_gb={max(peaks):.2f}",
            f"queries={score.get('queries')}",
            f"Failed: {failed}",
        ]
    )


def _identity_runtime(score: dict) -> tuple[float, float]:
    if score.get("cleaner") == "identity":
        return 0.0, float(score["peak_gb"])
    extra = _read(IDENTITY_RUNTIME)
    if extra is None:
        return 0.0, 0.0
    return float(extra["seconds"]), float(extra["peak_gb"])


def _failed(score: dict, check: dict | None) -> str:
    problems = []
    if score.get("identity_structure_max_abs_gap") not in (0, 0.0):
        problems.append("identity control does not match the published structure scores")
    if check is None:
        problems.append("native equivalence was not run")
    elif not check.get("identical_counts"):
        problems.append("native and sparse development counts differ")
    elif check.get("max_abs_gap") not in (0, 0.0):
        problems.append("native and sparse scores are not identical")
    if float(score["peak_gb"]) >= 10.0 or (check is not None and float(check.get("peak_gb") or 0) >= 10.0):
        problems.append("peak memory reached 10 GB")
    if "does not match" in _confirm_selection(str(score.get("cleaner"))):
        problems.append("querycrop2 findings do not name this cleaner as the fit selection")
    if not problems:
        return "none"
    return "; ".join(problems)


def _fmt(point) -> str:
    if point is None:
        return "none"
    return (
        f"threshold={point['threshold']:.2f} macro={point['document_macro_iou']:.6f} "
        f"recall={point['global_recall']:.6f} precision={point['global_precision']:.6f} "
        f"fpr={point['reviewed_negative_fpr']:.6f} fp={point['fp']}"
    )


def _report(score: dict, name: str):
    return FINDINGS_PATH.parent / f"dev_report_{name}.json"


def _read(path):
    if path is None or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
