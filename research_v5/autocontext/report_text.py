"""Write findings.txt from JSON artifacts. No numbers are computed from labels here."""
from __future__ import annotations

import json

import limits  # noqa: F401

from constants import FOLD_COUNT
from limits import (
    CACHE,
    CV_RESULTS,
    HERE,
    IMAGE_LOG,
    PROGRESS_PATH,
    SETTINGS_PATH,
    STAGE2_FOLD_CACHE,
    V6_COLOR_REPORT,
    log,
    read_json,
)

RAW_REPORT = HERE / "dev_report.json"
GATED_REPORT = HERE / "dev_report_color.json"
FINDINGS = HERE / "findings.txt"


def write_findings() -> None:
    lines: list[str] = []
    lines.append("Auto-context stage 3 on the frozen v6 stacker.")
    lines.append(
        "Stage 3 reads cross-fitted stage-2 score maps in square windows so a cell "
        "can be raised or lowered by what stage 2 thinks of its neighbourhood."
    )
    lines.append("Hyperparameters were chosen by grouped cross-validation on fit folds only.")
    lines.append("")
    _section(lines, "Settings", read_json(SETTINGS_PATH))
    _section(lines, "Stage 2 cross-fit", _fold_records())
    _section(lines, "Stage 3 cell cache", read_json(CACHE / "cell_summary.json"))
    _images(lines)
    _section(lines, "Stage 3 cross-validation", read_json(CV_RESULTS))
    _section(lines, "Stage 3 final fit", read_json(CACHE / "stage3_fit.json"))
    _development(lines)
    _section(lines, "Runtime", read_json(PROGRESS_PATH))
    _error(lines)
    FINDINGS.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"wrote {FINDINGS}")


def best_at_recall(report: dict | None, floor: float):
    if report is None:
        return None
    eligible = [point for point in report["points"] if (point["global_recall"] or 0.0) >= floor]
    if not eligible:
        return None
    return max(eligible, key=_iou_key)


def _development(lines: list[str]) -> None:
    raw = read_json(RAW_REPORT)
    gated = read_json(GATED_REPORT)
    v6 = read_json(V6_COLOR_REPORT)
    lines.append("Development")
    lines.append("The threshold sweep is development-selected. The stage-3 hyperparameters are not.")
    _report_block(lines, "stage 3, no color gate", raw)
    _report_block(lines, "stage 3 x color gate w16 linear 0.6", gated)
    _report_block(lines, "v6 stage 2 x color gate, from stacker/dev_report_color.json", v6)
    _compare(lines, gated, v6)
    lines.append("")


def _report_block(lines: list[str], title: str, report) -> None:
    lines.append(title)
    if report is None:
        lines.append("not recorded")
        return
    lines.append("best macro IoU: " + _fmt(report["selections"]["best_macro_iou"]))
    lines.append("best macro IoU at recall >= 0.95: " + _fmt(best_at_recall(report, 0.95)))
    held = report["selections"]["best_precision_at_recall_ge_0_98"]
    lines.append("best precision at recall >= 0.98: " + _fmt(held))
    lines.append("all_goals_met: " + str(report["selections"]["all_goals_met"]))


def _compare(lines: list[str], gated, v6) -> None:
    lines.append("Comparison")
    if gated is None or v6 is None:
        lines.append("No comparison: a development report is missing.")
        return
    pairs = (
        ("best macro IoU", gated["selections"]["best_macro_iou"], v6["selections"]["best_macro_iou"], "document_macro_iou"),
        ("macro IoU at recall >= 0.95", best_at_recall(gated, 0.95), best_at_recall(v6, 0.95), "document_macro_iou"),
        (
            "precision at recall >= 0.98",
            gated["selections"]["best_precision_at_recall_ge_0_98"],
            v6["selections"]["best_precision_at_recall_ge_0_98"],
            "global_precision",
        ),
    )
    for label, ours, theirs, key in pairs:
        lines.append(f"stage 3 x color versus v6, {label}: {_relation(ours, theirs, key)}.")


def _images(lines: list[str]) -> None:
    latest = _latest_images()
    lines.append("Per-image stage-3 cell build")
    if not latest:
        lines.append("not recorded")
        lines.append("")
        return
    seconds = [float(row["seconds"]) for row in latest.values()]
    lines.append(
        f"images={len(seconds)} sum_seconds={sum(seconds):.3f} max_seconds={max(seconds):.3f}"
    )
    lines.append("")


def _fold_records() -> list:
    return [read_json(STAGE2_FOLD_CACHE / f"fold_{fold}.json") for fold in range(FOLD_COUNT)]


def _latest_images() -> dict:
    latest = {}
    if not IMAGE_LOG.is_file():
        return latest
    for line in IMAGE_LOG.read_text(encoding="utf-8").splitlines():
        _keep_image(latest, line)
    return latest


def _keep_image(latest: dict, line: str) -> None:
    if not line.strip():
        return
    row = json.loads(line)
    latest[row["image"]] = row


def _error(lines: list[str]) -> None:
    path = HERE / "error.txt"
    lines.append("Errors")
    if not path.is_file():
        lines.append("none")
        return
    lines.append(path.read_text(encoding="utf-8"))


def _section(lines: list[str], title: str, payload) -> None:
    lines.append(title)
    if payload is None:
        lines.append("not recorded")
    else:
        lines.append(json.dumps(payload, indent=2))
    lines.append("")


def _fmt(point) -> str:
    if not point:
        return "none"
    return (
        f"macro_iou={point['document_macro_iou']:.10f} "
        f"recall={point['global_recall']:.10f} "
        f"precision={point['global_precision']:.10f} "
        f"threshold={point['threshold']}"
    )


def _relation(ours, theirs, key: str) -> str:
    if ours is None or theirs is None:
        return "not comparable"
    left = float(ours[key])
    right = float(theirs[key])
    if left > right:
        return f"{left:.10f} higher than v6 {right:.10f}"
    if left < right:
        return f"{left:.10f} lower than v6 {right:.10f}"
    return f"{left:.10f} equal to v6"


def _iou_key(point: dict) -> tuple:
    return (point["document_macro_iou"] or 0.0, point["global_precision"] or 0.0)
