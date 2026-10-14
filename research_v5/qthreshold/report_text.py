"""Write findings.txt from JSON this run already saved. No new measurements."""
from __future__ import annotations

import limits  # noqa: F401

from experiment import experiment_settings
from limits import CHOSEN_PATH, DEV_REPORT_PATH, FINDINGS_PATH, PUBLISHED_V6, read_json

GOALS = {
    "document_macro_iou": 0.95,
    "global_recall": 0.98,
    "global_precision": 0.98,
    "reviewed_negative_fpr": 0.05,
}


def write_findings(chosen: dict, development: dict) -> None:
    published = read_json(PUBLISHED_V6)
    lines = [
        "Per-query cutoff experiment for frozen v6.",
        "",
        "Settings (fixed before development scores were used for any choice)",
        _block(experiment_settings()),
        "",
        "Model selection",
        chosen["selection"],
        f"Chosen model: {chosen['chosen_model']} target={chosen['chosen_target']}",
        f"Trainable IoU targets: {_distribution(chosen['target_iou'])}",
        f"Trainable recall>=0.98 targets: {_distribution(chosen['target_recall98'])}",
        "",
        "Out-of-fold fit results (selection metric is oof macro IoU, not the clipped column)",
    ]
    lines.extend(_candidate_lines(chosen["candidates"]))
    lines.extend(
        [
            "",
            "Label-free rules on the fit split",
            chosen["fraction_selection"],
            f"Chosen fraction: {chosen['median_fraction']} ({chosen['median_fraction_name']})",
            _summary_line("otsu", chosen["otsu_fit"]),
        ]
    )
    lines.extend(_fraction_lines(chosen["fraction_ranked"]))
    lines.extend(["", "Published v6 development sweep (stacker/dev_report_color.json)"])
    lines.extend(_published_lines(published))
    lines.extend(["", "This run's v6 grid reproduction"])
    lines.append(_reproduction_line(development["reproduction"]))
    lines.extend(["", "Development results from devharness.summarize_decisions"])
    lines.extend(_variant_lines(development["variants"]))
    lines.extend(["", "Stated oracle from the experiment prompt (not recomputed)"])
    lines.append(_plain(development["stated_oracle_from_prompt"]))
    lines.extend(["", "What failed"])
    lines.extend(_failures(development))
    lines.append("")
    FINDINGS_PATH.write_text("\n".join(lines), encoding="utf-8")


def _candidate_lines(rows: list[dict]) -> list[str]:
    return [_candidate_line(row) for row in rows]


def _candidate_line(row: dict) -> str:
    mae = "none" if row["mae"] is None else f"{row['mae']:.6f}"
    corr = "none" if row["correlation"] is None else f"{row['correlation']:.6f}"
    return (
        f"{row['name']} oof {_metrics(row['oof'])} "
        f"clipped {_metrics(row['oof_clipped'])} mae={mae} corr={corr}"
    )


def _fraction_lines(rows: list[dict]) -> list[str]:
    return [_summary_line(row["name"], row["fit"]) for row in rows]


def _variant_lines(rows: list[dict]) -> list[str]:
    return [_variant_line(row) for row in rows]


def _variant_line(row: dict) -> str:
    marks = []
    if row.get("selected_on_fit"):
        marks.append("fit-selected")
    if row.get("selected_blend"):
        marks.append("fit-selected blend")
    mark = f" [{', '.join(marks)}]" if marks else ""
    extra = ""
    if "mae_vs_oracle_1001" in row:
        extra = f" mae_vs_oracle_1001={row['mae_vs_oracle_1001']:.6f}"
    return (
        f"{row['name']}{mark} {_metrics(row['summary'])} "
        f"cutoff_mean={row['cutoff_mean']:.6f} min={row['cutoff_min']:.6f} max={row['cutoff_max']:.6f}"
        f"{extra} missed={','.join(_missed(row['summary'])) or 'none'}"
    )


def _published_lines(published: dict) -> list[str]:
    selections = published["selections"]
    recall_95 = _best_at_recall(published["points"], 0.95)
    return [
        _summary_line("v6 best macro IoU", selections["best_macro_iou"]),
        _summary_line("v6 best precision at recall>=0.98", selections["best_precision_at_recall_ge_0_98"]),
        _summary_line("v6 best macro IoU at recall>=0.95", recall_95),
    ]


def _best_at_recall(points: list[dict], floor: float) -> dict:
    eligible = [point for point in points if (point["global_recall"] or 0.0) >= floor]
    return max(eligible, key=lambda point: point["document_macro_iou"] or 0.0)


def _reproduction_line(reproduction: dict) -> str:
    matched = "matched" if reproduction["counts_match"] else "DID NOT MATCH"
    return (
        f"threshold={reproduction['threshold']:.10f} counts {matched} published v6. "
        f"ours {_metrics(reproduction['ours'])} published {_metrics(reproduction['published'])}"
    )


def _failures(development: dict) -> list[str]:
    lines = []
    if development["reproduction"]["counts_match"]:
        lines.append("The rebuilt v6 grids match the published pixel counts at the global 0.44 threshold.")
    else:
        lines.append("The rebuilt v6 grids do not match the published pixel counts, so later comparisons are not on the same scores.")
    selected = [row for row in development["variants"] if row.get("selected_on_fit")]
    lines.extend(_miss_line(row) for row in selected)
    oracle = next(row for row in development["variants"] if row["name"] == "oracle_grid_1001")
    lines.append(
        "The label-using oracle is not deployable. Its development "
        f"{_metrics(oracle['summary'])} is the ceiling of per-query cutoffs on this grid, "
        "and it still misses precision 0.98."
    )
    return lines


def _miss_line(row: dict) -> str:
    missed = _missed(row["summary"])
    if not missed:
        return f"{row['name']} meets the stated goals."
    return f"{row['name']} misses {', '.join(missed)}. {_metrics(row['summary'])}"


def _missed(summary: dict) -> list[str]:
    missed = []
    if (summary["document_macro_iou"] or 0.0) < GOALS["document_macro_iou"]:
        missed.append("macro_iou<0.95")
    if (summary["global_recall"] or 0.0) < GOALS["global_recall"]:
        missed.append("recall<0.98")
    if (summary["global_precision"] or 0.0) < GOALS["global_precision"]:
        missed.append("precision<0.98")
    if (summary["reviewed_negative_fpr"] or 1.0) > GOALS["reviewed_negative_fpr"]:
        missed.append("fpr>0.05")
    return missed


def _summary_line(name: str, summary: dict) -> str:
    threshold = summary.get("threshold")
    prefix = f"{name} threshold={threshold:.10f} " if threshold is not None else f"{name} "
    return prefix + _metrics(summary)


def _metrics(summary: dict) -> str:
    return (
        f"macro_iou={_num(summary.get('document_macro_iou'))} "
        f"recall={_num(summary.get('global_recall'))} "
        f"precision={_num(summary.get('global_precision'))} "
        f"fpr={_num(summary.get('reviewed_negative_fpr'))} "
        f"tp={summary.get('tp')} fp={summary.get('fp')} fn={summary.get('fn')} tn={summary.get('tn')}"
    )


def _distribution(summary: dict) -> str:
    return f"count={summary['count']} mean={_num(summary['mean'])} std={_num(summary['std'])}"


def _num(value) -> str:
    if value is None:
        return "none"
    return f"{float(value):.10f}"


def _plain(payload: dict) -> str:
    return " ".join(f"{key}={value}" for key, value in payload.items())


def _block(payload: dict) -> str:
    return "\n".join(f"{key}: {value}" for key, value in payload.items())


def refresh_from_disk() -> None:
    write_findings(read_json(CHOSEN_PATH), read_json(DEV_REPORT_PATH))
