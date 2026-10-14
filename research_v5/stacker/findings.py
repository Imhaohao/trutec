"""Write findings.txt from artifacts this run actually produced."""
from __future__ import annotations

import json
from pathlib import Path

from constants import STACKER, settings_snapshot


def write_findings(extra: list[str] | None = None) -> None:
    lines = [
        "Supervised stacking: a second-stage context classifier on stride-8 cells.",
        "",
        "The first stage is the crop-descriptor texture HistGradientBoosting model.",
        "Fit-split evidence is cross-fit in 4 document folds so a fit query is scored",
        "by a model that did not see its document. Development queries use the original",
        "model at models/models/texture-hgb-crop-v1.joblib (the brief's texture-hgb-crop-v1).",
        "The query descriptor is the exact crop median at stride 4, matching training.",
        "Local descriptors are the same 46 channels sampled on a stride-8 lattice.",
        "",
        "Settings",
        json.dumps(settings_snapshot(), indent=2),
        "",
    ]
    _section(lines, "Folds", STACKER / "folds.json", _folds_text)
    _section(lines, "Stage 1 cross-fit", STACKER / "stage1_summary.json", _stage1_text)
    _section(lines, "Stage 2 cache", STACKER / "stage2_summary.json", _pretty)
    _section(lines, "Stage 2 cross-validation", STACKER / "cv_results.json", _pretty)
    _section(lines, "Development harness", STACKER / "dev_report.json", _harness_text)
    _section(lines, "HGB cache baseline", STACKER / "baseline_hgb.json", _harness_text)
    _section(lines, "Per-query recall", STACKER / "per_query_recall.json", _pretty)
    if (STACKER / "error.txt").is_file():
        lines.extend(["", "Error", (STACKER / "error.txt").read_text(encoding="utf-8")])
    if extra:
        lines.extend(["", *extra])
    (STACKER / "findings.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _section(lines: list[str], title: str, path: Path, render) -> None:
    lines.append(title)
    if not path.is_file():
        lines.extend(["not finished", ""])
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    lines.extend(render(payload))
    lines.append("")


def _folds_text(payload: dict) -> list[str]:
    lines = [payload["rule"], json.dumps(payload["summary"])]
    for fold, documents in payload["documents"].items():
        lines.append(f"fold {fold} documents ({len(documents)}): {', '.join(documents)}")
    return lines


def _stage1_text(payload: dict) -> list[str]:
    return [json.dumps(payload, indent=2)]


def _pretty(payload) -> list[str]:
    return [json.dumps(payload, indent=2)]


def _harness_text(payload: dict) -> list[str]:
    selections = payload.get("selections", payload)
    return [json.dumps(selections, indent=2)]
