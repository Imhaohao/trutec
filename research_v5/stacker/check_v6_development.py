"""Prove the native v6 path reproduces the sparse development scores at every reviewed pixel.

Labels are read only by the comparison at the end (through devharness).
"""
from __future__ import annotations

import bootstrap  # noqa: F401

import json
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np

import devharness
from evaluate_dev import load_original
from inference_v6 import DEFAULT_STAGE2, predict_drawing

HERE = Path(__file__).resolve().parent
SPARSE_SCORES = HERE / "cache" / "dev_scores"
SPARSE_REPORT = HERE / "dev_report_color.json"


def main() -> None:
    stage1, _metadata = load_original()
    stage2 = joblib.load(DEFAULT_STAGE2)["model"]
    queries = {query.id: query for query in devharness.development_queries()}
    rows = json.loads(devharness.DEVELOPMENT_MANIFEST.read_text(encoding="utf-8"))["examples"]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["image"]].append(devharness.request_view(row))
    native_at_reviewed: dict[str, np.ndarray] = {}

    def emit(request: dict, gated: np.ndarray) -> None:
        query = queries[request["id"]]
        native_at_reviewed[query.id] = gated[query.y, query.x]

    for requests in grouped.values():
        predict_drawing(requests, stage1, stage2, devharness.DATA_ROOT, emit)
    sparse_stage2_max_gap = _ungated_gap(native_at_reviewed)
    report = devharness.report("v6_native_check", native_at_reviewed)
    expected = json.loads(SPARSE_REPORT.read_text(encoding="utf-8"))["points"]
    mismatched = [
        (mine["threshold"], key)
        for mine, theirs in zip(report["points"], expected)
        for key in ("tp", "fp", "fn", "tn")
        if mine[key] != theirs[key]
    ]
    summary = {
        "queries": len(native_at_reviewed),
        "reviewed_pixels": int(sum(len(v) for v in native_at_reviewed.values())),
        "stage2_only_upper_bound_gap": sparse_stage2_max_gap,
        "threshold_count_mismatches": mismatched,
        "identical_to_sparse_color_report": not mismatched,
    }
    (HERE / "v6_native_check.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    sys.exit(0 if not mismatched else 1)


def _ungated_gap(native_at_reviewed: dict[str, np.ndarray]) -> float:
    """Gated scores can only be <= the sparse stage-2 score; report the largest violation."""
    worst = 0.0
    for query_id, gated in native_at_reviewed.items():
        sparse = np.load(SPARSE_SCORES / f"{query_id}.npy")
        worst = max(worst, float(np.max(gated - sparse)))
    return worst


if __name__ == "__main__":
    main()
