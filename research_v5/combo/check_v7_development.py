"""Prove inference_v7's full-resolution scores match the sparse development scores.

Labels are read only here, to choose the reviewed pixels and to count decisions.
"""
from __future__ import annotations

import bootstrap  # noqa: F401

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from bootstrap import DEV_CACHE, HERE, STRUCTURE_MODEL, current_gb, log, peak_gb
from devharness import DEVELOPMENT_MANIFEST, development_queries, report, request_view
from evaluate_dev import load_original
from inference_v7 import load_stage2, predict_drawing
from score_dev import _read

EQUIVALENCE_PATH = HERE / "equivalence.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cleaner", required=True)
    parser.add_argument("--stage2", type=Path, default=STRUCTURE_MODEL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    summary = run_check(args.cleaner, args.stage2)
    summary["seconds"] = time.perf_counter() - started
    summary["peak_gb"] = peak_gb()
    summary["current_gb"] = current_gb()
    EQUIVALENCE_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(json.dumps({key: summary[key] for key in ("identical_counts", "max_abs_gap", "queries")}))
    sys.exit(0 if summary["identical_counts"] else 1)


def run_check(cleaner: str, stage2_path: Path) -> dict:
    stage1, _metadata = load_original()
    stage2 = load_stage2(stage2_path)
    native = _sample_native(cleaner, stage1, stage2)
    sparse = _sparse(cleaner, native)
    gap = _max_gap(native, sparse)
    native_report = report(f"v7_native_{cleaner}", native)
    sparse_report = report(f"v7_sparse_{cleaner}", sparse)
    mismatched = _count_mismatches(native_report["points"], sparse_report["points"])
    return {
        "cleaner": cleaner,
        "stage2": str(stage2_path),
        "queries": len(native),
        "reviewed_pixels": int(sum(len(value) for value in native.values())),
        "max_abs_gap": gap,
        "scores_identical": gap == 0.0,
        "threshold_count_mismatches": mismatched,
        "identical_counts": not mismatched,
    }


def _sample_native(cleaner: str, stage1, stage2) -> dict[str, np.ndarray]:
    import devharness

    queries = {query.id: query for query in development_queries()}
    rows = json.loads(DEVELOPMENT_MANIFEST.read_text(encoding="utf-8"))["examples"]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["image"]].append(request_view(row))
    sampled: dict[str, np.ndarray] = {}

    def emit(request: dict, gated: np.ndarray) -> None:
        query = queries[request["id"]]
        sampled[query.id] = np.asarray(gated[query.y, query.x], dtype=np.float32)

    for requests in grouped.values():
        predict_drawing(requests, stage1, stage2, cleaner, devharness.DATA_ROOT, emit)
        log(f"equivalence sampled {requests[0]['image']} rss_gb={current_gb():.2f}")
    return sampled


def _sparse(cleaner: str, native: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    scores = {}
    for query_id in native:
        values = _read(cleaner, query_id)
        if values is None:
            raise RuntimeError(f"missing sparse scores for {query_id} in {DEV_CACHE / cleaner}")
        scores[query_id] = values
    return scores


def _max_gap(native: dict[str, np.ndarray], sparse: dict[str, np.ndarray]) -> float:
    worst = 0.0
    for query_id, sampled in native.items():
        reference = sparse[query_id]
        if sampled.shape != reference.shape:
            raise RuntimeError(f"{query_id} native {sampled.shape} != sparse {reference.shape}")
        worst = max(worst, float(np.max(np.abs(sampled.astype(np.float64) - reference.astype(np.float64)))))
    return worst


def _count_mismatches(native_points: list[dict], sparse_points: list[dict]) -> list:
    if len(native_points) != len(sparse_points):
        return [("length", len(native_points), len(sparse_points))]
    return [
        (mine["threshold"], key)
        for mine, theirs in zip(native_points, sparse_points)
        for key in ("tp", "fp", "fn", "tn")
        if mine[key] != theirs[key]
    ]


if __name__ == "__main__":
    main()
