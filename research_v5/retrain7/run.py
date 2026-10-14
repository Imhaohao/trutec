"""Build cleaned stage-2 cells, choose one variant on real fit queries, score development once."""
from __future__ import annotations

import argparse
import json
import time
import traceback
import unittest

import runtime

runtime.activate()

from build_cells import build_all
from devharness import fit_rows
from findings_writer import write_findings
from fit_models import VARIANTS, ensure_cv, ensure_model
from folds import load_document_folds
from runtime import (
    CLEANER,
    FINDINGS_PATH,
    FOLDS_PATH,
    FIT_REAL,
    HERE,
    SETTINGS_PATH,
    SUMMARY_PATH,
    guard_memory,
    log,
    peak_gb,
    require,
    write_json,
)
from score_dev import score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit-images", type=int, default=0)
    args = parser.parse_args()
    started = time.perf_counter()
    try:
        execute(started, args.limit_images)
    except Exception:
        (HERE / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def execute(started: float, limit: int) -> None:
    error = HERE / "error.txt"
    if error.is_file():
        error.unlink()
    _tests()
    rows, folds, real_ids = _inputs()
    _write_settings(rows, folds)
    summary = build_all(rows, folds, real_ids, limit)
    if limit:
        log(f"smoke stop seconds={time.perf_counter() - started:.1f} peak_gb={peak_gb():.2f}")
        return
    write_json(SUMMARY_PATH, summary)
    guard_memory()
    ranked = ensure_cv(rows, folds, real_ids)
    guard_memory()
    model = ensure_model(rows, folds, ranked[0]["name"])
    guard_memory()
    score(model, ranked[0]["name"])
    write_findings(started)
    log(f"done seconds={time.perf_counter() - started:.1f} peak_gb={peak_gb():.2f} findings={FINDINGS_PATH.name}")


def _inputs():
    rows = fit_rows(real_only=False)
    real_ids = _real_ids()
    folds = load_document_folds(FOLDS_PATH)
    require(len(rows) == 357, f"fit queries {len(rows)}")
    require(len(real_ids) == 137, f"real queries {len(real_ids)}")
    require(real_ids == {row["id"] for row in rows if row.get("kind") == "real"}, "fit-real ids")
    require(set(folds) == {row["document_id"] for row in rows}, "fold map")
    return rows, folds, real_ids


def _real_ids() -> set[str]:
    payload = json.loads(FIT_REAL.read_text(encoding="utf-8"))
    rows = payload["examples"] if isinstance(payload, dict) else payload
    return {row["id"] for row in rows}


def _write_settings(rows, folds) -> None:
    write_json(
        SETTINGS_PATH,
        {
            "cleaner": CLEANER,
            "fit_queries": len(rows),
            "documents": len(folds),
            "variants": [dict(item) for item in VARIANTS],
            "cv_metric": "real-query document-macro IoU at pooled recall>=0.95; tie precision",
            "development": "scored once after the fit-CV choice, through combo/pipeline.py",
        },
    )


def _tests() -> None:
    suite = unittest.defaultTestLoader.loadTestsFromName("test_retrain")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    require(result.wasSuccessful(), "unit tests failed")


if __name__ == "__main__":
    main()
