"""Run the structure experiment. Cached steps are skipped."""
from __future__ import annotations

import json
import time
import traceback
import unittest

import limits  # noqa: F401

from cells import build_cell_cache
from develop import score_development
from devharness import fit_rows
from folds import load_document_folds
from limits import CACHE, HERE, STACKER, guard_memory, log, peak_gb, require, write_json
from report_text import write_findings, write_settings
from train import ensure_cv, ensure_models


def main() -> None:
    started = time.perf_counter()
    try:
        execute(started)
    except Exception:
        (HERE / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def execute(started: float) -> None:
    error = HERE / "error.txt"
    if error.is_file():
        error.unlink()
    _tests()
    rows = fit_rows(real_only=False)
    require(len(rows) == 357, f"fit queries {len(rows)}")
    folds = load_document_folds(STACKER / "folds.json")
    _check_folds(rows, folds)
    write_settings({"fit_queries": len(rows), "documents": len(folds)})
    summary = build_cell_cache(rows, folds)
    write_json(CACHE / "cell_summary.json", summary)
    guard_memory()
    ranked = ensure_cv(folds)
    ensure_models(folds)
    guard_memory()
    score_development()
    write_findings(started, ranked)
    log(f"done seconds={time.perf_counter() - started:.1f} peak_gb={peak_gb():.2f}")


def _tests() -> None:
    suite = unittest.defaultTestLoader.loadTestsFromName("test_structure")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    require(result.wasSuccessful(), "unit tests failed")


def _check_folds(rows: list[dict], folds: dict[str, int]) -> None:
    documents = {row["document_id"] for row in rows}
    require(set(folds) == documents, "fold map does not match fit documents")
    payload = json.loads((STACKER / "folds.json").read_text(encoding="utf-8"))
    require(int(payload["fold_count"]) == 4, "fold count")


if __name__ == "__main__":
    main()
