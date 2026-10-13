"""Build symbol cells, choose on real fit queries, score development once."""
from __future__ import annotations

import argparse
import json
import time
import traceback
import unittest

import guard

guard.activate()

from devharness import fit_rows
from folds import load_document_folds
from guard import (
    BUDGET_SECONDS,
    CLEANER,
    FINDINGS_PATH,
    FIT_REAL,
    FOLDS_PATH,
    HERE,
    SETTINGS_PATH,
    guard_memory,
    log,
    peak_gb,
    require,
    write_json,
)
from join_cells import build_all
from score_symbols import score
from train_stage2 import CONTROL, SYMBOL, ensure_control, ensure_symbol, ensure_symbol_model
from write_findings import write_findings


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
    control = ensure_control(rows, folds) if limit == 0 else None
    if control is not None:
        _require_control(control)
    summary = build_all(rows, limit)
    if limit:
        log(f"smoke stop seconds={time.perf_counter() - started:.1f} peak_gb={peak_gb():.2f}")
        return
    guard_memory()
    ranked = ensure_symbol(rows, folds, control)
    guard_memory()
    if _budget_short(started):
        write_findings(started, dev_ran=False)
        log(f"stopped before development seconds={time.perf_counter() - started:.1f}")
        return
    model = ensure_symbol_model(rows, folds)
    guard_memory()
    score(model)
    write_findings(started, dev_ran=True)
    log(
        f"done seconds={time.perf_counter() - started:.1f} peak_gb={peak_gb():.2f} "
        f"selected={ranked['selected']} findings={FINDINGS_PATH.name}"
    )


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
            "variants": [dict(CONTROL), dict(SYMBOL)],
            "cv_metric": "real-query document-macro IoU at pooled recall>=0.95; tie precision",
            "development": "symbol model inside combo/pipeline.py with 21 extra channels",
        },
    )


def _require_control(control: dict) -> None:
    gaps = control["gaps"]
    thresholds = [value for key, value in gaps.items() if key.endswith("_threshold")]
    require(thresholds, "control produced no folds")
    require(max(thresholds) < 1e-9, f"control thresholds drifted {gaps}")
    require(gaps["mean_macro"] < 1e-4, f"control macro drifted {gaps}")
    require(gaps["mean_precision"] < 1e-4, f"control precision drifted {gaps}")


def _budget_short(started: float) -> bool:
    remaining = BUDGET_SECONDS - (time.perf_counter() - started)
    return remaining < 15 * 60


def _tests() -> None:
    suite = unittest.defaultTestLoader.loadTestsFromName("test_symbols")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    require(result.wasSuccessful(), "unit tests failed")


if __name__ == "__main__":
    main()
