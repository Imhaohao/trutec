"""Run auto-context stage 3. Cached steps are skipped."""
from __future__ import annotations

import gc
import json
import time
import traceback

import limits  # noqa: F401

import joblib

from constants import CELLS_PER_CLASS, CV_PER_CLASS, FOLD_COUNT, FIT_MANIFEST
from context_features import (
    FEATURE_VERSION,
    STATS,
    WINDOWS_PX,
    effective_side,
    feature_names,
)
from crossfit import build_cell_cache, ensure_stage2_models
from develop import score_development
from folds import load_document_folds
from limits import (
    CACHE,
    CV_RESULTS,
    FOLDS_PATH,
    FROZEN_STAGE2,
    HERE,
    PROGRESS_PATH,
    SETTINGS_PATH,
    STAGE3_MODEL,
    dump_joblib,
    free_gb,
    guard_memory,
    log,
    peak_gb,
    read_json,
    require,
    thread_report,
    write_json,
)
from report_text import write_findings
from train_stage3 import (
    CANDIDATES,
    STAGE3_SEED,
    cross_validate,
    fit_record,
    fit_stage3,
    load_fit_table,
    same_params,
)


def main() -> None:
    started = time.perf_counter()
    try:
        execute(started)
    except Exception:
        _fail(started)
        raise


def execute(started: float) -> None:
    error = HERE / "error.txt"
    if error.is_file():
        error.unlink()
    _require_disk()
    rows = read_fit_rows()
    folds = load_document_folds(FOLDS_PATH)
    check_folds(rows, folds)
    top = load_stage2_choice()
    frozen = joblib.load(FROZEN_STAGE2)["params"]
    write_settings(top, frozen)
    log(f"stage2 choice={top['name']} frozen_match={same_params(top['params'], frozen)}")
    write_json(CACHE / "stage2_fold_timings.json", ensure_stage2_models(folds, top["params"]))
    guard_memory()
    write_json(CACHE / "cell_summary.json", build_cell_cache(rows, folds))
    guard_memory()
    ranked = ensure_cv()
    ensure_model(ranked[0]["params"])
    score_development()
    _finish(started, "complete")


def read_fit_rows() -> list[dict]:
    payload = json.loads(FIT_MANIFEST.read_text(encoding="utf-8"))
    rows = payload["examples"] if isinstance(payload, dict) else payload
    require(len(rows) == 357, f"fit queries {len(rows)}")
    return rows


def check_folds(rows: list[dict], folds: dict[str, int]) -> None:
    documents = {row["document_id"] for row in rows}
    require(set(folds) == documents, "fold map does not match fit documents")
    require(set(folds.values()) == set(range(FOLD_COUNT)), "fold ids")


def load_stage2_choice() -> dict:
    ranked = read_json(limits.STACKER_CV_PATH)
    require(isinstance(ranked, list) and bool(ranked), "stacker cv_results.json is empty")
    return ranked[0]


def write_settings(top: dict, frozen: dict) -> None:
    write_json(SETTINGS_PATH, _settings_payload(top, frozen))
    log(f"wrote {SETTINGS_PATH}")


def ensure_cv() -> list[dict]:
    cached = read_json(CV_RESULTS)
    if _cv_current(cached):
        log(f"cv cached choice={cached['ranked'][0]['name']}")
        return cached["ranked"]
    table = load_fit_table(CV_PER_CLASS)
    log(f"cv rows={len(table.y)} features={table.x.shape[1]}")
    try:
        ranked = cross_validate(table)
    finally:
        del table
        gc.collect()
    write_json(CV_RESULTS, {"feature_version": FEATURE_VERSION, "candidates": list(CANDIDATES), "ranked": ranked})
    return ranked


def ensure_model(params: dict) -> dict:
    cached = _cached_record(params)
    if cached is not None:
        log(f"stage3 model cached choice={params['name']}")
        return cached
    table = load_fit_table(None)
    log(f"stage3 final rows={len(table.y)} features={table.x.shape[1]}")
    try:
        model, seconds = fit_stage3(table, params)
        record = fit_record(table, params, seconds)
    finally:
        del table
        gc.collect()
    dump_joblib(STAGE3_MODEL, {"model": model, "record": record})
    write_json(CACHE / "stage3_fit.json", record)
    return record


def _settings_payload(top: dict, frozen: dict) -> dict:
    return {
        "stage2_cv_top": top,
        "frozen_stage2_params": frozen,
        "frozen_matches_cv_top": _params_match(top["params"], frozen),
        "windows_px": list(WINDOWS_PX),
        "effective_sides_px": {str(window): effective_side(window) for window in WINDOWS_PX},
        "window_rule": (
            "nominal square side W px; Chebyshev half-width in cells is (W // 2) // 8; "
            "effective side is (2 * half + 1) * 8 px"
        ),
        "stats": list(STATS),
        "fraction_rule": "fraction of cells in the window whose value is strictly greater than 0.3 or 0.6",
        "percentile_rule": "32-bin histogram bin centers, the same method as the stacker window percentiles",
        "stage2_score": "stage-2 positive probability at the stride-8 cell",
        "stage1_probability": "stacker volume channel cell_score",
        "distance": "stacker volume channel signed_distance_over_256",
        "color_mismatch": "stacker volume channel color_mismatch_w32 at the cell",
        "cells_per_class": CELLS_PER_CLASS,
        "cv_per_class": CV_PER_CLASS,
        "sample_seed_rule": "STAGE2_SEED + query index from indexed_training_order",
        "unknown_rule": "a cell is negative only when its center pixel is reviewed and not positive",
        "stage3_candidates": list(CANDIDATES),
        "stage3_seed": STAGE3_SEED,
        "stage3_early_stopping": False,
        "selection_metric": (
            "mean over the 4 fit folds of document-macro precision at recall >= 0.98; "
            "training weights are document-balanced and the metric uses population weights"
        ),
        "color_gate": {"window": 16, "type": "linear", "strength": 0.6},
        "development_stage2": "research_v5/stacker/cache/stage2_model.joblib",
        "threads": 6,
        "thread_info": thread_report(),
        "feature_version": FEATURE_VERSION,
        "feature_names": feature_names(),
    }


def _cv_current(cached) -> bool:
    if not isinstance(cached, dict):
        return False
    if cached.get("feature_version") != FEATURE_VERSION:
        return False
    if cached.get("candidates") != list(CANDIDATES):
        return False
    ranked = cached.get("ranked")
    return isinstance(ranked, list) and len(ranked) == len(CANDIDATES)


def _cached_record(params: dict):
    if not STAGE3_MODEL.is_file():
        return None
    record = joblib.load(STAGE3_MODEL).get("record")
    if not _record_matches(record, params):
        return None
    return record


def _record_matches(record, params: dict) -> bool:
    if not isinstance(record, dict):
        return False
    if record.get("feature_version") != FEATURE_VERSION:
        return False
    stored = record.get("params")
    if not isinstance(stored, dict):
        return False
    return _params_match(stored, params)


def _params_match(left: dict, right: dict) -> bool:
    return same_params(left, right) and left.get("name") == right.get("name")


def _require_disk() -> None:
    free = free_gb()
    log(f"free_gb={free:.2f} peak_gb={peak_gb():.2f}")
    require(free > 1.5, f"only {free:.2f} GB free")


def _finish(started: float, status: str) -> None:
    elapsed = time.perf_counter() - started
    write_json(PROGRESS_PATH, {"wall_seconds": elapsed, "peak_gb": peak_gb(), "status": status})
    write_findings()
    log(f"done seconds={elapsed:.1f} peak_gb={peak_gb():.2f} status={status}")


def _fail(started: float) -> None:
    (HERE / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
    log(traceback.format_exc())
    try:
        _finish(started, "failed")
    except Exception:
        log(traceback.format_exc())


if __name__ == "__main__":
    main()
