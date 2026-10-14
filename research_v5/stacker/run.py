"""Run the stacker experiment. Cached steps are skipped."""
from __future__ import annotations

import bootstrap  # noqa: F401

import gc
import json
import time
import traceback

import cv2
import joblib

cv2.setNumThreads(4)

from constants import CACHE, CV_PER_CLASS, STACKER, log, require
from dataset import build_stage2_cache
from devharness import fit_rows
from evaluate_dev import (
    assert_development_unseen,
    baseline_report,
    dump_json,
    load_original,
    per_query_recall,
    score_development,
    write_report,
)
from findings import write_findings
from folds import assign_folds, fold_payload, save_folds
from features import feature_names
from model import cross_validate, fit_stage2, load_table
from stage1 import collect_stage1_samples, load_fold_payloads, train_fold_models
import checks


def main() -> None:
    started = time.perf_counter()
    try:
        _run(started)
    except Exception:
        (STACKER / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        write_findings([f"peak_rss_gb={_peak_gb():.2f}"])
        raise


def _run(started: float) -> None:
    error = STACKER / "error.txt"
    if error.is_file():
        error.unlink()
    checks.run()
    baseline_report()
    rows = fit_rows(real_only=False)
    _write_folds(rows)
    write_findings()
    stage1 = collect_stage1_samples(rows)
    fold_records = train_fold_models(rows, _document_folds())
    dump_json(STACKER / "stage1_summary.json", {"samples": stage1, "folds": fold_records})
    write_findings()
    _stage2(rows)
    write_findings()
    _development()
    _per_query()
    elapsed = time.perf_counter() - started
    write_findings([f"wall_seconds_this_process={elapsed:.1f}", f"peak_rss_gb={_peak_gb():.2f}"])
    log(f"done seconds={elapsed:.1f} peak_gb={_peak_gb():.2f}")


def _write_folds(rows: list[dict]) -> None:
    assigned = assign_folds(rows)
    save_folds(STACKER / "folds.json", fold_payload(rows, assigned))
    log(f"folds documents={len(assigned)}")


def _document_folds() -> dict[str, int]:
    payload = json.loads((STACKER / "folds.json").read_text(encoding="utf-8"))
    return {str(key): int(value) for key, value in payload["document_fold"].items()}


def _stage2(rows: list[dict]) -> None:
    assigned = _document_folds()
    summary = build_stage2_cache(rows, assigned, load_fold_payloads())
    dump_json(STACKER / "stage2_summary.json", summary)
    model_path = CACHE / "stage2_model.joblib"
    if model_path.is_file():
        log("stage2 model cached")
        return
    params = _choose_params(assigned)
    table = load_table(assigned, None)
    log(f"stage2 final rows={len(table.y)} features={table.x.shape[1]}")
    model = fit_stage2(table, params)
    del table
    gc.collect()
    joblib.dump(
        {"model": model, "params": params, "feature_names": feature_names()},
        model_path,
    )
    log("stage2 model saved")


def _choose_params(assigned: dict[str, int]) -> dict:
    path = STACKER / "cv_results.json"
    if path.is_file():
        ranked = json.loads(path.read_text(encoding="utf-8"))
        log(f"cv cached choice={ranked[0]['name']}")
        return ranked[0]["params"]
    table = load_table(assigned, CV_PER_CLASS)
    log(f"cv rows={len(table.y)} features={table.x.shape[1]}")
    ranked = cross_validate(table)
    del table
    gc.collect()
    dump_json(path, ranked)
    require(ranked, "cv produced no candidates")
    return ranked[0]["params"]


def _development() -> None:
    _model, metadata = load_original()
    assert_development_unseen(metadata)
    model_path = CACHE / "stage2_model.joblib"
    stored = joblib.load(model_path)
    scores = score_development(stored["model"], model_path.stat().st_mtime)
    result = write_report("stacker_context_hgb", scores, STACKER / "dev_report.json")
    log(json.dumps(result["selections"]))


def _per_query() -> None:
    dev = json.loads((STACKER / "dev_report.json").read_text(encoding="utf-8"))
    baseline = json.loads((STACKER / "baseline_hgb.json").read_text(encoding="utf-8"))
    operating = dev["selections"]["best_precision_at_recall_ge_0_98"]
    payload = {
        "stacker_operating_point": operating,
        "baseline_operating_point": baseline["selections"]["best_precision_at_recall_ge_0_98"],
        "note": (
            "The threshold is the development harness sweep point with the best "
            "pooled precision among thresholds whose pooled recall is at least 0.98. "
            "Hyperparameters were chosen on fit-split grouped cross-validation only."
        ),
    }
    if operating is None:
        payload["queries"] = []
        dump_json(STACKER / "per_query_recall.json", payload)
        return
    model_path = CACHE / "stage2_model.joblib"
    stored = joblib.load(model_path)
    scores = score_development(stored["model"], model_path.stat().st_mtime)
    payload["queries"] = per_query_recall(scores, float(operating["threshold"]))
    dump_json(STACKER / "per_query_recall.json", payload)


def _peak_gb() -> float:
    import resource
    import sys

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / 1e9
    return rss / 1e6


if __name__ == "__main__":
    main()
