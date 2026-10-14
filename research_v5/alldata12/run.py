"""Cross-fit stage 1 on fit plus development, rebuild cells, fit hardneg_0.75."""
from __future__ import annotations

import json
import subprocess
import sys
import time
import traceback

import runtime  # noqa: F401

from build_cells import build_cells, prove_fit_sampling
from fit_all import fit_all
from folds_ext import load_document_folds, write_folds
from frozen_check import verify_frozen
from manifests import combined_rows
from runtime import CACHE, HERE, log, peak_gb, write_json
from smoke import smoke_one_drawing
from stage1_train import ensure_samples, train_stage1


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
    steps = {}
    frozen_before = verify_frozen()
    folds = write_folds()
    rows = combined_rows()
    _mark(steps, "prove", prove_fit_sampling, rows)
    samples = _mark(steps, "stage1_samples", ensure_samples, rows)
    assigned = load_document_folds()
    stage1 = _mark(steps, "stage1_models", train_stage1, rows, assigned)
    cells = _mark(steps, "cells", build_cells, rows, assigned)
    model = _mark(steps, "stage2", fit_all)
    write_json(CACHE / "cell_summary.json", cells)
    test = _unit_test()
    smoke = smoke_one_drawing()
    frozen_after = verify_frozen()
    ruff = _ruff()
    wall = time.perf_counter() - started
    _write_findings(folds, samples, stage1, cells, model, test, smoke, frozen_before, frozen_after, ruff, steps, wall)
    log(f"done seconds={wall:.1f} peak_gb={peak_gb():.2f}")
    if not test["ok"] or not ruff["ok"]:
        raise RuntimeError("test or ruff failed")


def _mark(steps: dict, name: str, function, *args):
    log(f"start {name}")
    started = time.perf_counter()
    result = function(*args)
    steps[name] = time.perf_counter() - started
    write_json(CACHE / "progress.json", {"finished": name, "seconds": steps})
    log(f"finish {name} seconds={steps[name]:.1f}")
    return result


def _unit_test() -> dict:
    command = [sys.executable, "-m", "unittest", "test_model"]
    completed = subprocess.run(command, cwd=HERE, capture_output=True, text=True)
    output = (completed.stdout + completed.stderr).strip()
    log(output)
    return {"command": command, "ok": completed.returncode == 0, "returncode": completed.returncode, "output": output}


def _ruff() -> dict:
    command = [sys.executable, "-m", "ruff", "check", str(HERE)]
    completed = subprocess.run(command, cwd=HERE.parents[1], capture_output=True, text=True)
    output = (completed.stdout + completed.stderr).strip()
    log(output or "ruff ok")
    return {"command": command, "ok": completed.returncode == 0, "returncode": completed.returncode, "output": output}


def _write_findings(folds, samples, stage1, cells, model, test, smoke, frozen_before, frozen_after, ruff, steps, wall) -> None:
    text = _findings_text(folds, samples, stage1, cells, model, test, smoke, frozen_before, frozen_after, ruff, steps, wall)
    (HERE / "findings.txt").write_text(text, encoding="utf-8")


def _findings_text(folds, samples, stage1, cells, model, test, smoke, frozen_before, frozen_after, ruff, steps, wall) -> str:
    lines = [
        "v12 retrains stage 1 and stage 2 on fit plus development.",
        "Stage 1 uses the stacker crop-descriptor HGB recipe and the four fit folds,",
        "with the 12 development documents added round-robin by sorted document_id.",
        "Stage-2 cells use the held-out fold model and the uncleaned crop descriptor.",
        "Stage 2 is one hardneg_0.75 fit. No cross-validation and no threshold selection.",
        "Public validation was not read.",
        "",
        "command=" + " ".join(sys.argv),
        "threads=OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4",
        "",
        f"fold_rule={folds['rule']}",
        f"development_rule={folds['development_rule']}",
    ]
    lines.extend(_fold_lines(folds))
    lines.extend(_sample_lines(samples, steps))
    lines.extend(_stage1_lines(stage1, steps))
    lines.extend(_cell_lines(cells, steps))
    lines.extend(_stage2_lines(model, steps))
    lines.extend(_check_lines(test, smoke, frozen_before, frozen_after, ruff, wall))
    return "\n".join(lines) + "\n"


def _fold_lines(folds: dict) -> list[str]:
    lines = []
    for key in ("0", "1", "2", "3"):
        bucket = folds["summary"][key]
        lines.append(
            f"fold {key} documents={len(bucket['documents'])} queries={bucket['queries']} "
            f"fit_documents={len(bucket['fit_documents'])} fit_queries={len(bucket['fit_queries'])} "
            f"development_documents={len(bucket['development_documents'])} "
            f"development_queries={len(bucket['development_queries'])}"
        )
        lines.append("development_documents_" + key + "=" + ",".join(bucket["development_documents"]))
    return lines


def _sample_lines(samples: dict, steps: dict) -> list[str]:
    return [
        "",
        f"stage1_sample_seconds={steps['stage1_samples']:.1f}",
        f"stage1_seed_rule={samples['seed_rule']}",
        f"stage1_samples={samples['samples']}",
        f"stage1_positive={samples['positive']}",
        f"stage1_negative={samples['negative']}",
        f"stage1_fit_samples={samples['fit']['samples']}",
        f"stage1_fit_positive={samples['fit']['positive']}",
        f"stage1_fit_negative={samples['fit']['negative']}",
        f"stage1_development_samples={samples['development']['samples']}",
        f"stage1_development_positive={samples['development']['positive']}",
        f"stage1_development_negative={samples['development']['negative']}",
        f"stage1_index_check={json.dumps(samples['index_check'])}",
    ]


def _stage1_lines(stage1: dict, steps: dict) -> list[str]:
    lines = ["", f"stage1_model_seconds={steps['stage1_models']:.1f}", f"stage1_pair_width={stage1['pair_width']}"]
    for record in stage1["folds"]:
        lines.append(_model_line("stage1_fold", record))
    lines.append(_model_line("stage1_final", stage1["final"]))
    return lines


def _model_line(label: str, record: dict) -> str:
    return (
        f"{label}={record['held_out_fold']} sha256={record['sha256']} "
        f"samples={record['samples']} positive={record['positive']} negative={record['negative']} "
        f"documents={record['training_document_count']} seconds={record['seconds']:.1f} "
        f"features={record['feature_count']}"
    )


def _cell_lines(cells: dict, steps: dict) -> list[str]:
    fit = cells["fit"]
    development = cells["development"]
    return [
        "",
        f"cell_build_seconds={steps['cells']:.1f}",
        f"cell_seed_rule={cells['seed_rule']}",
        f"descriptor={cells['descriptor']}",
        f"descriptor_cleaning={cells['descriptor_cleaning']}",
        f"cells_per_class={cells['cells_per_class']}",
        f"queries_built_this_call={cells['queries_built_this_call']}",
        f"cell_queries={cells['queries']}",
        f"cell_rows={cells['cells']}",
        f"cell_positive={cells['positive']}",
        f"cell_negative={cells['negative']}",
        f"fit_queries={fit['queries']}",
        f"fit_rows={fit['cells']}",
        f"fit_positive={fit['positive']}",
        f"fit_negative={fit['negative']}",
        f"development_queries={development['queries']}",
        f"development_rows={development['cells']}",
        f"development_positive={development['positive']}",
        f"development_negative={development['negative']}",
    ]


def _stage2_lines(model: dict, steps: dict) -> list[str]:
    return [
        "",
        f"stage2_seconds={steps['stage2']:.1f}",
        "stage2_path=research_v5/alldata12/stage2_alldata_hardneg_0.75.joblib",
        f"stage2_sha256={model['sha256']}",
        f"rows={model['rows']}",
        f"positive={model['positive']}",
        f"negative={model['negative']}",
        f"fit_rows={model['fit_rows']}",
        f"fit_positive={model['fit_positive']}",
        f"fit_negative={model['fit_negative']}",
        f"development_rows={model['development_rows']}",
        f"development_positive={model['development_positive']}",
        f"development_negative={model['development_negative']}",
        f"fit_documents={model['fit_documents']}",
        f"development_documents={model['development_documents']}",
        f"hard_negative_fraction={model['hard_negative_fraction']}",
        f"fit_seconds={model['seconds']}",
        f"feature_count={len(model['feature_names'])}",
        f"recipe={json.dumps(model['signature']['recipe'])}",
        f"hard_weight={model['signature']['hard_weight']}",
        f"cutoff={model['signature']['setting']['cutoff']}",
    ]


def _check_lines(test, smoke, frozen_before, frozen_after, ruff, wall: float) -> list[str]:
    masks = smoke["masks"]
    return [
        "",
        f"unit_test_ok={test['ok']}",
        f"unit_test_returncode={test['returncode']}",
        "unit_test_command=" + " ".join(test["command"]),
        test["output"],
        "",
        f"smoke_image={smoke['image']}",
        f"smoke_queries={','.join(smoke['queries'])}",
        f"smoke_threshold={smoke['threshold']}",
        f"smoke_cleaner={smoke['cleaner']}",
        f"smoke_pixels={masks['pixels']}",
        f"smoke_positive_pixels={masks['positive_pixels']}",
        f"smoke_elapsed_seconds={smoke['invoke']['runtime']['elapsed_seconds']}",
        "smoke_command=" + " ".join(smoke["invoke"]["command"]),
        "",
        f"frozen_sources_before={frozen_before['sources_checked']}",
        f"frozen_models_before={frozen_before['models_checked']}",
        f"frozen_mismatches_before={len(frozen_before['mismatches'])}",
        f"frozen_sources_after={frozen_after['sources_checked']}",
        f"frozen_models_after={frozen_after['models_checked']}",
        f"frozen_mismatches_after={len(frozen_after['mismatches'])}",
        "",
        f"ruff_ok={ruff['ok']}",
        f"ruff_returncode={ruff['returncode']}",
        "ruff_command=" + " ".join(ruff["command"]),
        ruff["output"],
        "",
        f"wall_seconds={wall:.1f}",
        f"peak_rss_gb={peak_gb():.2f}",
    ]


if __name__ == "__main__":
    main()
