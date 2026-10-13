"""Cross-fit stage 1 on fit, development, and synthetic rows, then fit hardneg_0.75."""
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
from runtime import CACHE, HERE, STAGE1_PATH, STAGE2_PATH, log, peak_gb, require, sha256_file, write_json
from smoke import smoke_one_drawing
from stage1_train import ensure_samples, train_stage1


def main() -> None:
    started = time.perf_counter()
    try:
        execute(started)
    except Exception:
        _write_error()
        raise


def execute(started: float) -> None:
    done = HERE / "DONE.json"
    require(not done.is_file(), "DONE.json exists; refusing to modify synth13")
    error = HERE / "error.txt"
    if error.is_file():
        error.unlink()
    _make_cache()
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
    write_json(CACHE / "cell_summary.json", {key: value for key, value in cells.items() if key != "images"})
    test = _unit_test()
    smoke = smoke_one_drawing()
    frozen_after = verify_frozen()
    ruff = _ruff()
    wall = time.perf_counter() - started
    _write_findings(
        folds, samples, stage1, cells, model, test, smoke, frozen_before, frozen_after, ruff, steps, wall
    )
    require(test["ok"] and ruff["ok"], "test or ruff failed")
    _write_done(stage1, model)


def _make_cache() -> None:
    for path in (
        CACHE,
        CACHE / "stage1",
        CACHE / "folds",
        CACHE / "cells" / "fit",
        CACHE / "cells" / "development",
        CACHE / "cells" / "synthetic",
        CACHE / "banks",
    ):
        path.mkdir(parents=True, exist_ok=True)


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


def _write_findings(folds, samples, stage1, cells, model, test, smoke, before, after, ruff, steps, wall) -> None:
    text = _findings_text(folds, samples, stage1, cells, model, test, smoke, before, after, ruff, steps, wall)
    (HERE / "findings.txt").write_text(text, encoding="utf-8")


def _findings_text(folds, samples, stage1, cells, model, test, smoke, before, after, ruff, steps, wall) -> str:
    lines = _intro_lines()
    lines.extend(_count_lines(folds))
    lines.extend(_fold_lines(folds))
    lines.extend(_sample_lines(samples, steps))
    lines.extend(_stage1_lines(stage1, steps))
    lines.extend(_cell_lines(cells, steps))
    lines.extend(_stage2_lines(model, steps))
    lines.extend(_check_lines(test, smoke, before, after, ruff, wall))
    return "\n".join(lines) + "\n"


def _intro_lines() -> list[str]:
    return [
        "v13 retrains the v12 stage-1 and stage-2 recipes on fit, development, and synthetic trap sheets.",
        "Each synthetic sheet is its own document. Fit and development folds are copied from alldata12.",
        "Synthetic documents are assigned round-robin by sorted document_id.",
        "Stage 1 uses the stacker crop-descriptor HistGradientBoosting recipe.",
        "Stage-1 seeds use STAGE1_SEED plus the index from indexed_training_order of fit rows,",
        "then development rows, then synthetic rows. Fit samples are read from the stacker cache.",
        "Development samples are read from the alldata12 cache, which used that same combined index.",
        "Stage-2 cells use the held-out fold model and the uncleaned crop descriptor.",
        "Stage-2 seeds use the index from indexed_training_order of each row's own manifest:",
        "fit rows use the fit manifest index, development rows use the development manifest index,",
        "and synthetic rows use the synthetic manifest index.",
        "Fit and development images are read from dataset/. Synthetic images and masks are read from",
        "research_v5/synthetic/data. The data root is chosen per row and passed into the frozen readers.",
        "Stage 2 is one hardneg_0.75 fit. No cross-validation and no threshold selection.",
        "Public validation was not read.",
        "",
        "command=" + " ".join(sys.argv),
        "threads=OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4",
        "",
    ]


def _count_lines(folds: dict) -> list[str]:
    totals = _source_totals(folds)
    return [
        f"fold_rule={folds['rule']}",
        f"development_rule={folds['development_rule']}",
        f"synthetic_rule={folds['synthetic_rule']}",
        f"fit_documents={totals['fit_documents']} fit_queries={totals['fit_queries']}",
        f"development_documents={totals['development_documents']} development_queries={totals['development_queries']}",
        f"synthetic_documents={totals['synthetic_documents']} synthetic_queries={totals['synthetic_queries']}",
    ]


def _source_totals(folds: dict) -> dict[str, int]:
    totals = {name: 0 for name in (
        "fit_documents",
        "fit_queries",
        "development_documents",
        "development_queries",
        "synthetic_documents",
        "synthetic_queries",
    )}
    for key in ("0", "1", "2", "3"):
        bucket = folds["summary"][key]
        for name in totals:
            totals[name] += len(bucket[name])
    return totals


def _fold_lines(folds: dict) -> list[str]:
    lines = []
    for key in ("0", "1", "2", "3"):
        lines.append(_fold_line(key, folds["summary"][key]))
        lines.append("development_documents_" + key + "=" + ",".join(folds["summary"][key]["development_documents"]))
    return lines


def _fold_line(key: str, bucket: dict) -> str:
    return (
        f"fold {key} documents={len(bucket['documents'])} queries={bucket['queries']} "
        f"fit_documents={len(bucket['fit_documents'])} fit_queries={len(bucket['fit_queries'])} "
        f"development_documents={len(bucket['development_documents'])} "
        f"development_queries={len(bucket['development_queries'])} "
        f"synthetic_documents={len(bucket['synthetic_documents'])} "
        f"synthetic_queries={len(bucket['synthetic_queries'])}"
    )


def _sample_lines(samples: dict, steps: dict) -> list[str]:
    lines = [
        "",
        f"stage1_sample_seconds={steps['stage1_samples']:.1f}",
        f"stage1_seed_rule={samples['seed_rule']}",
        f"stage1_samples={samples['samples']}",
        f"stage1_positive={samples['positive']}",
        f"stage1_negative={samples['negative']}",
        f"stage1_index_check={json.dumps(samples['index_check'])}",
        f"stage1_sample_origins={json.dumps(samples['sample_origins'])}",
    ]
    for name in ("fit", "development", "synthetic"):
        bucket = samples[name]
        lines.append(
            f"stage1_{name}_samples={bucket['samples']} "
            f"positive={bucket['positive']} negative={bucket['negative']} files={bucket['files']}"
        )
    return lines


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
    lines = [
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
    ]
    for name in ("fit", "development", "synthetic"):
        bucket = cells[name]
        lines.append(
            f"{name}_queries={bucket['queries']} {name}_rows={bucket['cells']} "
            f"{name}_positive={bucket['positive']} {name}_negative={bucket['negative']}"
        )
    return lines


def _stage2_lines(model: dict, steps: dict) -> list[str]:
    return [
        "",
        f"stage2_seconds={steps['stage2']:.1f}",
        "stage2_path=research_v5/synth13/stage2_alldata_hardneg_0.75.joblib",
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
        f"synthetic_rows={model['synthetic_rows']}",
        f"synthetic_positive={model['synthetic_positive']}",
        f"synthetic_negative={model['synthetic_negative']}",
        f"fit_documents={model['fit_documents']}",
        f"development_documents={model['development_documents']}",
        f"synthetic_documents={model['synthetic_documents']}",
        f"hard_negative_fraction={model['hard_negative_fraction']}",
        f"fit_seconds={model['seconds']}",
        f"feature_count={len(model['feature_names'])}",
        f"recipe={json.dumps(model['signature']['recipe'])}",
        f"hard_weight={model['signature']['hard_weight']}",
        f"cutoff={model['signature']['setting']['cutoff']}",
    ]


def _check_lines(test, smoke, before, after, ruff, wall: float) -> list[str]:
    masks = smoke["masks"]
    lines = [
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
        f"frozen_sources_before={before['sources_checked']}",
        f"frozen_models_before={before['models_checked']}",
        f"frozen_mismatches_before={len(before['mismatches'])}",
        f"frozen_sources_after={after['sources_checked']}",
        f"frozen_models_after={after['models_checked']}",
        f"frozen_mismatches_after={len(after['mismatches'])}",
    ]
    lines.extend(_frozen_hash_lines(after))
    lines.extend(
        [
            "",
            f"ruff_ok={ruff['ok']}",
            f"ruff_returncode={ruff['returncode']}",
            "ruff_command=" + " ".join(ruff["command"]),
            ruff["output"],
            "",
            f"wall_seconds={wall:.1f}",
            f"peak_rss_gb={peak_gb():.2f}",
        ]
    )
    return lines


def _frozen_hash_lines(report: dict) -> list[str]:
    return [f"frozen_sha256 {item['path']} {item['sha256']}" for item in report["records"]]


def _write_done(stage1: dict, model: dict) -> None:
    payload = {
        "stage1_sha256": stage1["final"]["sha256"],
        "stage2_sha256": model["sha256"],
    }
    require(sha256_file(STAGE1_PATH) == payload["stage1_sha256"], "stage1 hash changed before DONE")
    require(sha256_file(STAGE2_PATH) == payload["stage2_sha256"], "stage2 hash changed before DONE")
    write_json(HERE / "DONE.json", payload)


def _write_error() -> None:
    if (HERE / "DONE.json").is_file():
        return
    (HERE / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")


if __name__ == "__main__":
    main()
