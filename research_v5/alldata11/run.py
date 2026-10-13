"""Build development cells, fit hardneg_0.75, smoke-test one drawing, write findings."""
from __future__ import annotations

import json
import subprocess
import sys
import time
import traceback

import runtime  # noqa: F401

from build_cells import build_development_cells
from fit_all import fit_all
from frozen_check import verify_frozen
from runtime import HERE, log, peak_gb
from smoke import compare_one_drawing


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
    frozen_before = verify_frozen()
    cell_started = time.perf_counter()
    cells = build_development_cells()
    cell_seconds = time.perf_counter() - cell_started
    (HERE / "cache" / "cell_summary.json").parent.mkdir(parents=True, exist_ok=True)
    (HERE / "cache" / "cell_summary.json").write_text(json.dumps(cells, indent=2), encoding="utf-8")
    model = fit_all()
    test = _unit_test()
    smoke = compare_one_drawing()
    frozen_after = verify_frozen()
    ruff = _ruff()
    wall = time.perf_counter() - started
    _write_findings(cells, model, test, smoke, frozen_before, frozen_after, ruff, cell_seconds, wall)
    log(f"done seconds={wall:.1f} peak_gb={peak_gb():.2f}")
    if not test["ok"] or not ruff["ok"]:
        raise RuntimeError("test or ruff failed")


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


def _write_findings(cells, model, test, smoke, frozen_before, frozen_after, ruff, cell_seconds, wall) -> None:
    text = _findings_text(cells, model, test, smoke, frozen_before, frozen_after, ruff, cell_seconds, wall)
    (HERE / "findings.txt").write_text(text, encoding="utf-8")


def _findings_text(cells, model, test, smoke, frozen_before, frozen_after, ruff, cell_seconds, wall) -> str:
    diff = smoke["diff"]
    lines = [
        "All-data hardneg_0.75 retrain. Fit cells were read from the structure cache.",
        "Development cells use the original stage-1 model and the uncleaned crop descriptor.",
        "No cross-validation and no threshold selection.",
        "",
        f"development_queries={cells['queries']}",
        f"development_cells={cells['cells']}",
        f"development_positive={cells['positive']}",
        f"development_negative={cells['negative']}",
        f"cells_per_class={cells['cells_per_class']}",
        f"seed_rule={cells['seed_rule']}",
        f"stage2_seed={cells['stage2_seed']}",
        f"descriptor={cells['descriptor']}",
        f"stage1={cells['stage1']}",
        f"cell_build_seconds={cell_seconds:.1f}",
        f"queries_built_this_call={cells['queries_built_this_call']}",
        "",
        f"model_path=research_v5/alldata11/stage2_alldata_hardneg_0.75.joblib",
        f"model_sha256={model['sha256']}",
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
        "",
        f"unit_test_ok={test['ok']}",
        f"unit_test_returncode={test['returncode']}",
        test["output"],
        "",
        f"smoke_image={smoke['image']}",
        f"smoke_queries={','.join(smoke['queries'])}",
        f"smoke_threshold={smoke['threshold']}",
        f"smoke_cleaner=g2",
        f"smoke_pixels={diff['pixels']}",
        f"smoke_differing_pixels={diff['differing_pixels']}",
        f"v11_elapsed_seconds={smoke['v11']['runtime']['elapsed_seconds']}",
        f"v7_elapsed_seconds={smoke['v7']['runtime']['elapsed_seconds']}",
        "v11_command=" + " ".join(smoke["v11"]["command"]),
        "v7_command=" + " ".join(smoke["v7"]["command"]),
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
        ruff["output"],
        "",
        f"wall_seconds={wall:.1f}",
        f"peak_rss_gb={peak_gb():.2f}",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
