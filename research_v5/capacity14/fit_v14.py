"""Fit the v14 stage 2: v12's hardneg_0.75 rows and weights, larger HGB chosen by CV.

The configuration is read from research_v5/synthw14/cv_capacity12.json ("chosen"), which
was selected on grouped CV over real fit and development documents. Stage 1, cells, cleaner,
color gate and threshold are v12's, unchanged.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "alldata12"))

import runtime  # noqa: E402,F401

import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

import joblib  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402

from cells import joined_names  # noqa: E402
from constants import STAGE2_SEED  # noqa: E402
from fit_all import _hardneg_setting, _load_table  # noqa: E402
from runtime import log, sha256_file  # noqa: E402
from train import _weights  # noqa: E402

CV_RESULT = HERE.parent / "synthw14" / "cv_capacity12.json"
STAGE2_PATH = HERE / "stage2_v14.joblib"


def chosen_config() -> dict:
    payload = json.loads(CV_RESULT.read_text(encoding="utf-8"))
    found = [item["config"] for item in payload["results"] if item["config"]["name"] == payload["chosen"]]
    assert len(found) == 1 and payload["chosen"] != "recipe", payload["chosen"]
    return found[0]


def classifier(config: dict) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=float(config["learning_rate"]),
        max_leaf_nodes=int(config["max_leaf_nodes"]),
        l2_regularization=float(config["l2_regularization"]),
        max_iter=int(config["max_iter"]),
        min_samples_leaf=int(config["min_samples_leaf"]),
        early_stopping=False,
        random_state=STAGE2_SEED,
    )


def main() -> None:
    config = chosen_config()
    setting = _hardneg_setting()
    table, counts = _load_table(joined_names())
    weights = _weights(table, setting["cutoff"])
    model = classifier(config)
    started = time.perf_counter()
    model.fit(table.x, table.y, sample_weight=weights)
    seconds = time.perf_counter() - started
    assert list(map(int, model.classes_)) == [0, 1] and int(model.n_features_in_) == len(joined_names())
    record = {
        "config": config,
        "setting": {"name": setting["name"], "cutoff": setting["cutoff"]},
        "rows": int(len(table.y)),
        "fit_rows": counts["fit"]["cells"],
        "development_rows": counts["development"]["cells"],
        "seconds": seconds,
        "feature_names": joined_names(),
        "cells": "research_v5/alldata12/cache/cells (v12 stage-1 cross-fit folds)",
    }
    temporary = STAGE2_PATH.with_name(f".{STAGE2_PATH.name}.{os.getpid()}.partial")
    joblib.dump({"model": model, "record": record}, temporary)
    os.replace(temporary, STAGE2_PATH)
    record["sha256"] = sha256_file(STAGE2_PATH)
    STAGE2_PATH.with_suffix(".json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    log(f"v14 stage2 config={config['name']} rows={record['rows']} seconds={seconds:.1f} sha256={record['sha256']}")


if __name__ == "__main__":
    main()
