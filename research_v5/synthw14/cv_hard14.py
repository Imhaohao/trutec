"""Grouped 4-fold CV on v12 cells of the v14 stage 2 over hard-negative weight and threshold.

Model: iter600/leaf63 (v14). Hard-negative weight in {3, 6, 10}; thresholds .15-.50.
Rule fixed before running: the (weight, threshold) with the largest real-document macro IoU
replaces v14's (3, .25) only if it is at least .005 above v14's own CV macro IoU.
Public validation is never read.
"""
from __future__ import annotations

import cv

import gc
import json
import time

import numpy as np

import cv_capacity
import weights as weight_module
from model import positive_probability
from runtime import ALLDATA12, log

cv.CELL_FIT = ALLDATA12 / "cache" / "cells" / "fit"
cv.CELL_DEV = ALLDATA12 / "cache" / "cells" / "development"
cv.CELL_SYN = ALLDATA12 / "cache" / "cells" / "no-synthetic"
HARD_WEIGHTS = (3.0, 6.0, 10.0)
THRESHOLDS = tuple(round(value, 2) for value in np.arange(0.15, 0.501, 0.05))
CURRENT = (3.0, 0.25)
MARGIN = 0.005
OUT = cv.HERE / "cv_hard14.json"


def v14_config() -> dict:
    return next(item for item in cv_capacity.configurations() if item["name"] == "iter600_leaf63")


def held_out_scores(table: dict, weights: np.ndarray, config: dict) -> np.ndarray:
    scores = np.zeros(len(table["y"]), dtype=np.float32)
    for fold in range(cv.FOLDS):
        train, test = table["fold"] != fold, table["fold"] == fold
        started = time.perf_counter()
        model = cv_capacity.classifier(config)
        model.fit(table["x"][train], table["y"][train], sample_weight=weights[train])
        scores[test] = positive_probability(model, table["x"][test])
        del model
        gc.collect()
        log(f"hard={weight_module.HARD_WEIGHT} fold={fold} seconds={time.perf_counter() - started:.1f}")
    return scores


def at_threshold(table: dict, scores: np.ndarray, threshold: float) -> dict:
    cv.THRESHOLD = threshold
    everything = np.ones(len(scores), dtype=bool)
    return cv.summarize(cv.held_out_counts(table, everything, scores))


def main() -> None:
    table = cv.load()
    config = v14_config()
    rows = []
    for hard in HARD_WEIGHTS:
        weight_module.HARD_WEIGHT = hard
        scores = held_out_scores(table, cv.training_weights(table, 0.0), config)
        for threshold in THRESHOLDS:
            rows.append({"hard_weight": hard, "threshold": threshold, **at_threshold(table, scores, threshold)})
            log(json.dumps(rows[-1]))
        OUT.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    current = next(row for row in rows if (row["hard_weight"], row["threshold"]) == CURRENT)
    best = max(rows, key=lambda row: row["macro_iou"])
    chosen = best if best["macro_iou"] >= current["macro_iou"] + MARGIN else current
    OUT.write_text(json.dumps({"rows": rows, "current": current, "chosen": chosen, "rule": __doc__}, indent=2), encoding="utf-8")
    log(f"chosen {json.dumps(chosen)}")


if __name__ == "__main__":
    main()
