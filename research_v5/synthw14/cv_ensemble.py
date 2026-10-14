"""Grouped 4-fold CV on v12 cells: average of the v14 stage 2 and the iter800/leaf127 stage 2.

Rule fixed before running: the average is adopted only if real-document macro IoU at .25 is at
least .005 above v14 alone (.8180) and real-document precision is not lower. Public validation
is never read.
"""
from __future__ import annotations

import cv

import json

import numpy as np

import cv_capacity
import cv_hard14

BASELINE = {"macro_iou": 0.8179722020024848, "precision": 0.6260944744028828}
MARGIN = 0.005
OUT = cv.HERE / "cv_ensemble.json"


def main() -> None:
    table = cv.load()
    real = np.load(cv.HERE / "real_mask.npy")
    config = next(item for item in cv_capacity.configurations() if item["name"] == "iter800_leaf127_lr05")
    other = cv_hard14.held_out_scores(table, cv.training_weights(table, 0.0), config)
    np.save(cv.HERE / "oof_scores_iter800.npy", other)
    v14 = np.load(cv.HERE / "oof_scores_v14.npy")
    subset = {key: value[real] for key, value in table.items()}
    rows = {}
    for name, scores in (("iter800", other), ("average", (v14 + other) / 2)):
        rows[name] = cv.summarize(cv.held_out_counts(subset, np.ones(int(real.sum()), bool), scores[real]))
    average = rows["average"]
    adopted = average["macro_iou"] >= BASELINE["macro_iou"] + MARGIN and average["precision"] >= BASELINE["precision"]
    OUT.write_text(json.dumps({"rows": rows, "baseline": BASELINE, "adopted": adopted, "rule": __doc__}, indent=2), encoding="utf-8")
    print(json.dumps(rows), f"adopted={adopted}")


if __name__ == "__main__":
    main()
