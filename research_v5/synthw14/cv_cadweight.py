"""Grouped 4-fold CV on v12 cells of the v14 stage 2 with generated-CAD documents down-weighted.

CAD weight in {1, .5, .25, 0}; scored on held-out real (doc-) documents only.
Rule fixed before running: a CAD weight below 1 is adopted only if its real-document macro
IoU at threshold .25 is at least .01 above weight 1 (v14, real-document CV .8180); the
largest passing value wins. Public validation is never read.
"""
from __future__ import annotations

import cv

import json

import numpy as np

import cv_hard14

CAD_WEIGHTS = (1.0, 0.5, 0.25, 0.0)
MARGIN = 0.01
OUT = cv.HERE / "cv_cadweight.json"


def real_documents(table: dict) -> np.ndarray:
    return np.load(cv.HERE / "real_mask.npy")


def real_summary(table: dict, real: np.ndarray, scores: np.ndarray) -> dict:
    subset = {key: value[real] for key, value in table.items()}
    return cv.summarize(cv.held_out_counts(subset, np.ones(int(real.sum()), bool), scores[real]))


def main() -> None:
    table = cv.load()
    real = real_documents(table)
    assert len(real) == len(table["y"])
    base = cv.training_weights(table, 0.0)
    results = []
    for cad_weight in CAD_WEIGHTS:
        weights = np.where(real, base, base * cad_weight).astype(np.float32)
        scores = cv_hard14.held_out_scores(table, weights, cv_hard14.v14_config())
        results.append({"cad_weight": cad_weight, **real_summary(table, real, scores)})
        print(json.dumps(results[-1]), flush=True)
        OUT.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
    baseline = results[0]["macro_iou"]
    passing = [row for row in results[1:] if row["macro_iou"] >= baseline + MARGIN]
    chosen = max(passing, key=lambda row: row["macro_iou"]) if passing else results[0]
    OUT.write_text(json.dumps({"results": results, "chosen": chosen, "rule": __doc__}, indent=2), encoding="utf-8")
    print(f"chosen cad_weight={chosen['cad_weight']}")


if __name__ == "__main__":
    main()
