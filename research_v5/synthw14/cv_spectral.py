"""Grouped 4-fold CV on v12 cells: v14 stage 2 with versus without spectral features.

Rule fixed before running: the spectral features are adopted only if real-document macro
IoU at threshold .25 is at least .01 above the same model without them (v14 CV .7804).
Thresholds .15-.50 are reported for the precision/recall trade-off. Public validation is
never read.
"""
from __future__ import annotations

import cv

import json
from pathlib import Path

import numpy as np

import cv_hard14

SPECTRAL = Path(__file__).resolve().parents[1] / "spectral15" / "cache"
BASELINE_IOU = 0.7804207972539993
MARGIN = 0.01
OUT = cv.HERE / "cv_spectral.json"


def load_with_spectral() -> dict:
    table = cv.load()
    blocks = []
    for directory in (cv.CELL_FIT, cv.CELL_DEV):
        for path in sorted(directory.glob("*.npz")):
            with np.load(path) as stored:
                if len(stored["y"]) == 0:
                    continue
                stored_y = np.array(stored["y"], dtype=np.int8)
            with np.load(SPECTRAL / path.name) as extra:
                assert np.array_equal(extra["y"], stored_y), path.name
                blocks.append(np.array(extra["S"], dtype=np.float32))
    spectral = np.concatenate(blocks)
    assert len(spectral) == len(table["y"])
    table["x"] = np.concatenate([table["x"], spectral], axis=1)
    return table


def main() -> None:
    table = load_with_spectral()
    scores = cv_hard14.held_out_scores(table, cv.training_weights(table, 0.0), cv_hard14.v14_config())
    np.save(cv.HERE / "oof_scores_spectral.npy", scores)
    rows = [{"threshold": t, **cv_hard14.at_threshold(table, scores, t)} for t in cv_hard14.THRESHOLDS]
    at_default = next(row for row in rows if row["threshold"] == 0.25)
    adopted = at_default["macro_iou"] >= BASELINE_IOU + MARGIN
    payload = {"rows": rows, "baseline_macro_iou": BASELINE_IOU, "adopted": adopted, "rule": __doc__}
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for row in rows:
        print(json.dumps(row))
    print(f"adopted={adopted}")


if __name__ == "__main__":
    main()
