"""Grouped 4-fold CV on real documents: v14 stage 2 with the 36 extra development CAD queries added.

The extra CAD cells (research_v5/cadextra16/cells) are training-only: added to every fold's
training rows, never scored. Scored on held-out real (doc-) documents only.
Rule fixed before running: adopt if real-document macro IoU at .25 is at least .005 above
v14's (.8180) and real-document pooled precision is not lower. Public validation is never read.
"""
from __future__ import annotations

import cv

import gc
import json
import time
from pathlib import Path

import numpy as np

import cv_hard14
from model import positive_probability

EXTRA = Path(__file__).resolve().parents[1] / "cadextra16" / "cells"
BASELINE = {"macro_iou": 0.8179722020024848, "precision": 0.6260944744028828}
MARGIN = 0.005
OUT = cv.HERE / "cv_cadextra.json"


def extra_table(codes_offset: int) -> dict:
    parts = {"x": [], "y": [], "doc": [], "fold": [], "pop": [], "syn": []}
    codes: dict[str, int] = {}
    names = cv.joined_names()
    for path in sorted(EXTRA.glob("*.npz")):
        cv._append(path, names, False, codes, parts)
    table = {key: np.concatenate(value) for key, value in parts.items()}
    table["doc"] = table["doc"] + codes_offset
    return table


def main() -> None:
    base = cv.load()
    real = np.load(cv.HERE / "real_mask.npy")
    extra = extra_table(int(base["doc"].max()) + 1)
    print(f"extra rows={len(extra['y'])} documents={len(np.unique(extra['doc']))}", flush=True)
    table = {key: np.concatenate([base[key], extra[key]]) for key in base}
    is_extra = np.concatenate([np.zeros(len(base["y"]), bool), np.ones(len(extra["y"]), bool)])
    weights = cv.training_weights(table, 0.0)
    scores = np.zeros(len(base["y"]), np.float32)
    for fold in range(cv.FOLDS):
        train = (table["fold"] != fold) | is_extra
        test = (base["fold"] == fold)
        started = time.perf_counter()
        model = cv_hard14.cv_capacity.classifier(cv_hard14.v14_config())
        model.fit(table["x"][train], table["y"][train], sample_weight=weights[train])
        scores[test] = positive_probability(model, base["x"][test])
        del model
        gc.collect()
        print(f"fold={fold} seconds={time.perf_counter() - started:.1f}", flush=True)
    subset = {key: value[real] for key, value in base.items()}
    result = cv.summarize(cv.held_out_counts(subset, np.ones(int(real.sum()), bool), scores[real]))
    adopted = result["macro_iou"] >= BASELINE["macro_iou"] + MARGIN and result["precision"] >= BASELINE["precision"]
    OUT.write_text(json.dumps({"result": result, "baseline": BASELINE, "adopted": adopted, "rule": __doc__}, indent=2), encoding="utf-8")
    print(json.dumps(result), f"adopted={adopted}")


if __name__ == "__main__":
    main()
