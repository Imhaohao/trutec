"""Which v14 stage-2 features separate false-positive from true-positive cells out of fold?

Out-of-fold scores from the v14 configuration on v12 cells (real fit and development
documents only). Among cells predicted positive at .25, report the single-feature AUC for
true versus false positives, the documents carrying most false-positive mass, and how many
false positives are structurally identical to the query. Public validation is never read.
"""
from __future__ import annotations

import cv

import json

import numpy as np
from sklearn.metrics import roc_auc_score

import cv_hard14
from cells import joined_names

OUT = cv.HERE / "diagnose_fp.json"


def feature_aucs(x: np.ndarray, true_positive: np.ndarray, names: list[str]) -> list[dict]:
    rows = []
    for index, name in enumerate(names):
        column = x[:, index]
        if not np.isfinite(column).all() or np.ptp(column) == 0:
            continue
        auc = roc_auc_score(true_positive, column)
        rows.append({"feature": name, "auc": float(auc), "separation": float(abs(auc - 0.5))})
    return sorted(rows, key=lambda row: -row["separation"])


def document_mass(table: dict, predicted: np.ndarray) -> list[dict]:
    rows = []
    for code in np.unique(table["doc"]):
        mask = table["doc"] == code
        fp = float(table["pop"][mask & predicted & (table["y"] == 0)].sum())
        tp = float(table["pop"][mask & predicted & (table["y"] == 1)].sum())
        rows.append({"doc": int(code), "fp": fp, "precision": tp / max(tp + fp, 1e-9)})
    total = sum(row["fp"] for row in rows)
    for row in rows:
        row["fp_share"] = row["fp"] / total
    return sorted(rows, key=lambda row: -row["fp"])[:10]


def identical_share(x: np.ndarray, names: list[str], false_positive: np.ndarray) -> dict:
    cos = x[:, names.index("orient_cos_w128")]
    ratio = x[:, names.index("period_ratio_w128")]
    color = x[:, names.index("color_mismatch_w32")]
    same = (cos >= 0.95) & (np.abs(ratio - 1.0) <= 0.1)
    return {
        "fp_cells": int(false_positive.sum()),
        "fp_same_angle_and_spacing": float(same[false_positive].mean()),
        "tp_same_angle_and_spacing": float(same[~false_positive].mean()),
        "fp_color_mismatch_median": float(np.median(color[false_positive])),
        "tp_color_mismatch_median": float(np.median(color[~false_positive])),
    }


def main() -> None:
    table = cv.load()
    names = joined_names()
    scores = cv_hard14.held_out_scores(table, cv.training_weights(table, 0.0), cv_hard14.v14_config())
    predicted = scores >= 0.25
    kept = predicted
    true_positive = table["y"][kept] == 1
    x = table["x"][kept]
    payload = {
        "predicted_cells": int(kept.sum()),
        "documents": document_mass(table, predicted),
        "identical": identical_share(x, names, ~true_positive),
        "top_features": feature_aucs(x, true_positive, names)[:25],
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    np.save(cv.HERE / "oof_scores_v14.npy", scores)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
