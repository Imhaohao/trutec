"""Grouped 4-fold CV of stage-2 capacity on real documents only (synthetic weight 0).

Rule fixed before running: a configuration replaces the recipe only if its real-document
macro IoU at threshold 0.25 is at least 0.01 above the recipe's. Public validation is
never read.
"""
from __future__ import annotations

import cv  # noqa: F401  (pins import paths and thread caps)

import gc
import json
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from constants import STAGE2_SEED
from cv import FOLDS, HERE, held_out_counts, load, summarize, training_weights
from model import positive_probability
from runtime import log
from train import recipe_params

MARGIN = 0.01
OUT = HERE / "cv_capacity.json"


def configurations() -> list[dict]:
    recipe = recipe_params()
    base = {key: recipe[key] for key in ("learning_rate", "max_leaf_nodes", "l2_regularization", "max_iter", "min_samples_leaf")}
    return [
        {"name": "recipe", **base},
        {"name": "iter600_leaf63", **base, "max_iter": 600, "max_leaf_nodes": 63},
        {"name": "iter800_leaf127_lr05", **base, "max_iter": 800, "max_leaf_nodes": 127, "learning_rate": 0.05},
    ]


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


def real_only(table: dict) -> dict:
    keep = ~table["syn"]
    return {key: value[keep] for key, value in table.items()}


def run_config(table: dict, weights: np.ndarray, config: dict) -> dict:
    counts: dict[int, list[float]] = {}
    for fold in range(FOLDS):
        train, test = table["fold"] != fold, table["fold"] == fold
        started = time.perf_counter()
        model = classifier(config)
        model.fit(table["x"][train], table["y"][train], sample_weight=weights[train])
        counts.update(held_out_counts(table, test, positive_probability(model, table["x"][test])))
        del model
        gc.collect()
        log(f"{config['name']} fold={fold} seconds={time.perf_counter() - started:.1f}")
    result = {"config": config, **summarize(counts)}
    log(json.dumps(result))
    return result


def main() -> None:
    table = real_only(load())
    weights = training_weights(table, 0.0)
    results = []
    for config in configurations():
        results.append(run_config(table, weights, config))
        OUT.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
    baseline = results[0]["macro_iou"]
    winners = [item for item in results[1:] if item["macro_iou"] >= baseline + MARGIN]
    chosen = max(winners, key=lambda item: item["macro_iou"]) if winners else results[0]
    OUT.write_text(json.dumps({"results": results, "chosen": chosen["config"]["name"], "rule": __doc__}, indent=2), encoding="utf-8")
    log(f"chosen {chosen['config']['name']}")


if __name__ == "__main__":
    main()
