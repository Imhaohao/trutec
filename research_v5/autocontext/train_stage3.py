"""Stage-3 HistGradientBoosting. Fit and model choice stay on the fit split."""
from __future__ import annotations

import time

import limits  # noqa: F401

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from constants import FOLD_COUNT
from context_features import FEATURE_VERSION, feature_names
from cells import iter_cached
from limits import log, require
from model import Table, document_balanced_weights, macro_precision_at_recall, positive_probability

STAGE3_SEED = 42
CANDIDATES = (
    {
        "name": "shallow",
        "learning_rate": 0.1,
        "max_leaf_nodes": 15,
        "l2_regularization": 1.0,
        "max_iter": 150,
        "min_samples_leaf": 40,
    },
    {
        "name": "recipe",
        "learning_rate": 0.08,
        "max_leaf_nodes": 31,
        "l2_regularization": 1.0,
        "max_iter": 200,
        "min_samples_leaf": 20,
    },
    {
        "name": "deep",
        "learning_rate": 0.06,
        "max_leaf_nodes": 63,
        "l2_regularization": 0.1,
        "max_iter": 250,
        "min_samples_leaf": 20,
    },
)


def load_fit_table(per_class_cap: int | None) -> Table:
    features, labels, docs, folds, pops = [], [], [], [], []
    codes: dict[str, int] = {}
    for block, target, document, fold, pop in iter_cached(per_class_cap):
        _append_block(features, labels, docs, folds, pops, codes, block, target, document, fold, pop)
    require(features, "stage3 cache is empty")
    return Table(
        x=np.concatenate(features, axis=0),
        y=np.concatenate(labels, axis=0),
        doc=np.concatenate(docs, axis=0),
        fold=np.concatenate(folds, axis=0),
        pop=np.concatenate(pops, axis=0),
        code_to_doc=tuple(sorted(codes, key=codes.get)),
    )


def cross_validate(table: Table) -> list[dict]:
    _require_folds(table)
    weights = document_balanced_weights(table.doc, table.y)
    ranked = [_score_candidate(table, weights, params) for params in CANDIDATES]
    ranked.sort(key=_rank)
    for row in ranked:
        log(f"cv {row['name']} mean={row['mean']:.4f} folds={row['fold_precision_at_recall_0_98']}")
    return ranked


def fit_stage3(table: Table, params: dict):
    weights = document_balanced_weights(table.doc, table.y)
    model = classifier(params)
    started = time.perf_counter()
    model.fit(table.x, table.y, sample_weight=weights)
    seconds = time.perf_counter() - started
    require(list(map(int, model.classes_)) == [0, 1], "stage3 classes")
    require(int(model.n_features_in_) == len(feature_names()), "stage3 width")
    log(f"stage3 fit rows={len(table.y)} seconds={seconds:.1f}")
    return model, seconds


def classifier(params: dict) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=float(params["learning_rate"]),
        max_leaf_nodes=int(params["max_leaf_nodes"]),
        l2_regularization=float(params["l2_regularization"]),
        max_iter=int(params["max_iter"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        early_stopping=False,
        random_state=STAGE3_SEED,
    )


def fit_record(table: Table, params: dict, seconds: float) -> dict:
    return {
        "params": params,
        "seconds": seconds,
        "rows": int(len(table.y)),
        "positive": int(np.count_nonzero(table.y == 1)),
        "negative": int(np.count_nonzero(table.y == 0)),
        "documents": len(table.code_to_doc),
        "feature_version": FEATURE_VERSION,
        "feature_names": feature_names(),
    }


def same_params(left: dict, right: dict) -> bool:
    keys = ("learning_rate", "max_leaf_nodes", "l2_regularization", "max_iter", "min_samples_leaf")
    return all(left[key] == right[key] for key in keys)


def _append_block(features, labels, docs, folds, pops, codes, block, target, document, fold, pop) -> None:
    code = codes.setdefault(document, len(codes))
    features.append(block)
    labels.append(target)
    docs.append(np.full(len(target), code, dtype=np.int32))
    folds.append(np.full(len(target), fold, dtype=np.int8))
    pops.append(pop)


def _score_candidate(table: Table, weights, params: dict) -> dict:
    scores = [_fold_precision(table, weights, params, fold) for fold in range(FOLD_COUNT)]
    return {
        "name": params["name"],
        "params": params,
        "fold_precision_at_recall_0_98": scores,
        "mean": float(np.mean(scores)),
    }


def _fold_precision(table: Table, weights, params: dict, fold: int) -> float:
    train = table.fold != fold
    test = table.fold == fold
    require(bool(np.any(train)) and bool(np.any(test)), f"empty fold {fold}")
    model = classifier(params)
    started = time.perf_counter()
    model.fit(table.x[train], table.y[train], sample_weight=weights[train])
    log(f"cv {params['name']} fold {fold} fit seconds={time.perf_counter() - started:.1f}")
    scores = positive_probability(model, table.x[test])
    return macro_precision_at_recall(table.doc[test], table.y[test], scores, table.pop[test])


def _require_folds(table: Table) -> None:
    for fold in range(FOLD_COUNT):
        require(bool(np.any(table.fold == fold)), f"no cells in fold {fold}")


def _rank(row: dict) -> tuple:
    return (-row["mean"], row["params"]["max_leaf_nodes"], row["params"]["max_iter"])
