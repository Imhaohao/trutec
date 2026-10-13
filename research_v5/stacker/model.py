"""Second-stage HistGradientBoosting, fit only on the fit split."""
from __future__ import annotations

import time
from typing import NamedTuple

import bootstrap  # noqa: F401

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from constants import CANDIDATES, FOLD_COUNT, PREDICT_BATCH, STAGE2_SEED, log, require
from dataset import iter_cached
from features import cell_index, feature_names


class Table(NamedTuple):
    x: np.ndarray
    y: np.ndarray
    doc: np.ndarray
    fold: np.ndarray
    pop: np.ndarray
    code_to_doc: tuple[str, ...]


def load_table(document_fold: dict[str, int], per_class_cap: int | None) -> Table:
    features, labels, docs, folds, pops = [], [], [], [], []
    codes: dict[str, int] = {}
    for block, target, document, pop in iter_cached(per_class_cap):
        code = codes.setdefault(document, len(codes))
        features.append(block)
        labels.append(target)
        docs.append(np.full(len(target), code, dtype=np.int32))
        folds.append(np.full(len(target), document_fold[document], dtype=np.int8))
        pops.append(pop)
    require(features, "stage2 cache is empty")
    order = tuple(sorted(codes, key=codes.get))
    return Table(
        x=np.concatenate(features, axis=0),
        y=np.concatenate(labels, axis=0),
        doc=np.concatenate(docs, axis=0),
        fold=np.concatenate(folds, axis=0),
        pop=np.concatenate(pops, axis=0),
        code_to_doc=order,
    )


def document_balanced_weights(documents: np.ndarray, labels: np.ndarray) -> np.ndarray:
    weights = np.zeros(len(labels), dtype=np.float64)
    for document in np.unique(documents):
        _assign_document(weights, documents, labels, int(document))
    return weights.astype(np.float32)


def cross_validate(table: Table) -> list[dict]:
    weights = document_balanced_weights(table.doc, table.y)
    ranked = []
    for params in CANDIDATES:
        fold_scores = [
            _fold_precision(table, weights, params, fold) for fold in range(FOLD_COUNT)
        ]
        ranked.append(
            {
                "name": params["name"],
                "params": params,
                "fold_precision_at_recall_0_98": fold_scores,
                "mean": float(np.mean(fold_scores)),
            }
        )
        log(f"cv {params['name']} mean={ranked[-1]['mean']:.4f} folds={fold_scores}")
    ranked.sort(key=_candidate_rank)
    return ranked


def fit_stage2(table: Table, params: dict) -> HistGradientBoostingClassifier:
    weights = document_balanced_weights(table.doc, table.y)
    model = _classifier(params)
    started = time.perf_counter()
    model.fit(table.x, table.y, sample_weight=weights)
    log(f"stage2 fit rows={len(table.y)} seconds={time.perf_counter() - started:.1f}")
    require(list(map(int, model.classes_)) == [0, 1], "stage2 classes")
    require(int(model.n_features_in_) == len(feature_names()), "stage2 width")
    return model


def scores_at_coordinates(model, volume: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Label-free lookup: each pixel reads the stage-2 score of its stride-8 cell."""
    require(int(volume.shape[-1]) == int(model.n_features_in_), "feature width")
    grid_w = int(volume.shape[1])
    iy, ix = cell_index(y, x, int(volume.shape[0]), grid_w)
    keys = iy * grid_w + ix
    unique, inverse = np.unique(keys, return_inverse=True)
    rows = unique // grid_w
    cols = unique % grid_w
    predicted = positive_probability(model, volume[rows, cols])
    return predicted[inverse]


def positive_probability(model, features: np.ndarray) -> np.ndarray:
    if len(features) == 0:
        return np.empty(0, dtype=np.float32)
    output = np.empty(len(features), dtype=np.float32)
    for start in range(0, len(features), PREDICT_BATCH):
        stop = min(start + PREDICT_BATCH, len(features))
        output[start:stop] = _positive_column(model, features[start:stop])
    return output


def macro_precision_at_recall(docs, labels, scores, weights) -> float:
    values = []
    for document in np.unique(docs):
        mask = docs == document
        if not np.any(labels[mask] == 1):
            continue
        values.append(
            precision_at_recall(labels[mask], scores[mask], weights[mask])
        )
    if not values:
        return 0.0
    return float(np.mean(values))


def precision_at_recall(labels, scores, weights, target: float = 0.98) -> float:
    order = np.argsort(-np.asarray(scores), kind="mergesort")
    ordered = np.asarray(labels)[order]
    mass = np.asarray(weights, dtype=np.float64)[order]
    positive_mass = float(mass[ordered == 1].sum())
    if positive_mass <= 0.0:
        return 0.0
    true_positive = np.cumsum(mass * (ordered == 1))
    false_positive = np.cumsum(mass * (ordered == 0))
    recall = true_positive / positive_mass
    precision = true_positive / np.maximum(true_positive + false_positive, 1e-12)
    eligible = np.flatnonzero(recall >= target)
    if len(eligible) == 0:
        return 0.0
    return float(precision[eligible].max())


def _fold_precision(table: Table, weights, params: dict, fold: int) -> float:
    train = table.fold != fold
    test = table.fold == fold
    require(bool(np.any(train)) and bool(np.any(test)), f"empty fold {fold}")
    model = _classifier(params)
    started = time.perf_counter()
    model.fit(table.x[train], table.y[train], sample_weight=weights[train])
    log(f"cv {params['name']} fold {fold} fit seconds={time.perf_counter() - started:.1f}")
    scores = positive_probability(model, table.x[test])
    return macro_precision_at_recall(table.doc[test], table.y[test], scores, table.pop[test])


def _classifier(params: dict) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=params["learning_rate"],
        max_leaf_nodes=params["max_leaf_nodes"],
        l2_regularization=params["l2_regularization"],
        max_iter=params["max_iter"],
        min_samples_leaf=params["min_samples_leaf"],
        early_stopping=False,
        random_state=STAGE2_SEED,
    )


def _assign_document(weights, documents, labels, document: int) -> None:
    mask = documents == document
    positive = mask & (labels == 1)
    negative = mask & (labels == 0)
    if np.any(positive) and np.any(negative):
        weights[positive] = 0.5 / np.count_nonzero(positive)
        weights[negative] = 0.5 / np.count_nonzero(negative)
        return
    present = positive if np.any(positive) else negative
    if np.any(present):
        weights[present] = 1.0 / np.count_nonzero(present)


def _positive_column(model, batch: np.ndarray) -> np.ndarray:
    raw = model.predict_proba(batch)
    require(list(map(int, model.classes_)) == [0, 1], "stage2 classes")
    return np.asarray(raw[:, 1], dtype=np.float32)


def _candidate_rank(row: dict) -> tuple:
    return (-row["mean"], row["params"]["max_leaf_nodes"], row["params"]["max_iter"])
