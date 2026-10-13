"""Cross-fit the structure stage-2 model on the stacker's document folds."""
from __future__ import annotations

import gc
import os
import time
from typing import NamedTuple

import limits  # noqa: F401

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from cells import joined_names
from constants import CANDIDATES, CV_PER_CLASS, FOLD_COUNT, STAGE2_SEED
from dataset import _cap_classes, _population_weights
from limits import CELL_CACHE, CV_PATH, MODEL_CACHE, current_gb, log, read_json, require, write_json
from model import document_balanced_weights, macro_precision_at_recall, positive_probability
from structure_features import FEATURE_VERSION
from weights import HARD_WEIGHT, SETTINGS, apply_hard_negative, hard_negative_fraction


class Table(NamedTuple):
    x: np.ndarray
    y: np.ndarray
    doc: np.ndarray
    fold: np.ndarray
    pop: np.ndarray


def recipe_params() -> dict:
    found = [item for item in CANDIDATES if item["name"] == "recipe"]
    require(len(found) == 1, "stacker recipe candidate")
    return dict(found[0])


def ensure_cv(document_fold: dict[str, int]) -> list[dict]:
    cached = read_json(CV_PATH)
    if cached is not None and cached.get("signature") == cv_signature():
        log(f"cv cached choice={cached['ranked'][0]['name']}")
        return cached["ranked"]
    table = load_table(document_fold, CV_PER_CLASS)
    log(f"cv rows={len(table.y)} features={table.x.shape[1]} rss_gb={current_gb():.2f}")
    try:
        ranked = cross_validate(table)
    finally:
        del table
        gc.collect()
    write_json(CV_PATH, {"signature": cv_signature(), "ranked": ranked})
    log(f"cv choice={ranked[0]['name']} mean={ranked[0]['mean']:.4f}")
    return ranked


def ensure_models(document_fold: dict[str, int]) -> None:
    table = None
    for setting in SETTINGS:
        if model_ready(setting):
            log(f"model cached {setting['name']}")
            continue
        table = table if table is not None else load_table(document_fold, None)
        _fit_and_store(table, setting)
        guard = current_gb()
        log(f"model stored {setting['name']} rss_gb={guard:.2f}")
        gc.collect()
    del table
    gc.collect()


def load_fitted(name: str):
    path = model_path(name)
    stored = joblib.load(path)
    require(stored["record"]["feature_names"] == joined_names(), f"feature names {name}")
    return stored["model"]


def model_path(name: str):
    return MODEL_CACHE / f"{name}.joblib"


def model_ready(setting: dict) -> bool:
    path = model_path(setting["name"])
    record = read_json(path.with_suffix(".json"))
    return path.is_file() and record is not None and record.get("signature") == fit_signature(setting)


def cv_signature() -> dict:
    return {
        "feature_version": FEATURE_VERSION,
        "settings": [_setting_record(item) for item in SETTINGS],
        "hard_weight": HARD_WEIGHT,
        "recipe": recipe_params(),
        "cv_per_class": CV_PER_CLASS,
        "metric": "document-macro precision at recall 0.98, population weights",
    }


def fit_signature(setting: dict) -> dict:
    return {
        "feature_version": FEATURE_VERSION,
        "setting": _setting_record(setting),
        "hard_weight": HARD_WEIGHT,
        "recipe": recipe_params(),
        "rows": "all cached fit cells",
    }


def load_table(document_fold: dict[str, int], per_class_cap: int | None) -> Table:
    features, labels, docs, folds, pops = [], [], [], [], []
    codes: dict[str, int] = {}
    names = joined_names()
    for path in sorted(CELL_CACHE.glob("*.npz")):
        loaded = _load_one(path, names, document_fold, per_class_cap)
        if loaded is None:
            continue
        block, target, document, fold, pop = loaded
        code = codes.setdefault(document, len(codes))
        features.append(block)
        labels.append(target)
        docs.append(np.full(len(target), code, dtype=np.int32))
        folds.append(np.full(len(target), fold, dtype=np.int8))
        pops.append(pop)
    require(features, "structure cell cache is empty")
    return Table(
        x=np.concatenate(features, axis=0),
        y=np.concatenate(labels, axis=0),
        doc=np.concatenate(docs, axis=0),
        fold=np.concatenate(folds, axis=0),
        pop=np.concatenate(pops, axis=0),
    )


def cross_validate(table: Table) -> list[dict]:
    base = document_balanced_weights(table.doc, table.y)
    ranked = []
    for setting in SETTINGS:
        ranked.append(_score_setting(table, base, setting))
        log(f"cv {setting['name']} mean={ranked[-1]['mean']:.4f}")
    ranked.sort(key=_rank)
    return ranked


def _fit_and_store(table: Table, setting: dict) -> None:
    log(f"fit {setting['name']} rows={len(table.y)} features={table.x.shape[1]}")
    model, seconds = fit_setting(table, setting)
    record = {
        "signature": fit_signature(setting),
        "seconds": seconds,
        "rows": int(len(table.y)),
        "hard_negative_fraction": hard_negative_fraction(table.y, table.x[:, 0], setting["cutoff"]),
        "feature_names": joined_names(),
    }
    path = model_path(setting["name"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    joblib.dump({"model": model, "record": record}, temporary)
    os.replace(temporary, path)
    write_json(path.with_suffix(".json"), record)
    del model


def fit_setting(table: Table, setting: dict):
    weights = _weights(table, setting["cutoff"])
    model = make_classifier()
    started = time.perf_counter()
    model.fit(table.x, table.y, sample_weight=weights)
    seconds = time.perf_counter() - started
    require(list(map(int, model.classes_)) == [0, 1], "stage2 classes")
    require(int(model.n_features_in_) == len(joined_names()), "stage2 width")
    log(f"fit {setting['name']} seconds={seconds:.1f}")
    return model, seconds


def make_classifier() -> HistGradientBoostingClassifier:
    params = recipe_params()
    return HistGradientBoostingClassifier(
        learning_rate=float(params["learning_rate"]),
        max_leaf_nodes=int(params["max_leaf_nodes"]),
        l2_regularization=float(params["l2_regularization"]),
        max_iter=int(params["max_iter"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        early_stopping=False,
        random_state=STAGE2_SEED,
    )


def _score_setting(table: Table, base, setting: dict) -> dict:
    weights = apply_hard_negative(base, table.y, table.x[:, 0], table.doc, setting["cutoff"])
    fold_scores = [
        _fold_precision(table, weights, fold, setting["name"]) for fold in range(FOLD_COUNT)
    ]
    return {
        "name": setting["name"],
        "cutoff": setting["cutoff"],
        "hard_weight": HARD_WEIGHT if setting["cutoff"] is not None else 1.0,
        "hard_negative_fraction": hard_negative_fraction(table.y, table.x[:, 0], setting["cutoff"]),
        "fold_precision_at_recall_0_98": fold_scores,
        "mean": float(np.mean(fold_scores)),
        "params": recipe_params(),
    }


def _fold_precision(table: Table, weights, fold: int, name: str) -> float:
    train = table.fold != fold
    test = table.fold == fold
    require(bool(np.any(train)) and bool(np.any(test)), f"empty fold {fold}")
    model = make_classifier()
    started = time.perf_counter()
    model.fit(table.x[train], table.y[train], sample_weight=weights[train])
    seconds = time.perf_counter() - started
    scores = positive_probability(model, table.x[test])
    value = macro_precision_at_recall(table.doc[test], table.y[test], scores, table.pop[test])
    del model
    gc.collect()
    log(f"cv {name} fold {fold} seconds={seconds:.1f} precision={value:.4f} rss_gb={current_gb():.2f}")
    return value


def _weights(table: Table, cutoff):
    base = document_balanced_weights(table.doc, table.y)
    return apply_hard_negative(base, table.y, table.x[:, 0], table.doc, cutoff)


def _load_one(path, names, document_fold, per_class_cap):
    with np.load(path) as stored:
        require(int(stored["version"]) == FEATURE_VERSION, f"stale cache {path.name}")
        features = np.array(stored["X"], dtype=np.float32)
        labels = np.array(stored["y"], dtype=np.int8)
        document = str(stored["document"].item())
        fold = int(stored["fold"])
        n_pos = int(stored["n_pos_domain"])
        n_neg = int(stored["n_neg_domain"])
    if len(labels) == 0:
        return None
    require(fold == int(document_fold[document]), f"fold drift {path.name}")
    require(features.shape[1] == len(names), f"feature width {path.name}")
    features, labels = _cap_classes(features, labels, per_class_cap, path.stem)
    return features, labels, document, fold, _population_weights(labels, n_pos, n_neg)


def _setting_record(setting: dict) -> dict:
    return {"name": setting["name"], "cutoff": setting["cutoff"]}


def _rank(row: dict) -> tuple:
    cutoff = -1.0 if row["cutoff"] is None else float(row["cutoff"])
    return (-row["mean"], cutoff)
