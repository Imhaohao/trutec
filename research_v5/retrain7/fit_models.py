"""Cross-fit three cleaned stage-2 variants and keep the real-query winner.

Training rows are every fit query whose document is outside the held-out fold,
including generated CAD. The score is document-macro IoU at pooled recall
>= 0.95 on the held-out fold's real queries only. Precision breaks ties.
"""
from __future__ import annotations

import gc
import os
import time
from pathlib import Path
from typing import NamedTuple

import runtime

runtime.activate()

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from build_cells import cell_path, population
from cell_metrics import RECALL_FLOOR, operating_point
from constants import CANDIDATES, FOLD_COUNT, STAGE2_SEED
from features import feature_names as stacker_names
from hardneg import apply_hard_negative, hard_negative_fraction
from model import document_balanced_weights, positive_probability
from runtime import CELL_VERSION, CLEANER, CV_PATH, MODEL_CACHE, current_gb, log, read_json, require, write_json
from structure_features import feature_names as structure_names

VARIANTS = (
    {"name": "cleaned_hardneg_0.75", "cells": "base", "cutoff": 0.75, "hard_weight": 3.0, "per_class": 2000},
    {"name": "cleaned_hardneg_0.75_x2", "cells": "double", "cutoff": 0.75, "hard_weight": 3.0, "per_class": 4000},
    {"name": "cleaned_hardneg_0.75_w5", "cells": "base", "cutoff": 0.75, "hard_weight": 5.0, "per_class": 2000},
)


class Table(NamedTuple):
    x: np.ndarray
    y: np.ndarray
    doc: np.ndarray
    fold: np.ndarray
    real: np.ndarray
    query: np.ndarray
    pop: np.ndarray
    doc_ids: tuple[str, ...]


def ensure_cv(rows: list[dict], document_fold: dict[str, int], real_ids: set[str]) -> list[dict]:
    _require_real_folds(rows, document_fold, real_ids)
    signature = cv_signature(rows)
    cached = read_json(CV_PATH)
    if cached is not None and cached.get("signature") == signature:
        log(f"cv cached choice={cached['selected']}")
        return cached["ranked"]
    ranked = _rank_fresh(rows, document_fold)
    write_json(CV_PATH, {"signature": signature, "ranked": ranked, "selected": ranked[0]["name"]})
    log(f"cv choice={ranked[0]['name']} mean_macro={ranked[0]['mean_macro_iou']:.4f}")
    return ranked


def ensure_model(rows: list[dict], document_fold: dict[str, int], selected: str) -> Path:
    setting = _variant(selected)
    path = model_path(selected)
    signature = fit_signature(rows, setting)
    record = read_json(path.with_suffix(".json"))
    if path.is_file() and record is not None and record.get("signature") == signature:
        log(f"model cached {selected}")
        return path
    table = load_table(rows, document_fold, setting["cells"])
    try:
        _fit_and_store(table, setting, path, signature)
    finally:
        del table
        gc.collect()
    return path


def model_path(name: str):
    return MODEL_CACHE / f"{name}.joblib"


def cv_signature(rows: list[dict]) -> dict:
    return {
        "cell_version": CELL_VERSION,
        "cleaner": CLEANER,
        "queries": len(rows),
        "cache_stamp": _cache_stamp(rows),
        "variants": [_public_variant(item) for item in VARIANTS],
        "recipe": _recipe(),
        "recall_floor": RECALL_FLOOR,
        "folds": FOLD_COUNT,
        "metric": "real-query document-macro IoU at pooled recall>=0.95; tie precision",
    }


def fit_signature(rows: list[dict], setting: dict) -> dict:
    return {
        "cell_version": CELL_VERSION,
        "cleaner": CLEANER,
        "cache_stamp": _cache_stamp(rows),
        "setting": _public_variant(setting),
        "recipe": _recipe(),
        "rows": "all cached fit cells for this sample size",
    }


def load_table(rows: list[dict], document_fold: dict[str, int], which: str) -> Table:
    parts = [_load_query(row, document_fold, which) for row in rows]
    kept = [part for part in parts if part is not None]
    require(kept, f"no {which} cells")
    return _pack(kept)


def make_classifier() -> HistGradientBoostingClassifier:
    params = _recipe()
    return HistGradientBoostingClassifier(
        learning_rate=float(params["learning_rate"]),
        max_leaf_nodes=int(params["max_leaf_nodes"]),
        l2_regularization=float(params["l2_regularization"]),
        max_iter=int(params["max_iter"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        early_stopping=False,
        random_state=STAGE2_SEED,
    )


def _rank_fresh(rows, document_fold) -> list[dict]:
    ranked = _score_width(rows, document_fold, "base")
    ranked.extend(_score_width(rows, document_fold, "double"))
    ranked.sort(key=_variant_key)
    return ranked


def _score_width(rows, document_fold, which: str) -> list[dict]:
    table = load_table(rows, document_fold, which)
    log(f"cv table {which} rows={len(table.y)} features={table.x.shape[1]} rss_gb={current_gb():.2f}")
    try:
        return [_score_variant(table, setting) for setting in VARIANTS if setting["cells"] == which]
    finally:
        del table
        gc.collect()


def _score_variant(table: Table, setting: dict) -> dict:
    weights = _weights(table, setting)
    folds = [_fold_point(table, weights, fold, setting) for fold in range(FOLD_COUNT)]
    record = _variant_record(table, setting, folds)
    log(f"cv {setting['name']} mean_macro={record['mean_macro_iou']:.4f} mean_precision={record['mean_precision']:.4f}")
    return record


def _fold_point(table: Table, weights, fold: int, setting: dict) -> dict:
    train = table.fold != fold
    test = (table.fold == fold) & (table.real == 1)
    require(bool(np.any(train)) and bool(np.any(test)), f"empty fold {fold}")
    model = make_classifier()
    started = time.perf_counter()
    model.fit(table.x[train], table.y[train], sample_weight=weights[train])
    seconds = time.perf_counter() - started
    scores = positive_probability(model, table.x[test])
    point = _point_from_scores(table, test, scores)
    del model
    gc.collect()
    point["fold"] = fold
    point["seconds"] = seconds
    log(
        f"cv {setting['name']} fold {fold} seconds={seconds:.1f} "
        f"macro={point['macro_iou']:.4f} rss_gb={current_gb():.2f}"
    )
    return point


def _point_from_scores(table: Table, test, scores) -> dict:
    point = operating_point(_query_records(table, test, scores), RECALL_FLOOR)
    if point is None:
        return {"failed": True, "macro_iou": 0.0, "precision": 0.0, "recall": None, "fpr": None, "threshold": None}
    point["failed"] = False
    return point


def _query_records(table: Table, test, scores) -> list[dict]:
    full = np.full(len(table.y), np.nan, dtype=np.float64)
    full[test] = scores
    records = []
    for code in np.unique(table.query[test]):
        chosen = np.flatnonzero(table.query == code)
        records.append(_one_record(table, full, chosen))
    return records


def _one_record(table: Table, full, chosen) -> dict:
    return {
        "document_id": table.doc_ids[int(table.doc[chosen[0]])],
        "labels": table.y[chosen],
        "scores": full[chosen],
        "pop": table.pop[chosen],
    }


def _fit_and_store(table: Table, setting: dict, path, signature: dict) -> None:
    log(f"fit {setting['name']} rows={len(table.y)} features={table.x.shape[1]}")
    weights = _weights(table, setting)
    model = make_classifier()
    started = time.perf_counter()
    model.fit(table.x, table.y, sample_weight=weights)
    seconds = time.perf_counter() - started
    require(list(map(int, model.classes_)) == [0, 1], "stage2 classes")
    require(int(model.n_features_in_) == len(_joined_names()), "stage2 width")
    record = _fit_record(signature, setting, table, seconds)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        joblib.dump({"model": model, "record": record}, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.is_file():
            temporary.unlink()
    write_json(path.with_suffix(".json"), record)
    log(f"fit {setting['name']} seconds={seconds:.1f}")
    del model


def _fit_record(signature, setting, table: Table, seconds: float) -> dict:
    return {
        "signature": signature,
        "seconds": seconds,
        "rows": int(len(table.y)),
        "real_rows": int(np.count_nonzero(table.real == 1)),
        "hard_negative_fraction_all": hard_negative_fraction(table.y, table.x[:, 0], setting["cutoff"]),
        "hard_negative_fraction_real": _real_fraction(table, setting["cutoff"]),
        "feature_names": _joined_names(),
        "cleaner": CLEANER,
    }


def _variant_record(table: Table, setting: dict, folds: list[dict]) -> dict:
    return {
        **_public_variant(setting),
        "folds": folds,
        "mean_macro_iou": float(np.mean([fold["macro_iou"] for fold in folds])),
        "mean_precision": float(np.mean([fold["precision"] for fold in folds])),
        "hard_negative_fraction_all": hard_negative_fraction(table.y, table.x[:, 0], setting["cutoff"]),
        "hard_negative_fraction_real": _real_fraction(table, setting["cutoff"]),
        "rows": int(len(table.y)),
        "real_rows": int(np.count_nonzero(table.real == 1)),
        "features": int(table.x.shape[1]),
    }


def _weights(table: Table, setting: dict):
    base = document_balanced_weights(table.doc, table.y)
    return apply_hard_negative(base, table.y, table.x[:, 0], table.doc, setting["cutoff"], setting["hard_weight"])


def _load_query(row: dict, document_fold: dict[str, int], which: str):
    with np.load(cell_path(row["id"])) as stored:
        block = _arrays(stored, which)
    if block is None:
        return None
    features, labels, document, fold, real, n_pos, n_neg = block
    require(fold == int(document_fold[document]), f"fold drift {row['id']}")
    require(features.shape[1] == len(_joined_names()), f"width {row['id']}")
    return {
        "features": features,
        "labels": labels,
        "document": document,
        "fold": fold,
        "real": real,
        "pop": population(labels, n_pos, n_neg),
        "query_id": row["id"],
    }


def _arrays(stored, which: str):
    features, labels = _matrix(stored, which)
    if len(labels) == 0:
        return None
    return (
        features,
        labels,
        str(stored["document"].item()),
        int(stored["fold"]),
        int(stored["real"]),
        int(stored["n_pos_domain"]),
        int(stored["n_neg_domain"]),
    )


def _matrix(stored, which: str):
    if which == "base":
        return np.array(stored["X"], dtype=np.float32), np.array(stored["y"], dtype=np.int8)
    require(which == "double", f"sample {which}")
    return np.array(stored["X2"], dtype=np.float32), np.array(stored["y2"], dtype=np.int8)


def _pack(parts: list[dict]) -> Table:
    doc_codes: dict[str, int] = {}
    query_codes: dict[str, int] = {}
    documents = _codes(parts, "document", doc_codes)
    queries = _codes(parts, "query_id", query_codes)
    return Table(
        x=np.concatenate([part["features"] for part in parts], axis=0),
        y=np.concatenate([part["labels"] for part in parts], axis=0),
        doc=documents,
        fold=np.concatenate([np.full(len(part["labels"]), part["fold"], dtype=np.int8) for part in parts]),
        real=np.concatenate([np.full(len(part["labels"]), part["real"], dtype=np.int8) for part in parts]),
        query=queries,
        pop=np.concatenate([part["pop"] for part in parts]),
        doc_ids=tuple(sorted(doc_codes, key=doc_codes.get)),
    )


def _codes(parts: list[dict], field: str, codes: dict[str, int]) -> np.ndarray:
    blocks = []
    for part in parts:
        code = codes.setdefault(part[field], len(codes))
        blocks.append(np.full(len(part["labels"]), code, dtype=np.int32))
    return np.concatenate(blocks)


def _real_fraction(table: Table, cutoff: float) -> float:
    real = table.real == 1
    return hard_negative_fraction(table.y[real], table.x[real, 0], cutoff)


def _require_real_folds(rows, document_fold, real_ids) -> None:
    counts = [0] * FOLD_COUNT
    for row in rows:
        if row["id"] in real_ids:
            counts[int(document_fold[row["document_id"]])] += 1
    require(all(count > 0 for count in counts), f"real queries per fold {counts}")
    log(f"real queries per fold {counts}")


def _variant(name: str) -> dict:
    found = [item for item in VARIANTS if item["name"] == name]
    require(len(found) == 1, f"variant {name}")
    return found[0]


def _public_variant(setting: dict) -> dict:
    return {
        "name": setting["name"],
        "cells": setting["cells"],
        "cutoff": setting["cutoff"],
        "hard_weight": setting["hard_weight"],
        "per_class": setting["per_class"],
    }


def _recipe() -> dict:
    found = [item for item in CANDIDATES if item["name"] == "recipe"]
    require(len(found) == 1, "recipe candidate")
    return dict(found[0])


def _joined_names() -> list[str]:
    return stacker_names() + structure_names()


def _cache_stamp(rows: list[dict]) -> int:
    stamps = [cell_path(row["id"]).stat().st_mtime_ns for row in rows]
    require(len(stamps) == len(rows), "cache stamp")
    return int(max(stamps))


def _variant_key(row: dict) -> tuple:
    return (-row["mean_macro_iou"], -row["mean_precision"], row["name"])
