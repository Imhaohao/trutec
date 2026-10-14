"""Cross-fit stage 2 on the stacker's folds, then cache stage-3 training cells.

Stage 2 for fold k is trained on the other folds' cached stage-2 rows and
applied to fold k through drawing.consume_volumes and that fold's stage-1 model.
"""
from __future__ import annotations

import gc
import time

import limits  # noqa: F401

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from constants import FOLD_COUNT, STAGE2_SEED
from cells import cell_indices, cell_path, cell_ready, gather_labeled, write_cell_cache
from drawing import consume_volumes
from features import feature_names as stacker_feature_names
from folds import indexed_training_order
from grids import feature_volume_from_stage2
from limits import (
    IMAGE_LOG,
    STAGE2_FOLD_CACHE,
    append_jsonl,
    current_gb,
    dump_joblib,
    guard_memory,
    log,
    read_json,
    require,
    write_json,
)
from model import document_balanced_weights, load_table
from stage1 import fold_model_path


def stage2_fold_path(fold: int):
    return STAGE2_FOLD_CACHE / f"fold_{fold}.joblib"


def ensure_stage2_models(document_fold: dict[str, int], params: dict) -> list[dict]:
    if _folds_ready():
        log("stage2 folds cached")
        return [read_fold_record(fold) for fold in range(FOLD_COUNT)]
    table = load_table(document_fold, None)
    log(f"stage2 table rows={len(table.y)} features={table.x.shape[1]}")
    try:
        return [ensure_one_fold(table, params, fold) for fold in range(FOLD_COUNT)]
    finally:
        del table
        gc.collect()


def read_fold_record(fold: int) -> dict:
    path = stage2_fold_path(fold).with_suffix(".json")
    payload = read_json(path)
    require(payload is not None, f"missing stage2 fold record {fold}")
    return payload


def build_cell_cache(rows: list[dict], document_fold: dict[str, int]) -> dict:
    started = time.perf_counter()
    built = 0
    for fold in range(FOLD_COUNT):
        built += build_fold(rows, document_fold, fold)
    summary = summarize_cells(rows)
    summary["queries_built_this_call"] = built
    summary["seconds"] = time.perf_counter() - started
    log(
        f"stage3 cells queries={summary['queries']} cells={summary['cells']} "
        f"pos={summary['positive']} neg={summary['negative']} built={built}"
    )
    return summary


def summarize_cells(rows: list[dict]) -> dict:
    queries = cells = positive = negative = 0
    for row in rows:
        queries, cells, positive, negative = _count_one(row, queries, cells, positive, negative)
    return {"queries": queries, "cells": cells, "positive": positive, "negative": negative}


def ensure_one_fold(table, params: dict, fold: int) -> dict:
    if stage2_fold_path(fold).is_file() and stage2_fold_path(fold).with_suffix(".json").is_file():
        return read_fold_record(fold)
    model, seconds, rows = fit_held_out(table, params, fold)
    record = {"held_out_fold": fold, "seconds": seconds, "rows": rows, "params": params}
    _store_fold(fold, model, record)
    log(f"stage2 fold {fold} rows={rows} seconds={seconds:.1f}")
    return record


def fit_held_out(table, params: dict, fold: int):
    train = table.fold != fold
    require(bool(np.any(train)), f"fold {fold} has no training rows")
    weights = document_balanced_weights(table.doc[train], table.y[train])
    model = stage2_classifier(params)
    started = time.perf_counter()
    model.fit(table.x[train], table.y[train], sample_weight=weights)
    seconds = time.perf_counter() - started
    require(list(map(int, model.classes_)) == [0, 1], "stage2 classes")
    require(int(model.n_features_in_) == len(stacker_feature_names()), "stage2 width")
    return model, seconds, int(np.count_nonzero(train))


def build_fold(rows, document_fold, fold: int) -> int:
    grouped = pending_by_image(rows, document_fold, fold)
    if not grouped:
        log(f"stage3 cells cached fold {fold}")
        return 0
    return _score_fold(grouped, fold)


def pending_by_image(rows, document_fold, fold: int) -> dict:
    grouped: dict = {}
    for index, row in indexed_training_order(rows):
        _keep_pending(grouped, document_fold, fold, index, row)
    return grouped


def stage2_classifier(params: dict) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=float(params["learning_rate"]),
        max_leaf_nodes=int(params["max_leaf_nodes"]),
        l2_regularization=float(params["l2_regularization"]),
        max_iter=int(params["max_iter"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        early_stopping=False,
        random_state=STAGE2_SEED,
    )


def _folds_ready() -> bool:
    return all(_fold_ready(fold) for fold in range(FOLD_COUNT))


def _fold_ready(fold: int) -> bool:
    return stage2_fold_path(fold).is_file() and stage2_fold_path(fold).with_suffix(".json").is_file()


def _store_fold(fold: int, model, record: dict) -> None:
    path = stage2_fold_path(fold)
    dump_joblib(path, {"model": model, "params": record["params"], "held_out_fold": fold})
    write_json(path.with_suffix(".json"), record)


def _count_one(row, queries, cells, positive, negative):
    require(cell_ready(row["id"]), f"missing stage3 cache {row['id']}")
    with np.load(cell_path(row["id"])) as stored:
        labels = stored["y"]
    return (
        queries + 1,
        cells + int(len(labels)),
        positive + int(np.count_nonzero(labels == 1)),
        negative + int(np.count_nonzero(labels == 0)),
    )


def _score_fold(grouped: dict, fold: int) -> int:
    payload = joblib.load(fold_model_path(fold))
    stage1 = payload["model"]
    stage2 = joblib.load(stage2_fold_path(fold))["model"]
    trained = set(payload["training_documents"])
    built = 0
    try:
        for image, group in grouped.items():
            built += build_image(image, group, stage1, stage2, trained, fold)
            guard_memory()
    finally:
        del stage1, stage2, payload
        gc.collect()
    return built


def build_image(image, group, stage1, stage2, trained, fold: int) -> int:
    document = group[0][1]["document_id"]
    _same_document(group, document)
    require(document not in trained, f"stage1 saw {document}")
    spec = {key: group[0][1][key] for key in ("id", "image", "width", "height")}
    captured = list(group)
    started = time.perf_counter()
    consume_volumes(
        spec,
        [row["query_box"] for _index, row in captured],
        [stage1] * len(captured),
        lambda index, volume: save_one(captured[index][0], captured[index][1], volume, stage2, fold),
    )
    seconds = time.perf_counter() - started
    append_jsonl(
        IMAGE_LOG,
        {"image": image, "fold": fold, "queries": len(captured), "seconds": seconds, "rss_gb": current_gb()},
    )
    log(f"stage3 image {image} queries={len(captured)} seconds={seconds:.1f} rss_gb={current_gb():.2f}")
    return len(captured)


def save_one(index: int, row: dict, volume, stage2, fold: int) -> None:
    centers_y, centers_x, sampled = cell_indices(row, index)
    pos_idx, neg_idx, n_pos, n_neg = sampled
    features = feature_volume_from_stage2(volume, stage2)
    _require_grid(features, centers_y, centers_x)
    block, target = gather_labeled(features, pos_idx, neg_idx)
    del features
    write_cell_cache(row["id"], block, target, row["document_id"], fold, n_pos, n_neg)


def _keep_pending(grouped, document_fold, fold: int, index: int, row: dict) -> None:
    if document_fold[row["document_id"]] != fold:
        return
    if cell_ready(row["id"]):
        return
    grouped.setdefault(row["image"], []).append((index, row))


def _same_document(group, document: str) -> None:
    for _index, row in group:
        require(row["document_id"] == document, "image spans documents")


def _require_grid(features, centers_y, centers_x) -> None:
    require(features.shape[0] == len(centers_y), "grid height")
    require(features.shape[1] == len(centers_x), "grid width")
