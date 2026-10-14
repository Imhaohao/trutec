"""Fit-real cross-validation for the control cells and the symbol cells.

The control calls retrain7's fold scorer on retrain7's base matrices, so it is
the cleaned_hardneg_0.75 recipe: hard-negative cutoff 0.75, weight 3, 2000
cells per class. The symbol run uses those same rows with 21 extra channels.
"""
from __future__ import annotations

import gc
import os
import time

import guard

guard.activate()

import joblib
import numpy as np

from constants import FOLD_COUNT, STAGE2_SEED
from dataset import _population_weights
from fit_models import _pack, _recipe, _score_variant, load_table, make_classifier
from guard import (
    BASE_WIDTH,
    CELL_CACHE,
    CELL_VERSION,
    CONTROL_PATH,
    CV_PATH,
    MODEL_CACHE,
    RETRAIN_CELLS,
    RETRAIN_CV,
    current_gb,
    log,
    read_json,
    require,
    safe_id,
    write_json,
)
from hardneg import apply_hard_negative, hard_negative_fraction
from model import document_balanced_weights
from symbol_features import CHANNEL_COUNT, feature_names

CONTROL = {
    "name": "control_hardneg_0.75",
    "cells": "base",
    "cutoff": 0.75,
    "hard_weight": 3.0,
    "per_class": 2000,
}
SYMBOL = {
    "name": "symbols_hardneg_0.75",
    "cells": "base",
    "cutoff": 0.75,
    "hard_weight": 3.0,
    "per_class": 2000,
}
REFERENCE_NAME = "cleaned_hardneg_0.75"


def ensure_control(rows: list[dict], document_fold: dict[str, int]) -> dict:
    signature = _control_signature(rows)
    cached = read_json(CONTROL_PATH)
    if cached is not None and cached.get("signature") == signature:
        log("control cv cached")
        return cached
    table = load_table(rows, document_fold, "base")
    require(int(table.x.shape[1]) == BASE_WIDTH, f"control width {table.x.shape[1]}")
    log(f"control table rows={len(table.y)} features={table.x.shape[1]} rss_gb={current_gb():.2f}")
    try:
        record = _score_variant(table, CONTROL)
    finally:
        del table
        gc.collect()
    payload = _control_payload(signature, record)
    write_json(CONTROL_PATH, payload)
    _log_gaps(payload["gaps"])
    return payload


def ensure_symbol(rows: list[dict], document_fold: dict[str, int], control: dict) -> dict:
    signature = _symbol_signature(rows)
    cached = read_json(CV_PATH)
    if cached is not None and cached.get("signature") == signature:
        log(f"cv cached choice={cached['selected']}")
        return cached
    record = _symbol_record(rows, document_fold)
    ranked = _rank(control["record"], record)
    payload = {
        "signature": signature,
        "recipe": _recipe(),
        "ranked": ranked,
        "selected": ranked[0]["name"],
        "control_gaps": control["gaps"],
        "recall_floor": 0.95,
        "folds": FOLD_COUNT,
        "seed": STAGE2_SEED,
    }
    write_json(CV_PATH, payload)
    log(f"cv choice={payload['selected']} symbol_macro={record['mean_macro_iou']:.6f}")
    return payload


def ensure_symbol_model(rows: list[dict], document_fold: dict[str, int]) -> object:
    path = model_path()
    signature = _symbol_signature(rows)
    record = read_json(path.with_suffix(".json"))
    if path.is_file() and record is not None and record.get("signature") == signature:
        log("symbol model cached")
        return path
    table = _symbol_table(rows, document_fold)
    try:
        _fit(table, path, signature)
    finally:
        del table
        gc.collect()
    return path


def model_path():
    return MODEL_CACHE / f"{SYMBOL['name']}.joblib"


def _symbol_record(rows, document_fold) -> dict:
    table = _symbol_table(rows, document_fold)
    log(f"symbol table rows={len(table.y)} features={table.x.shape[1]} rss_gb={current_gb():.2f}")
    try:
        return _score_variant(table, SYMBOL)
    finally:
        del table
        gc.collect()


def _symbol_table(rows, document_fold):
    parts = []
    for row in rows:
        part = _symbol_part(row, document_fold)
        if part is not None:
            parts.append(part)
    require(parts, "no symbol cells")
    return _pack(parts)


def _symbol_part(row, document_fold):
    with np.load(_cell_path(row["id"])) as stored:
        block = _arrays(stored)
    if block is None:
        return None
    features, labels, document, fold, real, n_pos, n_neg = block
    require(fold == int(document_fold[document]), f"fold drift {row['id']}")
    require(features.shape[1] == BASE_WIDTH + CHANNEL_COUNT, f"width {row['id']}")
    return _part(row, features, labels, document, fold, real, n_pos, n_neg)


def _arrays(stored):
    features = np.array(stored["X"], dtype=np.float32)
    labels = np.array(stored["y"], dtype=np.int8)
    if len(labels) == 0:
        return None
    require(int(stored["version"]) == CELL_VERSION, "stale symbol cell")
    return (
        features,
        labels,
        str(stored["document"].item()),
        int(stored["fold"]),
        int(stored["real"]),
        int(stored["n_pos_domain"]),
        int(stored["n_neg_domain"]),
    )


def _part(row, features, labels, document, fold, real, n_pos, n_neg) -> dict:
    return {
        "features": features,
        "labels": labels,
        "document": document,
        "fold": fold,
        "real": real,
        "pop": _population_weights(labels, n_pos, n_neg),
        "query_id": row["id"],
    }


def _fit(table, path, signature: dict) -> None:
    log(f"fit {SYMBOL['name']} rows={len(table.y)} features={table.x.shape[1]}")
    weights = _weights(table, SYMBOL)
    model = make_classifier()
    started = time.perf_counter()
    model.fit(table.x, table.y, sample_weight=weights)
    seconds = time.perf_counter() - started
    require(list(map(int, model.classes_)) == [0, 1], "stage2 classes")
    require(int(model.n_features_in_) == BASE_WIDTH + CHANNEL_COUNT, "stage2 width")
    record = _fit_record(signature, table, seconds)
    _dump(path, model, record)
    write_json(path.with_suffix(".json"), record)
    log(f"fit {SYMBOL['name']} seconds={seconds:.1f}")
    del model


def _fit_record(signature, table, seconds: float) -> dict:
    return {
        "signature": signature,
        "seconds": seconds,
        "rows": int(len(table.y)),
        "real_rows": int(np.count_nonzero(table.real == 1)),
        "features": feature_names(),
        "base_width": BASE_WIDTH,
        "hard_negative_fraction_all": hard_negative_fraction(table.y, table.x[:, 0], SYMBOL["cutoff"]),
        "hard_negative_fraction_real": _real_fraction(table),
    }


def _weights(table, setting: dict):
    base = document_balanced_weights(table.doc, table.y)
    return apply_hard_negative(base, table.y, table.x[:, 0], table.doc, setting["cutoff"], setting["hard_weight"])


def _real_fraction(table) -> float:
    real = table.real == 1
    return hard_negative_fraction(table.y[real], table.x[real, 0], SYMBOL["cutoff"])


def _dump(path, model, record) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        joblib.dump({"model": model, "record": record}, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.is_file():
            temporary.unlink()


def _control_payload(signature: dict, record: dict) -> dict:
    reference = _reference_row()
    return {"signature": signature, "record": record, "reference_name": REFERENCE_NAME, "gaps": _gaps(record, reference)}


def _reference_row() -> dict:
    payload = read_json(RETRAIN_CV)
    require(payload is not None, "missing retrain7 cv_results.json")
    found = [row for row in payload["ranked"] if row["name"] == REFERENCE_NAME]
    require(len(found) == 1, "retrain7 control row")
    return found[0]


def _gaps(ours: dict, reference: dict) -> dict:
    gaps = {
        "mean_macro": abs(float(ours["mean_macro_iou"]) - float(reference["mean_macro_iou"])),
        "mean_precision": abs(float(ours["mean_precision"]) - float(reference["mean_precision"])),
    }
    for ours_fold, ref_fold in zip(ours["folds"], reference["folds"]):
        number = int(ours_fold["fold"])
        gaps[f"fold{number}_macro"] = abs(float(ours_fold["macro_iou"]) - float(ref_fold["macro_iou"]))
        gaps[f"fold{number}_threshold"] = abs(float(ours_fold["threshold"]) - float(ref_fold["threshold"]))
    return gaps


def _log_gaps(gaps: dict) -> None:
    worst = max(gaps.values())
    log(f"control gap worst={worst:.8f} mean_macro={gaps['mean_macro']:.8f}")


def _rank(control: dict, symbol: dict) -> list[dict]:
    ranked = [control, symbol]
    ranked.sort(key=_variant_key)
    return ranked


def _variant_key(row: dict) -> tuple:
    return (-float(row["mean_macro_iou"]), -float(row["mean_precision"]), row["name"])


def _control_signature(rows: list[dict]) -> dict:
    return {"kind": "retrain7-base", "queries": len(rows), "stamp": _retrain_stamp(rows), "setting": CONTROL, "recipe": _recipe()}


def _symbol_signature(rows: list[dict]) -> dict:
    return {
        "kind": "symbols7",
        "queries": len(rows),
        "stamp": _symbol_stamp(rows),
        "setting": SYMBOL,
        "channels": feature_names(),
        "recipe": _recipe(),
        "cell_version": CELL_VERSION,
    }


def _retrain_stamp(rows: list[dict]) -> int:
    stamps = [(RETRAIN_CELLS / f"{safe_id(row['id'])}.npz").stat().st_mtime_ns for row in rows]
    return int(max(stamps))


def _symbol_stamp(rows: list[dict]) -> int:
    stamps = [_cell_path(row["id"]).stat().st_mtime_ns for row in rows]
    return int(max(stamps))


def _cell_path(query_id: str):
    return CELL_CACHE / f"{safe_id(query_id)}.npz"
