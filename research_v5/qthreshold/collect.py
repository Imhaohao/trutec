"""Collect fit and development caches, one drawing at a time."""
from __future__ import annotations

import time
from collections import defaultdict

import limits  # noqa: F401

import joblib
import numpy as np

from constants import DATA_ROOT
from cutoffs import best_cutoffs
from devharness import development_queries, fit_rows
from evaluate import reviewed_domains
from evaluate_dev import load_original
from folds import load_document_folds
from gating import emit_gated
from limits import (
    FOLDS_PATH,
    FROZEN_STAGE2,
    IMAGE_LOG,
    STAGE1_FOLD_DIR,
    STAGE2_FOLD_DIR,
    append_jsonl,
    guard_memory,
    log,
)
from scoremap import map_features
from store import dev_ready, fit_ready, save_dev, save_fit


def collect_fit() -> None:
    folds = load_document_folds(FOLDS_PATH)
    groups = _groups(fit_rows())
    models: dict = {}
    for image, group in groups.items():
        _collect_fit_image(image, group, folds, models)
    missing = [row["id"] for row in fit_rows() if not fit_ready(row["id"])]
    if missing:
        raise RuntimeError(f"fit cache missing {len(missing)} queries")
    log(f"fit cache complete queries={len(fit_rows())}")


def collect_dev() -> None:
    stage1, _metadata = load_original()
    stage2 = joblib.load(FROZEN_STAGE2)["model"]
    groups: dict = defaultdict(list)
    for query in development_queries():
        groups[query.image].append(query)
    for image, queries in groups.items():
        _collect_dev_image(image, queries, stage1, stage2)
    missing = [query.id for query in development_queries() if not dev_ready(query.id)]
    if missing:
        raise RuntimeError(f"dev cache missing {len(missing)} queries")
    log(f"dev cache complete queries={len(development_queries())}")


def _collect_fit_image(image, group, folds, models) -> None:
    pending = [row for row in group if not fit_ready(row["id"])]
    if not pending:
        log(f"fit cached {image}")
        return
    document, fold = _document_fold(group, folds)
    payload, stage2 = _cached_fold(models, fold)
    _reject_seen_document(payload, document)
    log(f"fit start {image} queries={len(pending)} fold={fold}")
    started = time.perf_counter()
    boxes = [row["query_box"] for row in pending]
    emit_gated(_spec_row(pending[0]), boxes, payload["model"], stage2, _fit_consumer(pending, fold))
    _record(image, "fit", len(pending), time.perf_counter() - started)
    guard_memory()


def _collect_dev_image(image, queries, stage1, stage2) -> None:
    pending = [query for query in queries if not dev_ready(query.id)]
    if not pending:
        log(f"dev cached {image}")
        return
    log(f"dev start {image} queries={len(pending)}")
    started = time.perf_counter()
    boxes = [query.query_box for query in pending]
    emit_gated(_spec_query(pending[0]), boxes, stage1, stage2, _dev_consumer(pending))
    _record(image, "dev", len(pending), time.perf_counter() - started)
    guard_memory()


def _fit_consumer(pending, fold):
    def consume(index: int, gated) -> None:
        row = pending[index]
        features = map_features(gated, row["query_box"])
        positive, negative = _reviewed_scores(row, gated)
        iou_cutoff, recall_cutoff = best_cutoffs(positive, negative)
        save_fit(row["id"], features, positive, negative, iou_cutoff, recall_cutoff, row["document_id"], fold)

    return consume


def _dev_consumer(pending):
    def consume(index: int, gated) -> None:
        query = pending[index]
        features = map_features(gated, query.query_box)
        reviewed = np.array(gated[query.y, query.x], dtype=np.float32, copy=True)
        if len(reviewed) != len(query.y):
            raise RuntimeError(f"reviewed length {query.id}")
        save_dev(query.id, features, reviewed)

    return consume


def _reviewed_scores(row, gated):
    positive, known, _blank, _meta = reviewed_domains(row, DATA_ROOT)
    pos_scores = np.array(gated[positive], dtype=np.float32, copy=True)
    neg_scores = np.array(gated[known & ~positive], dtype=np.float32, copy=True)
    del positive, known
    return pos_scores, neg_scores


def _cached_fold(models: dict, fold: int):
    if fold not in models:
        models[fold] = _load_fold(fold)
    return models[fold]


def _load_fold(fold: int):
    stage1_payload = joblib.load(STAGE1_FOLD_DIR / f"fold_{fold}.joblib")
    stage2_payload = joblib.load(STAGE2_FOLD_DIR / f"fold_{fold}.joblib")
    if int(stage1_payload["held_out_fold"]) != fold or int(stage2_payload["held_out_fold"]) != fold:
        raise RuntimeError(f"fold file does not match fold {fold}")
    return stage1_payload, stage2_payload["model"]


def _reject_seen_document(stage1_payload, document: str) -> None:
    if document in set(stage1_payload["training_documents"]):
        raise RuntimeError(f"stage-1 fold saw document {document}")


def _document_fold(group, folds) -> tuple[str, int]:
    documents = {row["document_id"] for row in group}
    if len(documents) != 1:
        raise RuntimeError("one drawing spans documents")
    document = documents.pop()
    if document not in folds:
        raise RuntimeError(f"document missing from folds.json: {document}")
    assigned = {int(folds[row["document_id"]]) for row in group}
    if len(assigned) != 1:
        raise RuntimeError("one drawing spans folds")
    return document, assigned.pop()


def _groups(rows: list[dict]) -> dict:
    grouped: dict = defaultdict(list)
    for row in rows:
        grouped[row["image"]].append(row)
    return grouped


def _spec_row(row: dict) -> dict:
    return {"id": row["id"], "image": row["image"], "width": int(row["width"]), "height": int(row["height"])}


def _spec_query(query) -> dict:
    return {"id": query.id, "image": query.image, "width": int(query.width), "height": int(query.height)}


def _record(image: str, split: str, queries: int, seconds: float) -> None:
    append_jsonl(IMAGE_LOG, {"image": image, "split": split, "queries": queries, "seconds": seconds})
    log(f"{split} scored {image} queries={queries} seconds={seconds:.1f}")
