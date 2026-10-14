"""Fit-split context cells for the second stage. Labels stay in this file."""
from __future__ import annotations

import hashlib
from pathlib import Path

import bootstrap  # noqa: F401

import numpy as np

from bootstrap import atomic_savez
from constants import (
    CACHE,
    CELLS_PER_CLASS,
    DATA_ROOT,
    FEATURE_VERSION,
    GRID_STRIDE,
    STAGE2_SEED,
    log,
    require,
)
from drawing import consume_volumes
from evaluate import reviewed_domains
from features import feature_names
from folds import indexed_training_order
from texture_learning import grid_centers


def stage2_path(query_id: str) -> Path:
    require(bool(query_id) and "/" not in query_id and "\\" not in query_id, "unsafe id")
    return CACHE / "stage2" / f"{query_id}.npz"


def build_stage2_cache(rows: list[dict], document_fold: dict[str, int], payloads: dict) -> dict:
    grouped = _pending_groups(rows)
    started_queries = 0
    for image, group in grouped.items():
        started_queries += _build_image(image, group, document_fold, payloads)
    summary = _cache_summary(rows)
    summary["queries_built_this_call"] = started_queries
    log(
        f"stage2 cache queries={summary['queries']} cells={summary['cells']} "
        f"pos={summary['positive']} neg={summary['negative']}"
    )
    return summary


def iter_cached(per_class_cap: int | None):
    names = feature_names()
    for path in sorted((CACHE / "stage2").glob("*.npz")):
        yield from _load_one(path, names, per_class_cap)


def _pending_groups(rows: list[dict]) -> dict[str, list[tuple[int, dict]]]:
    grouped: dict[str, list[tuple[int, dict]]] = {}
    for index, row in indexed_training_order(rows):
        grouped.setdefault(row["image"], []).append((index, row))
    return grouped


def _build_image(image, group, document_fold, payloads) -> int:
    pending = [(index, row) for index, row in group if not stage2_path(row["id"]).is_file()]
    if not pending:
        log(f"stage2 cached {image}")
        return 0
    document = pending[0][1]["document_id"]
    _require_same_document(pending, document)
    payload = payloads[document_fold[document]]
    require(document not in set(payload["training_documents"]), f"stage1 saw {document}")
    spec = {key: pending[0][1][key] for key in ("id", "image", "width", "height")}
    boxes = [row["query_box"] for _index, row in pending]
    captured = list(pending)

    def consume(index: int, volume, captured=captured) -> None:
        _save_query(captured[index][0], captured[index][1], volume)

    consume_volumes(spec, boxes, [payload["model"]] * len(boxes), consume)
    log(f"stage2 image {image} queries={len(pending)}")
    return len(pending)


def _save_query(index: int, row: dict, volume: np.ndarray) -> None:
    positive, known, _blank, _meta = reviewed_domains(row, DATA_ROOT)
    centers_y, centers_x = grid_centers(int(row["height"]), int(row["width"]), GRID_STRIDE)
    pos_idx, neg_idx, n_pos, n_neg = _sample_cells(
        positive, known, centers_y, centers_x, CELLS_PER_CLASS, index
    )
    del positive, known
    features, labels = _gather(volume, pos_idx, neg_idx)
    _write_cache(stage2_path(row["id"]), features, labels, row["document_id"], n_pos, n_neg)


def _sample_cells(positive, known, centers_y, centers_x, per_class, index):
    known_grid = known[centers_y[:, None], centers_x[None, :]]
    pos_grid = positive[centers_y[:, None], centers_x[None, :]] & known_grid
    neg_grid = known_grid & ~pos_grid
    rng = np.random.default_rng(STAGE2_SEED + index)
    pos_idx = _pick(np.argwhere(pos_grid), per_class, rng)
    neg_idx = _pick(np.argwhere(neg_grid), per_class, rng)
    return pos_idx, neg_idx, int(np.count_nonzero(pos_grid)), int(np.count_nonzero(neg_grid))


def _gather(volume, pos_idx, neg_idx):
    blocks = []
    labels = []
    _append_block(blocks, labels, volume, pos_idx, 1)
    _append_block(blocks, labels, volume, neg_idx, 0)
    width = int(volume.shape[-1])
    if not blocks:
        return np.zeros((0, width), np.float32), np.zeros((0,), np.int8)
    return np.concatenate(blocks, axis=0), np.concatenate(labels, axis=0)


def _append_block(blocks, labels, volume, indices, label: int) -> None:
    if len(indices) == 0:
        return
    blocks.append(np.asarray(volume[indices[:, 0], indices[:, 1]], dtype=np.float32))
    labels.append(np.full(len(indices), label, dtype=np.int8))


def _pick(candidates: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    if len(candidates) <= count:
        return candidates
    chosen = rng.choice(len(candidates), size=count, replace=False)
    chosen.sort()
    return candidates[chosen]


def _write_cache(path, features, labels, document, n_pos, n_neg) -> None:
    atomic_savez(
        path,
        X=features,
        y=labels,
        document=np.array(document),
        n_pos_domain=np.int64(n_pos),
        n_neg_domain=np.int64(n_neg),
        version=np.int32(FEATURE_VERSION),
    )


def _load_one(path: Path, names: list[str], per_class_cap: int | None):
    with np.load(path) as stored:
        require(int(stored["version"]) == FEATURE_VERSION, f"stale cache {path.name}")
        features = np.array(stored["X"], dtype=np.float32)
        labels = np.array(stored["y"], dtype=np.int8)
        document = str(stored["document"].item())
        n_pos = int(stored["n_pos_domain"])
        n_neg = int(stored["n_neg_domain"])
    if len(labels) == 0:
        return
    require(features.shape[1] == len(names), f"feature width {path.name}")
    features, labels = _cap_classes(features, labels, per_class_cap, path.stem)
    pop = _population_weights(labels, n_pos, n_neg)
    yield features, labels, document, pop


def _cap_classes(features, labels, per_class_cap, stem: str):
    if per_class_cap is None or len(labels) <= per_class_cap * 2:
        return features, labels
    rng = np.random.default_rng(_stem_seed(stem))
    pos = _pick(np.flatnonzero(labels == 1), per_class_cap, rng)
    neg = _pick(np.flatnonzero(labels == 0), per_class_cap, rng)
    chosen = np.sort(np.concatenate([pos, neg]))
    return features[chosen], labels[chosen]


def _population_weights(labels, n_pos_domain: int, n_neg_domain: int) -> np.ndarray:
    weights = np.ones(len(labels), dtype=np.float64)
    n_pos = int(np.count_nonzero(labels == 1))
    n_neg = int(np.count_nonzero(labels == 0))
    if n_pos:
        weights[labels == 1] = n_pos_domain / n_pos
    if n_neg:
        weights[labels == 0] = n_neg_domain / n_neg
    return weights


def _cache_summary(rows: list[dict]) -> dict:
    queries = cells = positive = negative = 0
    for row in rows:
        path = stage2_path(row["id"])
        require(path.is_file(), f"missing stage2 cache {row['id']}")
        with np.load(path) as stored:
            labels = stored["y"]
            queries += 1
            cells += int(len(labels))
            positive += int(np.count_nonzero(labels == 1))
            negative += int(np.count_nonzero(labels == 0))
    return {"queries": queries, "cells": cells, "positive": positive, "negative": negative}


def _require_same_document(pending, document: str) -> None:
    for _index, row in pending:
        require(row["document_id"] == document, "image spans documents")


def _stem_seed(stem: str) -> int:
    digest = hashlib.sha256(f"cv:{stem}".encode()).digest()
    return int.from_bytes(digest[:4], "little")
