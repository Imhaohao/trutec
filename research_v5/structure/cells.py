"""Join stacker stage-2 cells with structure features. Labels stay in this file."""
from __future__ import annotations

import gc

import limits  # noqa: F401

import cv2
import numpy as np

from banks import ensure_bank, load_rgb
from constants import CELLS_PER_CLASS, DATA_ROOT, GRID_STRIDE
from dataset import _gather, _sample_cells, stage2_path
from evaluate import reviewed_domains
from features import feature_names as stacker_feature_names
from folds import indexed_training_order
from limits import CELL_CACHE, atomic_savez, current_gb, guard_memory, log, require
from structure_features import (
    FEATURE_VERSION,
    component_maps,
    feature_names,
    query_reference,
    structure_volume,
)
from texture_learning import grid_centers

STACKER_VERSION = 1


def build_cell_cache(rows: list[dict], document_fold: dict[str, int]) -> dict:
    built = 0
    for image, group in _groups(rows).items():
        built += _build_image(image, group, document_fold)
        guard_memory()
    summary = _summary(rows)
    summary["queries_built_this_call"] = built
    log(
        f"cells queries={summary['queries']} cells={summary['cells']} "
        f"pos={summary['positive']} neg={summary['negative']} built={built}"
    )
    return summary


def joined_names() -> list[str]:
    names = stacker_feature_names() + feature_names()
    require(names[0] == "cell_score", "stage1 column")
    require(len(names) == len(set(names)), "duplicate joined name")
    return names


def _groups(rows: list[dict]) -> dict[str, list]:
    grouped: dict[str, list] = {}
    for index, row in indexed_training_order(rows):
        grouped.setdefault(row["image"], []).append((index, row))
    return grouped


def _build_image(image: str, group, document_fold) -> int:
    pending = [(index, row) for index, row in group if not cell_ready(row["id"])]
    if not pending:
        log(f"cells cached {image}")
        return 0
    log(f"cells start {image} pending={len(pending)}")
    rgb = load_rgb(image)
    try:
        return _score_pending(image, pending, document_fold, rgb)
    finally:
        del rgb
        gc.collect()


def _score_pending(image, pending, document_fold, rgb) -> int:
    bank = ensure_bank(image, rgb)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    area_map, side_map, ink = component_maps(rgb)
    for index, row in pending:
        reference = query_reference(gray, row["query_box"], area_map, side_map, ink)
        _save_query(index, row, bank, reference, document_fold)
    del gray, area_map, side_map, ink, bank
    log(f"cells image {image} queries={len(pending)} rss_gb={current_gb():.2f}")
    return len(pending)


def _save_query(index: int, row: dict, bank, reference, document_fold) -> None:
    positive, known, _blank, _meta = reviewed_domains(row, DATA_ROOT)
    centers_y, centers_x = grid_centers(int(row["height"]), int(row["width"]), GRID_STRIDE)
    sampled = _sample_cells(positive, known, centers_y, centers_x, CELLS_PER_CLASS, index)
    del positive, known
    pos_idx, neg_idx, n_pos, n_neg = sampled
    volume = structure_volume(bank, reference)
    struct, labels = _gather(volume, pos_idx, neg_idx)
    del volume
    _write_joined(row, struct, labels, document_fold, n_pos, n_neg)


def cell_ready(query_id: str) -> bool:
    path = _cell_path(query_id)
    if not path.is_file():
        return False
    try:
        with np.load(path) as stored:
            return _stored_ok(stored)
    except (OSError, ValueError, KeyError):
        return False


def _write_joined(row, struct, labels, document_fold, n_pos: int, n_neg: int) -> None:
    base, cached_y, document, cached_pos, cached_neg = _read_stacker(row)
    require(document == row["document_id"], f"document {row['id']}")
    require(np.array_equal(labels, cached_y), f"sampling drift {row['id']}")
    require(cached_pos == n_pos and cached_neg == n_neg, f"domain counts {row['id']}")
    require(base.shape[0] == struct.shape[0], f"rows {row['id']}")
    features = _concat(base, struct)
    require(features.shape[1] == len(joined_names()) or len(labels) == 0, "joined width")
    atomic_savez(
        _cell_path(row["id"]),
        X=features,
        y=np.asarray(labels, dtype=np.int8),
        document=np.array(document),
        fold=np.int8(document_fold[document]),
        n_pos_domain=np.int64(n_pos),
        n_neg_domain=np.int64(n_neg),
        version=np.int32(FEATURE_VERSION),
    )


def _read_stacker(row):
    with np.load(stage2_path(row["id"])) as stored:
        require(int(stored["version"]) == STACKER_VERSION, f"stacker version {row['id']}")
        base = np.array(stored["X"], dtype=np.float32)
        labels = np.array(stored["y"], dtype=np.int8)
        document = str(stored["document"].item())
        n_pos = int(stored["n_pos_domain"])
        n_neg = int(stored["n_neg_domain"])
    require(base.shape[1] == len(stacker_feature_names()), f"base width {row['id']}")
    return base, labels, document, n_pos, n_neg


def _concat(base, struct) -> np.ndarray:
    if len(base) == 0:
        return np.zeros((0, base.shape[1] + struct.shape[1]), np.float32)
    return np.concatenate([base, struct], axis=1)


def _summary(rows: list[dict]) -> dict:
    queries = cells = positive = negative = 0
    for row in rows:
        queries, cells, positive, negative = _count_one(row, queries, cells, positive, negative)
    return {"queries": queries, "cells": cells, "positive": positive, "negative": negative}


def _count_one(row, queries, cells, positive, negative):
    require(cell_ready(row["id"]), f"missing cell cache {row['id']}")
    with np.load(_cell_path(row["id"])) as stored:
        labels = stored["y"]
    return (
        queries + 1,
        cells + int(len(labels)),
        positive + int(np.count_nonzero(labels == 1)),
        negative + int(np.count_nonzero(labels == 0)),
    )


def _cell_path(query_id: str):
    require(bool(query_id) and "/" not in query_id and "\\" not in query_id, "unsafe id")
    return CELL_CACHE / f"{query_id}.npz"


def _stored_ok(stored) -> bool:
    if int(stored["version"]) != FEATURE_VERSION:
        return False
    features = stored["X"]
    if features.ndim != 2:
        return False
    return int(features.shape[1]) == len(joined_names())
