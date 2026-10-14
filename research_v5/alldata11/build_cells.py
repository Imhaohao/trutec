"""Development stage-2 cells: original stage-1 grid, uncleaned crop descriptor, structure channels."""
from __future__ import annotations

import gc
import json
import time

import runtime  # noqa: F401

import cv2
import numpy as np

from banks import _check_grid, _load_bank, _save_bank, bank_path, load_rgb
from cells import joined_names
from constants import CELLS_PER_CLASS, DATA_ROOT, GRID_STRIDE, STAGE2_SEED
from dataset import _gather, _sample_cells
from drawing import consume_volumes
from evaluate import reviewed_domains
from evaluate_dev import assert_development_unseen, load_original
from features import feature_names as stacker_feature_names
from folds import indexed_training_order
from runtime import BANK_DIR, CELL_DIR, DEVELOPMENT_MANIFEST, guard_rss, log, require
from structure_features import FEATURE_VERSION, bank_from_rgb, component_maps, query_reference, structure_volume
from texture_learning import grid_centers

cv2.setNumThreads(4)


def build_development_cells() -> dict:
    rows = _development_rows()
    width = len(joined_names())
    require(joined_names()[0] == "cell_score", "stage1 column")
    stage1, metadata = load_original()
    assert_development_unseen(metadata)
    log(
        f"stage1 original training_documents={len(metadata['training_documents'])} "
        f"descriptor=crop uncleaned"
    )
    built = 0
    images = []
    for image, group in _groups(rows).items():
        count, seconds = _build_image(image, group, stage1, width)
        built += count
        images.append({"image": image, "queries": len(group), "built": count, "seconds": seconds})
        guard_rss()
    summary = _summary(rows)
    summary["queries_built_this_call"] = built
    summary["images"] = images
    summary["cells_per_class"] = CELLS_PER_CLASS
    summary["stage2_seed"] = STAGE2_SEED
    summary["seed_rule"] = "STAGE2_SEED + index from indexed_training_order(development-real.json)"
    summary["descriptor"] = "crop"
    summary["stage1"] = "models/models/texture-hgb-crop-v1.joblib"
    log(
        f"cells queries={summary['queries']} cells={summary['cells']} "
        f"pos={summary['positive']} neg={summary['negative']} built={built}"
    )
    return summary


def _development_rows() -> list[dict]:
    payload = json.loads(DEVELOPMENT_MANIFEST.read_text(encoding="utf-8"))
    rows = payload["examples"] if isinstance(payload, dict) else payload
    require(len(rows) == 55, f"development queries {len(rows)}")
    require(all("labels" in row for row in rows), "development row missing labels")
    return rows


def _groups(rows: list[dict]) -> dict[str, list]:
    grouped: dict[str, list] = {}
    for index, row in indexed_training_order(rows):
        grouped.setdefault(row["image"], []).append((index, row))
    return grouped


def _build_image(image: str, group, stage1, width: int) -> tuple[int, float]:
    pending = [(index, row) for index, row in group if not _cell_ready(row["id"], width)]
    if not pending:
        log(f"cells cached {image}")
        return 0, 0.0
    log(f"cells start {image} pending={len(pending)}")
    started = time.perf_counter()
    rgb = load_rgb(image)
    try:
        _score_pending(image, pending, stage1, rgb, width)
    finally:
        del rgb
        gc.collect()
    return len(pending), time.perf_counter() - started


def _score_pending(image, pending, stage1, rgb, width: int) -> None:
    bank = _acquire_bank(image, rgb)
    references = _references(rgb, pending)
    spec = {key: pending[0][1][key] for key in ("id", "image", "width", "height")}
    boxes = [row["query_box"] for _index, row in pending]
    _emit_pending(spec, boxes, pending, stage1, bank, references, width)
    del bank, references
    log(f"cells image {image} queries={len(pending)}")


def _emit_pending(spec, boxes, pending, stage1, bank, references, width: int) -> None:
    def consume(local: int, volume, pending=pending, references=references) -> None:
        index, row = pending[local]
        _save_query(index, row, volume, bank, references[local], width)

    consume_volumes(spec, boxes, [stage1] * len(boxes), consume)


def _acquire_bank(image: str, rgb: np.ndarray) -> dict:
    local = BANK_DIR / bank_path(image).name
    cached = _load_bank(local, image, rgb.shape)
    if cached is not None:
        log(f"bank local {image}")
        return cached
    cached = _load_bank(bank_path(image), image, rgb.shape)
    if cached is not None:
        log(f"bank read-only structure cache {image}")
        return cached
    bank = bank_from_rgb(rgb)
    _check_grid(bank, rgb.shape)
    _save_bank(local, image, rgb.shape, bank)
    log(f"bank built {image}")
    return bank


def _references(rgb, pending):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    area_map, side_map, ink = component_maps(rgb)
    return [query_reference(gray, row["query_box"], area_map, side_map, ink) for _index, row in pending]


def _save_query(index: int, row: dict, volume, bank, reference, width: int) -> None:
    positive, known, _blank, _meta = reviewed_domains(row, DATA_ROOT)
    centers_y, centers_x = grid_centers(int(row["height"]), int(row["width"]), GRID_STRIDE)
    sampled = _sample_cells(positive, known, centers_y, centers_x, CELLS_PER_CLASS, index)
    pos_idx, neg_idx, n_pos, n_neg = sampled
    _assert_known_only(known, centers_y, centers_x, pos_idx, row["id"])
    _assert_known_only(known, centers_y, centers_x, neg_idx, row["id"])
    del positive, known
    _write_sampled(index, row, volume, bank, reference, sampled, width)


def _write_sampled(index, row, volume, bank, reference, sampled, width: int) -> None:
    pos_idx, neg_idx, n_pos, n_neg = sampled
    base, labels = _gather(volume, pos_idx, neg_idx)
    struct, struct_labels = _gather(structure_volume(bank, reference), pos_idx, neg_idx)
    _require_pair(row, base, labels, struct, struct_labels, width)
    features = _concat(base, struct)
    _write_cell(row, features, labels, n_pos, n_neg)
    log(
        f"cells save {row['id']} index={index} seed={STAGE2_SEED + index} "
        f"sampled={len(labels)} domain_pos={n_pos} domain_neg={n_neg}"
    )


def _assert_known_only(known, centers_y, centers_x, indices, query_id: str) -> None:
    if len(indices) == 0:
        return
    ys = centers_y[indices[:, 0]]
    xs = centers_x[indices[:, 1]]
    require(bool(known[ys, xs].all()), f"unknown pixel sampled {query_id}")


def _require_pair(row, base, labels, struct, struct_labels, width: int) -> None:
    require(np.array_equal(labels, struct_labels), f"label drift {row['id']}")
    require(base.shape[0] == struct.shape[0], f"rows {row['id']}")
    require(base.shape[1] == len(stacker_feature_names()) or len(labels) == 0, f"stacker width {row['id']}")
    require(features_width(base, struct) == width or len(labels) == 0, f"joined width {row['id']}")
    require(int(np.count_nonzero(labels == 1)) <= CELLS_PER_CLASS, f"pos cap {row['id']}")
    require(int(np.count_nonzero(labels == 0)) <= CELLS_PER_CLASS, f"neg cap {row['id']}")


def features_width(base, struct) -> int:
    return int(base.shape[1] + struct.shape[1])


def _concat(base, struct) -> np.ndarray:
    if len(base) == 0:
        return np.zeros((0, features_width(base, struct)), np.float32)
    return np.concatenate([base, struct], axis=1)


def _write_cell(row, features, labels, n_pos: int, n_neg: int) -> None:
    from limits import atomic_savez

    atomic_savez(
        _cell_path(row["id"]),
        X=features,
        y=np.asarray(labels, dtype=np.int8),
        document=np.array(row["document_id"]),
        fold=np.int8(-1),
        n_pos_domain=np.int64(n_pos),
        n_neg_domain=np.int64(n_neg),
        version=np.int32(FEATURE_VERSION),
    )


def _cell_ready(query_id: str, width: int) -> bool:
    path = _cell_path(query_id)
    if not path.is_file():
        return False
    try:
        with np.load(path) as stored:
            return _stored_ok(stored, width)
    except (OSError, ValueError, KeyError):
        return False


def _stored_ok(stored, width: int) -> bool:
    if int(stored["version"]) != FEATURE_VERSION:
        return False
    features = stored["X"]
    if features.ndim != 2:
        return False
    return int(features.shape[1]) == width or int(features.shape[0]) == 0


def _cell_path(query_id: str):
    require(bool(query_id) and "/" not in query_id and "\\" not in query_id, "unsafe id")
    return CELL_DIR / f"{query_id}.npz"


def _summary(rows: list[dict]) -> dict:
    queries = cells = positive = negative = 0
    for row in rows:
        queries, cells, positive, negative = _count_one(row, queries, cells, positive, negative)
    return {"queries": queries, "cells": cells, "positive": positive, "negative": negative}


def _count_one(row, queries, cells, positive, negative):
    require(_cell_ready(row["id"], len(joined_names())), f"missing cell cache {row['id']}")
    with np.load(_cell_path(row["id"])) as stored:
        labels = stored["y"]
    return (
        queries + 1,
        cells + int(len(labels)),
        positive + int(np.count_nonzero(labels == 1)),
        negative + int(np.count_nonzero(labels == 0)),
    )
