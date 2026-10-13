"""Rebuild stage-2 cells for fit, development, and synthetic queries.

Sampling caps and the 75 stacker channels plus structure channels follow
research_v5/structure/cells.py. The stage-1 grid uses the fold model that
held the query document out, and the uncleaned crop descriptor inside
drawing.consume_volumes. The sample seed is the row's index inside its own
manifest.
"""
from __future__ import annotations

import gc
import time

import runtime  # noqa: F401

import cv2
import joblib
import numpy as np

from banks import _check_grid, _load_bank, _save_bank, bank_path, load_rgb
from bootstrap import atomic_savez
from cells import joined_names
from constants import CELLS_PER_CLASS, GRID_STRIDE, STAGE2_SEED
from dataset import _gather, _sample_cells
from drawing import consume_volumes
from evaluate import reviewed_domains
from features import feature_names as stacker_feature_names
from manifests import cell_index_map, data_root_for, development_ids, fit_ids
from runtime import (
    BANK_DIR,
    CELL_DEV,
    CELL_FIT,
    CELL_SYN,
    FOLD_DIR,
    STRUCTURE_CELLS,
    guard_rss,
    inside,
    log,
    require,
    sha256_file,
)
from stage1_train import PAIR_WIDTH
from structure_features import FEATURE_VERSION, bank_from_rgb, component_maps, query_reference, structure_volume
from texture_learning import grid_centers

cv2.setNumThreads(4)

CELL_RECIPE = "own-manifest-uncleaned-crop"
SEED_RULE = (
    "fit: STAGE2_SEED + index from indexed_training_order(fit rows); "
    "development: STAGE2_SEED + index from indexed_training_order(development rows); "
    "synthetic: STAGE2_SEED + index from indexed_training_order(synthetic rows)"
)
STACKER_WIDTH = 75
FIT_CELLS = (357, 1030140, 471096, 559044)


def prove_fit_sampling(rows: list[dict]) -> None:
    """Fail fast if fit cell coordinates would leave the structure recipe."""
    indexes = cell_index_map()
    checked = 0
    owned = fit_ids()
    for row in rows:
        if row["id"] not in owned:
            continue
        index = indexes[row["id"]]
        log(f"prove {row['id']} index={index}")
        labels, n_pos, n_neg = _sampled_labels(index, row)
        _match_structure(row, labels, n_pos, n_neg)
        checked += 1
        if checked == 2:
            break
    require(checked == 2, "fit sampling proof saw fewer than 2 queries")
    log("fit sampling proof ok")


def build_cells(rows: list[dict], document_fold: dict[str, int]) -> dict:
    require(len(stacker_feature_names()) == STACKER_WIDTH, "stacker width")
    require(len(joined_names()) == 144, f"joined width {len(joined_names())}")
    models, shas = _load_models()
    owned = fit_ids()
    built = 0
    images = []
    for image, group in _groups(rows).items():
        count, seconds = _build_image(image, group, document_fold, models, shas, owned)
        built += count
        images.append({"image": image, "queries": len(group), "built": count, "seconds": seconds})
        guard_rss()
    del models
    gc.collect()
    summary = _summary(rows, owned, shas)
    summary["queries_built_this_call"] = built
    summary["images"] = images
    log(
        f"cells queries={summary['queries']} cells={summary['cells']} "
        f"pos={summary['positive']} neg={summary['negative']} built={built}"
    )
    return summary


def _groups(rows: list[dict]) -> dict[str, list]:
    indexes = cell_index_map()
    grouped: dict[str, list] = {}
    for row in rows:
        grouped.setdefault(row["image"], []).append((indexes[row["id"]], row))
    return grouped


def _load_models() -> tuple[dict, dict]:
    models = {}
    shas = {}
    for fold in range(4):
        path = FOLD_DIR / f"fold_{fold}.joblib"
        payload = joblib.load(path)
        require(int(payload["held_out_fold"]) == fold, f"fold file {fold}")
        require(int(payload["model"].n_features_in_) == PAIR_WIDTH, f"fold width {fold}")
        models[fold] = payload
        shas[fold] = sha256_file(path)
    return models, shas


def _build_image(image, group, document_fold, models, shas, owned) -> tuple[int, float]:
    pending = [item for item in group if not _cell_ready(item[1], owned, shas[document_fold[item[1]["document_id"]]])]
    if not pending:
        log(f"cells cached {image}")
        return 0, 0.0
    _bind_root(_one_root(pending))
    log(f"cells start {image} pending={len(pending)}")
    started = time.perf_counter()
    model, fold = _model_for(pending, document_fold, models)
    _emit(image, pending, model, fold, shas[fold], owned)
    return len(pending), time.perf_counter() - started


def _one_root(pending):
    roots = {data_root_for(row) for _index, row in pending}
    require(len(roots) == 1, "image has mixed data roots")
    return next(iter(roots))


def _bind_root(root) -> None:
    import banks
    import drawing

    drawing.DATA_ROOT = root
    banks.DATA_ROOT = root


def _model_for(pending, document_fold, models):
    document = pending[0][1]["document_id"]
    for _index, row in pending:
        require(row["document_id"] == document, "image spans documents")
    fold = int(document_fold[document])
    payload = models[fold]
    require(document not in set(payload["training_documents"]), f"stage1 saw {document}")
    return payload["model"], fold


def _emit(image, pending, model, fold, sha: str, owned) -> None:
    rgb = load_rgb(image)
    try:
        bank = _acquire_bank(image, rgb)
        references = _references(rgb, pending)
    finally:
        del rgb
        gc.collect()
    spec = {key: pending[0][1][key] for key in ("id", "image", "width", "height")}
    boxes = [row["query_box"] for _index, row in pending]
    _consume(spec, boxes, pending, model, bank, references, fold, sha, owned)
    del bank, references
    gc.collect()
    log(f"cells image {image} queries={len(pending)}")


def _consume(spec, boxes, pending, model, bank, references, fold, sha, owned) -> None:
    def consume(local: int, volume, pending=pending, references=references) -> None:
        index, row = pending[local]
        _save_query(index, row, volume, bank, references[local], fold, sha, owned)

    consume_volumes(spec, boxes, [model] * len(boxes), consume)


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
    return _build_bank(local, image, rgb)


def _build_bank(local, image: str, rgb: np.ndarray) -> dict:
    bank = bank_from_rgb(rgb)
    _check_grid(bank, rgb.shape)
    inside(local)
    _save_bank(local, image, rgb.shape, bank)
    log(f"bank built {image}")
    return bank


def _references(rgb, pending):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    area_map, side_map, ink = component_maps(rgb)
    references = [query_reference(gray, row["query_box"], area_map, side_map, ink) for _index, row in pending]
    del gray, area_map, side_map, ink
    return references


def _save_query(index, row, volume, bank, reference, fold, sha, owned) -> None:
    positive, known, _blank, _meta = reviewed_domains(row, data_root_for(row))
    centers_y, centers_x = grid_centers(int(row["height"]), int(row["width"]), GRID_STRIDE)
    pos_idx, neg_idx, n_pos, n_neg = _sample_cells(positive, known, centers_y, centers_x, CELLS_PER_CLASS, index)
    _assert_known_only(known, centers_y, centers_x, pos_idx, row["id"])
    _assert_known_only(known, centers_y, centers_x, neg_idx, row["id"])
    expected = _label_vector(pos_idx, neg_idx)
    if row["id"] in owned:
        _match_structure(row, expected, n_pos, n_neg)
    del positive, known
    _write_joined(index, row, volume, bank, reference, (pos_idx, neg_idx, n_pos, n_neg), expected, fold, sha, owned)


def _write_joined(index, row, volume, bank, reference, sampled, expected, fold, sha, owned) -> None:
    pos_idx, neg_idx, n_pos, n_neg = sampled
    base, labels = _gather(volume, pos_idx, neg_idx)
    struct, struct_labels = _gather(structure_volume(bank, reference), pos_idx, neg_idx)
    _require_pair(row, base, labels, struct, struct_labels, expected)
    features = _concat(base, struct)
    _write_cell(row, features, labels, n_pos, n_neg, fold, sha, index, owned)
    log(
        f"cells save {row['id']} index={index} seed={STAGE2_SEED + index} "
        f"sampled={len(labels)} domain_pos={n_pos} domain_neg={n_neg}"
    )


def _sampled_labels(index: int, row: dict):
    positive, known, _blank, _meta = reviewed_domains(row, data_root_for(row))
    centers_y, centers_x = grid_centers(int(row["height"]), int(row["width"]), GRID_STRIDE)
    pos_idx, neg_idx, n_pos, n_neg = _sample_cells(positive, known, centers_y, centers_x, CELLS_PER_CLASS, index)
    del positive, known
    return _label_vector(pos_idx, neg_idx), n_pos, n_neg


def _match_structure(row, labels, n_pos: int, n_neg: int) -> None:
    path = STRUCTURE_CELLS / f"{row['id']}.npz"
    require(path.is_file(), f"missing structure cell {row['id']}")
    with np.load(path) as stored:
        cached = np.array(stored["y"])
        cached_pos = int(stored["n_pos_domain"])
        cached_neg = int(stored["n_neg_domain"])
    require(np.array_equal(labels, cached), f"sampling drift {row['id']} {len(labels)} vs {len(cached)}")
    require(cached_pos == n_pos and cached_neg == n_neg, f"domain drift {row['id']}")


def _label_vector(pos_idx, neg_idx) -> np.ndarray:
    parts = []
    if len(pos_idx):
        parts.append(np.ones(len(pos_idx), dtype=np.int8))
    if len(neg_idx):
        parts.append(np.zeros(len(neg_idx), dtype=np.int8))
    if not parts:
        return np.zeros((0,), dtype=np.int8)
    return np.concatenate(parts)


def _assert_known_only(known, centers_y, centers_x, indices, query_id: str) -> None:
    if len(indices) == 0:
        return
    ys = centers_y[indices[:, 0]]
    xs = centers_x[indices[:, 1]]
    require(bool(known[ys, xs].all()), f"unknown pixel sampled {query_id}")


def _require_pair(row, base, labels, struct, struct_labels, expected) -> None:
    require(np.array_equal(labels, struct_labels), f"label drift {row['id']}")
    require(np.array_equal(labels, expected), f"gather drift {row['id']}")
    require(base.shape[0] == struct.shape[0], f"rows {row['id']}")
    require(base.shape[1] == STACKER_WIDTH or len(labels) == 0, f"stacker width {row['id']}")
    require(features_width(base, struct) == len(joined_names()) or len(labels) == 0, f"joined width {row['id']}")
    require(int(np.count_nonzero(labels == 1)) <= CELLS_PER_CLASS, f"pos cap {row['id']}")
    require(int(np.count_nonzero(labels == 0)) <= CELLS_PER_CLASS, f"neg cap {row['id']}")


def features_width(base, struct) -> int:
    return int(base.shape[1] + struct.shape[1])


def _concat(base, struct) -> np.ndarray:
    if len(base) == 0:
        return np.zeros((0, features_width(base, struct)), np.float32)
    return np.concatenate([base, struct], axis=1)


def _write_cell(row, features, labels, n_pos, n_neg, fold, sha, index, owned) -> None:
    atomic_savez(
        _cell_path(row, owned),
        X=features,
        y=np.asarray(labels, dtype=np.int8),
        document=np.array(row["document_id"]),
        fold=np.int8(fold),
        n_pos_domain=np.int64(n_pos),
        n_neg_domain=np.int64(n_neg),
        version=np.int32(FEATURE_VERSION),
        stage1_sha=np.array(sha),
        sample_index=np.int32(index),
        recipe=np.array(CELL_RECIPE),
    )


def _cell_path(row, owned):
    _require_safe_id(row["id"])
    path = _cell_dir(row, owned) / f"{row['id']}.npz"
    inside(path)
    return path


def _cell_dir(row, owned):
    if row["id"] in owned:
        return CELL_FIT
    if row["id"] in development_ids():
        return CELL_DEV
    return CELL_SYN


def _cell_ready(row, owned, sha: str) -> bool:
    path = _cell_path(row, owned)
    if not path.is_file():
        return False
    try:
        with np.load(path) as stored:
            return _stored_ok(stored, sha)
    except (OSError, ValueError, KeyError):
        return False


def _stored_ok(stored, sha: str) -> bool:
    if int(stored["version"]) != FEATURE_VERSION:
        return False
    if str(stored["recipe"].item()) != CELL_RECIPE:
        return False
    if str(stored["stage1_sha"].item()) != sha:
        return False
    features = stored["X"]
    if features.ndim != 2:
        return False
    return int(features.shape[1]) == len(joined_names()) or int(features.shape[0]) == 0


def _summary(rows, owned, shas) -> dict:
    buckets = {name: _empty_bucket() for name in ("fit", "development", "synthetic")}
    dev_ids = development_ids()
    for row in rows:
        _count_row(row, owned, dev_ids, shas, buckets)
    _require_fit_totals(buckets["fit"])
    _require_development_queries(buckets["development"])
    return _summary_payload(buckets, shas)


def _summary_payload(buckets: dict, shas: dict) -> dict:
    fit = buckets["fit"]
    development = buckets["development"]
    synthetic = buckets["synthetic"]
    return {
        "queries": fit["queries"] + development["queries"] + synthetic["queries"],
        "cells": fit["cells"] + development["cells"] + synthetic["cells"],
        "positive": fit["positive"] + development["positive"] + synthetic["positive"],
        "negative": fit["negative"] + development["negative"] + synthetic["negative"],
        "fit": fit,
        "development": development,
        "synthetic": synthetic,
        "cells_per_class": CELLS_PER_CLASS,
        "stage2_seed": STAGE2_SEED,
        "seed_rule": SEED_RULE,
        "descriptor": "crop",
        "descriptor_cleaning": "uncleaned",
        "cell_recipe": CELL_RECIPE,
        "stage1_fold_sha256": {str(fold): digest for fold, digest in shas.items()},
    }


def _count_row(row, owned, dev_ids, shas, buckets) -> None:
    labels = _ready_labels(row, owned, shas)
    bucket = buckets[_split_name(row, owned, dev_ids)]
    bucket["queries"] += 1
    bucket["cells"] += int(len(labels))
    bucket["positive"] += int(np.count_nonzero(labels == 1))
    bucket["negative"] += int(np.count_nonzero(labels == 0))


def _split_name(row, owned, dev_ids) -> str:
    if row["id"] in owned:
        return "fit"
    if row["id"] in dev_ids:
        return "development"
    return "synthetic"


def _ready_labels(row, owned, shas):
    path = _cell_path(row, owned)
    require(path.is_file(), f"missing cell cache {row['id']}")
    with np.load(path) as stored:
        fold = int(stored["fold"])
        require(_stored_ok(stored, shas[fold]), f"stale cell cache {row['id']}")
        return np.array(stored["y"])


def _require_fit_totals(fit: dict) -> None:
    require((fit["queries"], fit["cells"], fit["positive"], fit["negative"]) == FIT_CELLS, f"fit cells {fit}")


def _require_development_queries(development: dict) -> None:
    require(development["queries"] == 55, f"development queries {development['queries']}")


def _empty_bucket() -> dict:
    return {"queries": 0, "cells": 0, "positive": 0, "negative": 0}


def _require_safe_id(query_id: str) -> None:
    require(bool(query_id) and "/" not in query_id and "\\" not in query_id, "unsafe id")
    require(not query_id.startswith("."), "unsafe id")
