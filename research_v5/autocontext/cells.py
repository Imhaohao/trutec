"""Fit-split cell sampler. Labels are read only here, and only on the fit split.

Unknown pixels are never negatives: a cell is negative only when its center
pixel is reviewed and not positive. Sampling matches the stacker: up to 2000
cells per class, seed = stage-2 seed + the query's training-order index.
"""
from __future__ import annotations

import hashlib

import limits  # noqa: F401

import numpy as np

from bootstrap import atomic_savez
from constants import CELLS_PER_CLASS, DATA_ROOT, GRID_STRIDE, STAGE2_SEED
from context_features import FEATURE_VERSION, feature_names
from evaluate import reviewed_domains
from limits import CELL_CACHE, require
from texture_learning import grid_centers


def cell_path(query_id: str):
    require(bool(query_id) and "/" not in query_id and "\\" not in query_id, "unsafe id")
    return CELL_CACHE / f"{query_id}.npz"


def cell_ready(query_id: str) -> bool:
    path = cell_path(query_id)
    if not path.is_file():
        return False
    return _cache_matches(path)


def cell_indices(row: dict, index: int):
    positive, known, _blank, _meta = reviewed_domains(row, DATA_ROOT)
    centers_y, centers_x = grid_centers(int(row["height"]), int(row["width"]), GRID_STRIDE)
    sampled = sample_indices(
        positive, known, centers_y, centers_x, CELLS_PER_CLASS, STAGE2_SEED + index
    )
    del positive, known
    return centers_y, centers_x, sampled


def sample_indices(positive, known, centers_y, centers_x, per_class: int, seed: int):
    known_grid = known[centers_y[:, None], centers_x[None, :]]
    pos_grid = positive[centers_y[:, None], centers_x[None, :]] & known_grid
    neg_grid = known_grid & ~pos_grid
    rng = np.random.default_rng(seed)
    pos_idx = pick(np.argwhere(pos_grid), per_class, rng)
    neg_idx = pick(np.argwhere(neg_grid), per_class, rng)
    return pos_idx, neg_idx, int(np.count_nonzero(pos_grid)), int(np.count_nonzero(neg_grid))


def gather_labeled(volume: np.ndarray, pos_idx: np.ndarray, neg_idx: np.ndarray):
    blocks: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    _append(blocks, labels, volume, pos_idx, 1)
    _append(blocks, labels, volume, neg_idx, 0)
    width = int(volume.shape[-1])
    if not blocks:
        return np.zeros((0, width), np.float32), np.zeros((0,), np.int8)
    return np.concatenate(blocks, axis=0), np.concatenate(labels, axis=0)


def write_cell_cache(query_id: str, features, labels, document: str, fold: int, n_pos: int, n_neg: int) -> None:
    require(features.shape[0] == labels.shape[0], "sample length")
    require(features.shape[1] == len(feature_names()) or len(labels) == 0, "feature width")
    if len(labels):
        require(bool(np.isfinite(features).all()), "non-finite stage3 features")
    atomic_savez(
        cell_path(query_id),
        X=np.asarray(features, dtype=np.float32),
        y=np.asarray(labels, dtype=np.int8),
        document=np.array(document),
        fold=np.int8(fold),
        n_pos_domain=np.int64(n_pos),
        n_neg_domain=np.int64(n_neg),
        version=np.int32(FEATURE_VERSION),
    )


def iter_cached(per_class_cap: int | None):
    names = feature_names()
    for path in sorted(CELL_CACHE.glob("*.npz")):
        yield from _load_one(path, names, per_class_cap)


def pick(candidates: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    if len(candidates) <= count:
        return candidates
    chosen = rng.choice(len(candidates), size=count, replace=False)
    chosen.sort()
    return candidates[chosen]


def _append(blocks, labels, volume, indices, label: int) -> None:
    if len(indices) == 0:
        return
    blocks.append(np.asarray(volume[indices[:, 0], indices[:, 1]], dtype=np.float32))
    labels.append(np.full(len(indices), label, dtype=np.int8))


def _cache_matches(path) -> bool:
    try:
        with np.load(path) as stored:
            return _stored_ok(stored)
    except (OSError, ValueError, KeyError):
        return False


def _stored_ok(stored) -> bool:
    if int(stored["version"]) != FEATURE_VERSION:
        return False
    features = stored["X"]
    if features.ndim != 2:
        return False
    return int(features.shape[1]) == len(feature_names())


def _load_one(path, names: list[str], per_class_cap: int | None):
    with np.load(path) as stored:
        require(int(stored["version"]) == FEATURE_VERSION, f"stale cache {path.name}")
        features = np.array(stored["X"], dtype=np.float32)
        labels = np.array(stored["y"], dtype=np.int8)
        document = str(stored["document"].item())
        fold = int(stored["fold"])
        n_pos = int(stored["n_pos_domain"])
        n_neg = int(stored["n_neg_domain"])
    if len(labels) == 0:
        return
    require(features.shape[1] == len(names), f"feature width {path.name}")
    features, labels = cap_classes(features, labels, per_class_cap, path.stem)
    yield features, labels, document, fold, population_weights(labels, n_pos, n_neg)


def cap_classes(features, labels, per_class_cap, stem: str):
    if per_class_cap is None or len(labels) <= per_class_cap * 2:
        return features, labels
    rng = np.random.default_rng(_stem_seed(stem))
    chosen = _capped_index(labels, per_class_cap, rng)
    return features[chosen], labels[chosen]


def population_weights(labels, n_pos_domain: int, n_neg_domain: int) -> np.ndarray:
    weights = np.ones(len(labels), dtype=np.float64)
    n_pos = int(np.count_nonzero(labels == 1))
    n_neg = int(np.count_nonzero(labels == 0))
    if n_pos:
        weights[labels == 1] = n_pos_domain / n_pos
    if n_neg:
        weights[labels == 0] = n_neg_domain / n_neg
    return weights


def _capped_index(labels, per_class_cap: int, rng) -> np.ndarray:
    pos = pick(np.flatnonzero(labels == 1), per_class_cap, rng)
    neg = pick(np.flatnonzero(labels == 0), per_class_cap, rng)
    return np.sort(np.concatenate([pos, neg]))


def _stem_seed(stem: str) -> int:
    digest = hashlib.sha256(f"cv:{stem}".encode()).digest()
    return int.from_bytes(digest[:4], "little")
