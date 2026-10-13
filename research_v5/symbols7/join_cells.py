"""Join retrain7 cleaned stage-2 cells with overlay-symbol channels.

Sampling is the fit-manifest index, seed 42 plus that index, and 2000 cells
per class. The prefix of every saved row is the retrain7 base matrix, unread
from its double sample.
"""
from __future__ import annotations

import gc
import time

import guard

guard.activate()

import numpy as np

from constants import CELLS_PER_CLASS, DATA_ROOT, GRID_STRIDE
from dataset import _gather, _sample_cells
from devharness import fit_rows
from drawing import _load_rgb
from evaluate import reviewed_domains
from folds import indexed_training_order
from guard import (
    BASE_WIDTH,
    CELL_CACHE,
    CELL_VERSION,
    RETRAIN_CELLS,
    SUMMARY_PATH,
    WINDOW_CACHE,
    WINDOW_VERSION,
    atomic_savez,
    current_gb,
    guard_memory,
    log,
    require,
    safe_id,
    write_json,
)
from inference import read_drawing
from symbol_features import CHANNEL_COUNT, channels_from_cube, feature_names, query_stats, window_cube
from texture_learning import grid_centers


def cached_window_cube(image: str, rgb: np.ndarray) -> np.ndarray:
    return _cached_cube(image, rgb)


def build_all(rows: list[dict], limit: int = 0) -> dict:
    require(CELLS_PER_CLASS == 2000, f"cells per class {CELLS_PER_CLASS}")
    require(len(feature_names()) == CHANNEL_COUNT, "channel count")
    grouped = _groups(rows)
    images = _by_area(grouped)
    if limit:
        images = images[:limit]
    slowest = 0.0
    for image in images:
        seconds = _build_image(image, grouped[image])
        slowest = max(slowest, seconds)
        guard_memory()
    if limit:
        log(f"smoke built {len(images)} images slowest={slowest:.1f}")
        return {"smoke": len(images), "slowest_window_seconds": slowest}
    summary = _summarize(rows, slowest)
    write_json(SUMMARY_PATH, summary)
    log(f"cells queries={summary['queries']} joined={summary['cells']} slowest={slowest:.1f}")
    return summary


def _build_image(image: str, group) -> float:
    pending = [pair for pair in group if not _cell_ready(pair[1]["id"])]
    if not pending:
        log(f"cells cached {image}")
        return 0.0
    started = time.perf_counter()
    rgb = _load_rgb(image)
    cube = _cached_cube(image, rgb)
    del rgb
    gray = _gray(image, pending[0][1])
    try:
        for global_index, row in pending:
            _join_one(global_index, row, gray, cube)
    finally:
        del gray, cube
        gc.collect()
    seconds = time.perf_counter() - started
    log(f"cells image {image} queries={len(pending)} seconds={seconds:.1f} rss_gb={current_gb():.2f}")
    return seconds


def _join_one(global_index: int, row: dict, gray, cube) -> None:
    symbol, labels, n_pos, n_neg = _symbol_rows(global_index, row, gray, cube)
    base = _read_retrain(row["id"])
    _require_alignment(row["id"], base, labels, symbol, n_pos, n_neg)
    features = _concat(base["features"], symbol)
    _write_cell(row, features, labels, base)


def _symbol_rows(global_index: int, row: dict, gray, cube):
    stats = query_stats(gray, row["query_box"])
    volume = channels_from_cube(cube, stats)
    positive, known, _blank, _meta = reviewed_domains(row, DATA_ROOT)
    centers_y, centers_x = grid_centers(int(row["height"]), int(row["width"]), GRID_STRIDE)
    pos_idx, neg_idx, n_pos, n_neg = _sample_cells(
        positive, known, centers_y, centers_x, CELLS_PER_CLASS, global_index
    )
    del positive, known
    symbol, labels = _gather(volume, pos_idx, neg_idx)
    del volume
    return symbol, labels, n_pos, n_neg


def _cached_cube(image: str, rgb: np.ndarray) -> np.ndarray:
    path = _window_path(image)
    cached = _read_cube(path, rgb.shape)
    if cached is not None:
        return cached
    started = time.perf_counter()
    cube = window_cube(rgb)
    seconds = time.perf_counter() - started
    if seconds > 60.0:
        log(f"windows slow {image} seconds={seconds:.1f}")
    require(seconds < 90.0, f"symbol windows too slow {image} {seconds:.1f}s")
    atomic_savez(
        path,
        cube=np.asarray(cube, dtype=np.float32),
        height=np.int32(rgb.shape[0]),
        width=np.int32(rgb.shape[1]),
        version=np.int32(WINDOW_VERSION),
        seconds=np.float32(seconds),
    )
    log(f"windows {image} seconds={seconds:.1f} grid={cube.shape[0]}x{cube.shape[1]}")
    return cube


def _read_cube(path, shape):
    if not path.is_file():
        return None
    try:
        with np.load(path) as stored:
            return _cube_if_current(stored, shape)
    except (OSError, ValueError, KeyError):
        return None


def _cube_if_current(stored, shape):
    if int(stored["version"]) != WINDOW_VERSION:
        return None
    if int(stored["height"]) != int(shape[0]) or int(stored["width"]) != int(shape[1]):
        return None
    cube = np.array(stored["cube"], dtype=np.float32)
    require(cube.ndim == 4 and cube.shape[-1] == 3, "window cube")
    return cube


def _write_cell(row, features, labels, base) -> None:
    atomic_savez(
        _cell_path(row["id"]),
        X=np.asarray(features, dtype=np.float32),
        y=np.asarray(labels, dtype=np.int8),
        document=np.array(base["document"]),
        fold=np.int8(base["fold"]),
        real=np.int8(base["real"]),
        n_pos_domain=np.int64(base["n_pos"]),
        n_neg_domain=np.int64(base["n_neg"]),
        version=np.int32(CELL_VERSION),
    )


def _read_retrain(query_id: str) -> dict:
    path = RETRAIN_CELLS / f"{safe_id(query_id)}.npz"
    require(path.is_file(), f"missing retrain7 cell {query_id}")
    with np.load(path) as stored:
        return {
            "features": np.array(stored["X"], dtype=np.float32),
            "labels": np.array(stored["y"], dtype=np.int8),
            "document": str(stored["document"].item()),
            "fold": int(stored["fold"]),
            "real": int(stored["real"]),
            "n_pos": int(stored["n_pos_domain"]),
            "n_neg": int(stored["n_neg_domain"]),
        }


def _require_alignment(query_id: str, base: dict, labels, symbol, n_pos: int, n_neg: int) -> None:
    require(np.array_equal(labels, base["labels"]), f"sampling drift {query_id}")
    require(n_pos == base["n_pos"] and n_neg == base["n_neg"], f"domain counts {query_id}")
    require(base["features"].shape[1] == BASE_WIDTH or len(labels) == 0, f"base width {query_id}")
    require(symbol.shape[0] == base["features"].shape[0], f"symbol rows {query_id}")
    require(symbol.shape[1] == CHANNEL_COUNT or len(labels) == 0, f"symbol width {query_id}")


def _summarize(rows: list[dict], slowest: float) -> dict:
    totals = _empty_totals(slowest)
    for row in rows:
        _add_query(row, totals)
    totals["queries"] = len(rows)
    return totals


def _add_query(row, totals) -> None:
    require(_cell_ready(row["id"]), f"missing symbol cell {row['id']}")
    with np.load(_cell_path(row["id"])) as stored:
        labels = np.array(stored["y"], dtype=np.int8)
        totals["cells"] += int(len(labels))
        totals["positive"] += int(np.count_nonzero(labels == 1))
        totals["negative"] += int(np.count_nonzero(labels == 0))
        totals["real_queries"] += int(stored["real"])
        seconds = _window_seconds(row["image"])
    totals["slowest_window_seconds"] = max(totals["slowest_window_seconds"], seconds)


def _window_seconds(image: str) -> float:
    path = _window_path(image)
    if not path.is_file():
        return 0.0
    with np.load(path) as stored:
        if "seconds" not in stored:
            return 0.0
        return float(stored["seconds"])


def _groups(rows: list[dict]) -> dict[str, list]:
    grouped: dict[str, list] = {}
    for index, row in indexed_training_order(rows):
        grouped.setdefault(row["image"], []).append((index, row))
    return grouped


def _by_area(grouped: dict) -> list[str]:
    def area(image: str) -> int:
        row = grouped[image][0][1]
        return int(row["width"]) * int(row["height"])

    return sorted(grouped, key=area)


def _cell_ready(query_id: str) -> bool:
    path = _cell_path(query_id)
    if not path.is_file():
        return False
    try:
        with np.load(path) as stored:
            return _stored_ok(stored)
    except (OSError, ValueError, KeyError):
        return False


def _stored_ok(stored) -> bool:
    if int(stored["version"]) != CELL_VERSION:
        return False
    matrix = stored["X"]
    return matrix.ndim == 2 and int(matrix.shape[1]) == BASE_WIDTH + CHANNEL_COUNT


def _cell_path(query_id: str):
    return CELL_CACHE / f"{safe_id(query_id)}.npz"


def _window_path(image: str):
    name = image.replace("/", "__")
    require(name and ".." not in name, "unsafe image")
    return WINDOW_CACHE / f"{name}.npz"


def _gray(image: str, row: dict):
    spec = {"id": row["id"], "image": image, "width": int(row["width"]), "height": int(row["height"])}
    gray = read_drawing(DATA_ROOT, spec)
    return gray if gray.flags.c_contiguous else np.ascontiguousarray(gray)


def _concat(base, symbol) -> np.ndarray:
    if len(base) == 0:
        return np.zeros((0, base.shape[1] + symbol.shape[1]), np.float32)
    return np.concatenate([base, symbol], axis=1)


def _empty_totals(slowest: float) -> dict:
    return {
        "queries": 0,
        "cells": 0,
        "positive": 0,
        "negative": 0,
        "real_queries": 0,
        "base_width": BASE_WIDTH,
        "symbol_channels": CHANNEL_COUNT,
        "slowest_window_seconds": slowest,
        "fit_queries_expected": len(fit_rows(real_only=False)),
    }
