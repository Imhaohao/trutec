"""Score stage 3 on the development split.

Stage-2 grids come from the frozen v6 stage-2 model. Reviewed pixels read the
stage-3 score of their stride-8 cell. The color gate is w16, linear, strength 0.6.
"""
from __future__ import annotations

import gc
import time
from collections import defaultdict

import limits  # noqa: F401

import joblib
import numpy as np

from bootstrap import atomic_save_npy
from color.color_features import (
    apply_gate,
    build_image_color_cache,
    compute_sample_mismatch,
    extract_robust_query_signature,
)
from context_features import feature_names, lookup
from devharness import development_queries, load_rgb, report
from drawing import consume_volumes
from evaluate_dev import assert_development_unseen, load_original
from grids import stage3_score_grid
from limits import DEV_GATED, DEV_RAW, FROZEN_STAGE2, STAGE3_MODEL, guard_memory, log, require

COLOR_WINDOW = 16
COLOR_STRENGTH = 0.6


def score_development() -> dict:
    stage1, metadata = load_original()
    assert_development_unseen(metadata)
    stage2 = joblib.load(FROZEN_STAGE2)["model"]
    stage3, record = load_stage3()
    stamp = STAGE3_MODEL.stat().st_mtime
    started = time.perf_counter()
    for image, queries in _grouped().items():
        _score_image(image, queries, stage1, stage2, stage3, stamp)
        guard_memory()
    log(f"dev images seconds={time.perf_counter() - started:.1f}")
    return {
        "raw": _write_report("autocontext_stage3", DEV_RAW, HERE_REPORT_RAW),
        "gated": _write_report("autocontext_stage3_color_w16_linear_0.6", DEV_GATED, HERE_REPORT_GATED),
        "stage3_fit": record,
    }


HERE_REPORT_RAW = STAGE3_MODEL.parents[1] / "dev_report.json"
HERE_REPORT_GATED = STAGE3_MODEL.parents[1] / "dev_report_color.json"


def load_stage3():
    stored = joblib.load(STAGE3_MODEL)
    record = stored["record"]
    require(list(record["feature_names"]) == feature_names(), "stage3 feature names drifted")
    return stored["model"], record


def _grouped():
    grouped: dict = defaultdict(list)
    for query in development_queries():
        grouped[query.image].append(query)
    return grouped


def _score_image(image, queries, stage1, stage2, stage3, stamp: float) -> None:
    if all(_fresh(query.id, stamp) for query in queries):
        log(f"dev cached {image}")
        return
    started = time.perf_counter()
    grids = _grids(queries, stage1, stage2, stage3)
    rgb = load_rgb(image)
    color_maps = build_image_color_cache(rgb, window_sizes=(COLOR_WINDOW,))[COLOR_WINDOW]
    for query, grid in zip(queries, grids):
        _save_query(query, grid, rgb, color_maps)
    del rgb, color_maps, grids
    gc.collect()
    log(f"dev scored {image} queries={len(queries)} seconds={time.perf_counter() - started:.1f}")


def _grids(queries, stage1, stage2, stage3) -> list:
    captured: list = []

    def consume(index: int, volume) -> None:
        del index
        captured.append(stage3_score_grid(volume, stage2, stage3))

    spec = {
        "id": queries[0].id,
        "image": queries[0].image,
        "width": queries[0].width,
        "height": queries[0].height,
    }
    consume_volumes(
        spec,
        [query.query_box for query in queries],
        [stage1] * len(queries),
        consume,
    )
    require(len(captured) == len(queries), "grid count")
    return captured


def _save_query(query, grid, rgb, color_maps) -> None:
    raw = lookup(grid, query.y, query.x)
    require(len(raw) == len(query.y), f"score length {query.id}")
    require(bool(np.isfinite(raw).all()), f"non-finite {query.id}")
    signature = extract_robust_query_signature(rgb, tuple(int(value) for value in query.query_box))
    mismatch = compute_sample_mismatch(signature, color_maps, query.y, query.x)
    gated = apply_gate(raw, mismatch, gate_type="linear", strength=COLOR_STRENGTH)
    require(bool(np.isfinite(gated).all()), f"non-finite gate {query.id}")
    atomic_save_npy(_raw_path(query.id), np.asarray(raw, dtype=np.float32))
    atomic_save_npy(_gated_path(query.id), np.asarray(gated, dtype=np.float32))


def _write_report(name: str, directory, output):
    scores = {}
    for query in development_queries():
        values = np.load(directory / f"{query.id}.npy")
        require(len(values) == len(query.y), f"score length {query.id}")
        scores[query.id] = values
    result = report(name, scores, output)
    log(name + " " + _selection_line(result))
    return result["selections"]


def _fresh(query_id: str, stamp: float) -> bool:
    return _newer(_raw_path(query_id), stamp) and _newer(_gated_path(query_id), stamp)


def _newer(path, stamp: float) -> bool:
    return path.is_file() and path.stat().st_mtime > stamp


def _raw_path(query_id: str):
    return DEV_RAW / f"{query_id}.npy"


def _gated_path(query_id: str):
    return DEV_GATED / f"{query_id}.npy"


def _selection_line(result: dict) -> str:
    best = result["selections"]["best_macro_iou"]
    return f"best_macro_iou={best['document_macro_iou']:.4f}" if best else "best_macro_iou=none"
