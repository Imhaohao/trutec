"""Score structure stage-2 variants on the development split.

Stage 1 is the original crop model. Each reviewed pixel reads the stage-2
probability of its stride-8 cell, then the fixed w16 linear color gate at
strength 0.6. That is the same lookup the v6 development numbers use.
"""
from __future__ import annotations

import gc
import time
from collections import defaultdict

import limits  # noqa: F401

import cv2
import numpy as np

from banks import ensure_bank, load_rgb
from cells import joined_names
from color.color_features import (
    apply_gate,
    build_image_color_cache,
    compute_sample_mismatch,
    extract_robust_query_signature,
)
from devharness import development_queries, report
from drawing import consume_volumes
from evaluate_dev import assert_development_unseen, load_original
from limits import DEV_GATED, DEV_RAW, atomic_save_npy, current_gb, log, require, write_json
from model import scores_at_coordinates
from structure_features import component_maps, query_reference, structure_volume
from train import SETTINGS, load_fitted, model_path
from report_text import best_at_recall

COLOR_WINDOW = 16
COLOR_STRENGTH = 0.6


def score_development() -> dict:
    stage1, metadata = load_original()
    assert_development_unseen(metadata)
    models = {setting["name"]: load_fitted(setting["name"]) for setting in SETTINGS}
    stamps = {name: model_path(name).stat().st_mtime for name in models}
    started = time.perf_counter()
    for image, queries in _grouped().items():
        _score_image(image, queries, stage1, models, stamps)
        guard_after_image()
    log(f"dev images seconds={time.perf_counter() - started:.1f}")
    return {name: _write_variant(name) for name in models}


def guard_after_image() -> None:
    from limits import guard_memory

    guard_memory()


def _grouped():
    grouped = defaultdict(list)
    for query in development_queries():
        grouped[query.image].append(query)
    return grouped


def _score_image(image, queries, stage1, models, stamps) -> None:
    if _image_fresh(queries, stamps):
        log(f"dev cached {image}")
        return
    started = time.perf_counter()
    rgb = load_rgb(image)
    bank = ensure_bank(image, rgb)
    references = _references(rgb, queries)
    del rgb
    gc.collect()
    _consume(image, queries, stage1, models, stamps, bank, references)
    del bank, references
    gc.collect()
    _gate_image(image, queries, stamps)
    log(f"dev scored {image} queries={len(queries)} seconds={time.perf_counter() - started:.1f} rss_gb={current_gb():.2f}")


def _references(rgb, queries):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    area_map, side_map, ink = component_maps(rgb)
    return [query_reference(gray, query.query_box, area_map, side_map, ink) for query in queries]


def _consume(image, queries, stage1, models, stamps, bank, references) -> None:
    spec = {
        "id": queries[0].id,
        "image": image,
        "width": queries[0].width,
        "height": queries[0].height,
    }

    def consume(index: int, volume) -> None:
        _predict_one(queries[index], volume, bank, references[index], models, stamps)

    consume_volumes(spec, [query.query_box for query in queries], [stage1] * len(queries), consume)


def _predict_one(query, volume, bank, reference, models, stamps) -> None:
    needed = [name for name, stamp in stamps.items() if not _newer(_raw_path(name, query.id), stamp)]
    if not needed:
        return
    augmented = np.concatenate([volume, structure_volume(bank, reference)], axis=-1)
    require(augmented.shape[-1] == len(joined_names()), "joined width")
    for name in needed:
        raw = scores_at_coordinates(models[name], augmented, query.y, query.x)
        _save_raw(name, query, raw)


def _gate_image(image, queries, stamps) -> None:
    if all(_newer(_gated_path(name, query.id), stamp) for name, stamp in stamps.items() for query in queries):
        return
    rgb = load_rgb(image)
    maps = build_image_color_cache(rgb, window_sizes=(COLOR_WINDOW,))[COLOR_WINDOW]
    for query in queries:
        _gate_query(query, rgb, maps, stamps)
    del rgb, maps


def _gate_query(query, rgb, maps, stamps) -> None:
    signature = extract_robust_query_signature(rgb, tuple(int(value) for value in query.query_box))
    mismatch = compute_sample_mismatch(signature, maps, query.y, query.x)
    for name, stamp in stamps.items():
        _gate_one(name, query, mismatch, stamp)


def _gate_one(name, query, mismatch, stamp: float) -> None:
    if _newer(_gated_path(name, query.id), stamp):
        return
    raw = np.load(_raw_path(name, query.id))
    require(len(raw) == len(query.y), f"raw length {query.id}")
    gated = apply_gate(raw, mismatch, gate_type="linear", strength=COLOR_STRENGTH)
    require(bool(np.isfinite(gated).all()), f"non-finite gate {query.id}")
    atomic_save_npy(_gated_path(name, query.id), np.asarray(gated, dtype=np.float32))


def _write_variant(name: str) -> dict:
    scores = {}
    for query in development_queries():
        values = np.load(_gated_path(name, query.id))
        require(len(values) == len(query.y), f"gated length {query.id}")
        scores[query.id] = values
    result = report(f"structure_{name}_color_w16_linear_0.6", scores, _report_path(name))
    result["recall_ge_0_95"] = best_at_recall(result["points"], 0.95)
    write_json(_report_path(name), result)
    best = result["selections"]["best_macro_iou"]
    log(f"{name} best_macro_iou={best['document_macro_iou']:.4f}")
    return result


def _save_raw(name, query, raw) -> None:
    require(len(raw) == len(query.y), f"score length {query.id}")
    require(bool(np.isfinite(raw).all()), f"non-finite {query.id}")
    atomic_save_npy(_raw_path(name, query.id), np.asarray(raw, dtype=np.float32))


def _image_fresh(queries, stamps) -> bool:
    return all(
        _newer(_raw_path(name, query.id), stamp) and _newer(_gated_path(name, query.id), stamp)
        for name, stamp in stamps.items()
        for query in queries
    )


def _newer(path, stamp: float) -> bool:
    return path.is_file() and path.stat().st_mtime > stamp


def _raw_path(name: str, query_id: str):
    return DEV_RAW / name / f"{query_id}.npy"


def _gated_path(name: str, query_id: str):
    return DEV_GATED / name / f"{query_id}.npy"


def _report_path(name: str):
    from limits import HERE

    return HERE / f"dev_{name}.json"
