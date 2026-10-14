"""Score the symbol stage-2 model with the frozen v7 pipeline plus symbol channels.

The wrapper calls research_v5/combo/pipeline.py. It does not edit that file.
Labels stay in this module: the pipeline receives only the request fields.
"""
from __future__ import annotations

import time
from collections import defaultdict

import guard

guard.activate()

import numpy as np

from constants import DATA_ROOT
from devharness import development_queries, report
from evaluate_dev import assert_development_unseen, load_original
from guard import (
    CLEANER,
    DEV_CACHE,
    DEV_REPORT,
    SCORE_VERSION,
    TARGET_DOCUMENT,
    V7_DEV_CACHE,
    V7_REPORT,
    atomic_savez,
    current_gb,
    guard_memory,
    log,
    read_json,
    require,
    safe_id,
    write_json,
)
from inference_texture import predict_probability_grid
from inference_v6 import gated_native, native_cell_lookup
from inference_v7 import load_stage2
from join_cells import cached_window_cube
from model import positive_probability
from operating import recall_point
from pipeline import _assemble, _ensure_structure, _structure_channels, cleaner_by_name, prepare_drawing, release_drawing
from symbol_features import CHANNEL_COUNT, channels_from_cube, query_stats

guard.activate()


def score(model_path) -> dict:
    started = time.perf_counter()
    stage1, metadata = load_original()
    assert_development_unseen(metadata)
    stage2 = load_stage2(model_path)
    _require_width(stage2)
    cleaner = cleaner_by_name(CLEANER)
    stamp = int(model_path.stat().st_mtime_ns)
    _score_all(stage1, stage2, cleaner, stamp)
    result = _write_report(stamp, time.perf_counter() - started)
    best = result["selections"]["best_macro_iou"]["document_macro_iou"]
    log(f"dev best_macro={best:.4f}")
    return result


def request_from(query) -> dict:
    return {
        "id": query.id,
        "image": query.image,
        "width": int(query.width),
        "height": int(query.height),
        "query_box": [int(value) for value in query.query_box],
        "context_boxes": [[int(value) for value in box] for box in query.context_boxes],
    }


def _score_all(stage1, stage2, cleaner, stamp: int) -> None:
    grouped: dict[str, list] = defaultdict(list)
    for query in development_queries():
        grouped[query.image].append(query)
    for image, queries in _by_area(grouped):
        _score_image(image, queries, stage1, stage2, cleaner, stamp)
        guard_memory()


def _score_image(image, queries, stage1, stage2, cleaner, stamp: int) -> None:
    pending = [query for query in queries if _read(query.id, stamp) is None]
    if not pending:
        log(f"dev cached {image}")
        return
    started = time.perf_counter()
    requests = [request_from(query) for query in pending]
    context = prepare_drawing(requests, DATA_ROOT)
    try:
        cube = cached_window_cube(image, context.rgb)
        _score_pending(context, pending, requests, stage1, stage2, cleaner, cube, stamp)
    finally:
        release_drawing(context)
    log(f"dev scored {image} queries={len(pending)} seconds={time.perf_counter() - started:.1f} rss_gb={current_gb():.2f}")


def _score_pending(context, pending, requests, stage1, stage2, cleaner, cube, stamp: int) -> None:
    for index, query in enumerate(pending):
        gated = _gated(context, index, requests[index], stage1, stage2, cleaner, cube)
        sampled = np.asarray(gated[query.y, query.x], dtype=np.float32)
        del gated
        _check_sample(query, sampled)
        _write(query.id, sampled, stamp)


def _gated(context, index, request, stage1, stage2, cleaner, cube):
    probability = predict_probability_grid(stage1, context.features, cleaner(context.gray, request["query_box"]))
    volume = _assemble(context, probability, request["query_box"], index)
    del probability
    model_input = _with_symbols(stage2, volume, context, index, cube, request)
    del volume
    grid = _probability_grid(stage2, model_input)
    del model_input
    native = native_cell_lookup(grid, int(request["height"]), int(request["width"]))
    del grid
    return gated_native(native, context.rgb, context.color_maps, request["query_box"])


def _with_symbols(stage2, volume, context, index, cube, request):
    _ensure_structure(context)
    struct = _structure_channels(context, index)
    joined = np.concatenate([volume, struct], axis=-1)
    del struct
    symbol = channels_from_cube(cube, query_stats(context.gray, request["query_box"]))
    require(symbol.shape[:2] == joined.shape[:2], "symbol grid")
    full = np.concatenate([joined, symbol], axis=-1)
    del joined, symbol
    require(full.shape[-1] == int(stage2.n_features_in_), f"stage2 width {full.shape[-1]}")
    return full


def _probability_grid(stage2, volume: np.ndarray) -> np.ndarray:
    height, width, channels = volume.shape
    flat = positive_probability(stage2, volume.reshape(-1, channels))
    return flat.reshape(height, width)


def _write_report(stamp: int, seconds: float) -> dict:
    scores = {query.id: _require_scores(query, stamp) for query in development_queries()}
    name = f"symbols7_{CLEANER}_symbols_hardneg_0.75_color_w16_linear_0.6"
    result = report(name, scores, DEV_REPORT)
    result["recall_ge_0_95"] = recall_point(result["points"], 0.95)
    result["recall_ge_0_98"] = recall_point(result["points"], 0.98)
    result["document_fp"] = _document_block(scores, result["recall_ge_0_95"])
    result["model_stamp"] = stamp
    result["seconds"] = seconds
    result["cleaner"] = CLEANER
    result["channels"] = CHANNEL_COUNT
    write_json(DEV_REPORT, result)
    return result


def _document_block(scores, ours_point) -> dict:
    v7 = read_json(V7_REPORT)
    require(v7 is not None and ours_point is not None, "missing operating point")
    v7_point = recall_point(v7["points"], 0.95)
    require(v7_point is not None, "v7 recall point")
    v7_scores = _v7_scores()
    ours_threshold = float(ours_point["threshold"])
    v7_threshold = float(v7_point["threshold"])
    return {
        "document_id": TARGET_DOCUMENT,
        "ours_threshold": ours_threshold,
        "ours_fp": _document_fp(scores, ours_threshold),
        "v7_threshold": v7_threshold,
        "v7_fp": _document_fp(v7_scores, v7_threshold),
        "ours_fp_at_v7_threshold": _document_fp(scores, v7_threshold),
    }


def _document_fp(scores, threshold: float) -> int:
    total = 0
    for query in development_queries():
        if query.document_id != TARGET_DOCUMENT:
            continue
        total += _query_fp(scores[query.id], query.positive, threshold)
    return total


def _query_fp(scores, positive, threshold: float) -> int:
    chosen = np.asarray(scores) >= threshold
    negative = ~np.asarray(positive).astype(bool)
    return int(np.count_nonzero(chosen & negative))


def _v7_scores() -> dict:
    scores = {}
    for query in development_queries():
        values = _read_v7(query.id)
        require(values is not None and len(values) == len(query.y), f"missing v7 scores {query.id}")
        scores[query.id] = values
    return scores


def _read_v7(query_id: str):
    path = V7_DEV_CACHE / f"{safe_id(query_id)}.npz"
    if not path.is_file():
        return None
    with np.load(path) as stored:
        return np.array(stored["scores"], dtype=np.float32)


def _require_width(stage2) -> None:
    from features import feature_names as stacker_names
    from structure_features import feature_names as structure_names

    expected = len(stacker_names()) + len(structure_names()) + CHANNEL_COUNT
    require(int(stage2.n_features_in_) == expected, f"model width {stage2.n_features_in_}")


def _require_scores(query, stamp: int) -> np.ndarray:
    values = _read(query.id, stamp)
    require(values is not None and len(values) == len(query.y), f"missing dev scores {query.id}")
    return values


def _by_area(grouped: dict) -> list:
    return sorted(grouped.items(), key=lambda item: int(item[1][0].width) * int(item[1][0].height))


def _check_sample(query, sampled: np.ndarray) -> None:
    if sampled.shape != query.y.shape or not np.isfinite(sampled).all():
        raise RuntimeError(f"bad sample {query.id}")


def _write(query_id: str, values: np.ndarray, stamp: int) -> None:
    atomic_savez(_cache_path(query_id), scores=values, stamp=np.int64(stamp), version=np.int32(SCORE_VERSION))


def _read(query_id: str, stamp: int):
    path = _cache_path(query_id)
    if not path.is_file():
        return None
    try:
        with np.load(path) as stored:
            return _cached_scores(stored, stamp)
    except (OSError, ValueError, KeyError):
        return None


def _cached_scores(stored, stamp: int):
    if int(stored["version"]) != SCORE_VERSION or int(stored["stamp"]) != stamp:
        return None
    return np.array(stored["scores"], dtype=np.float32)


def _cache_path(query_id: str):
    return DEV_CACHE / f"{safe_id(query_id)}.npz"
