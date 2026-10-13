"""Score the selected stage-2 model with the frozen v7 pipeline.

The wrapper calls research_v5/combo/pipeline.py. It does not edit that file.
Labels stay in this module: the pipeline receives only the request fields.
"""
from __future__ import annotations

import time
from collections import defaultdict

import runtime

runtime.activate()

import numpy as np

from devharness import DATA_ROOT, development_queries, report
from evaluate_dev import assert_development_unseen, load_original
from inference_v7 import load_stage2
from operating import recall_point
from pipeline import cleaner_by_name, map_request, prepare_drawing, release_drawing
from runtime import CLEANER, DEV_CACHE, DEV_REPORT, current_gb, guard_memory, log, require, write_json

runtime.hold()

SCORE_VERSION = 1


def score(model_path, variant: str) -> dict:
    started = time.perf_counter()
    stage1, metadata = load_original()
    assert_development_unseen(metadata)
    stage2 = load_stage2(model_path)
    cleaner = cleaner_by_name(CLEANER)
    runtime.hold()
    stamp = int(model_path.stat().st_mtime_ns)
    _score_all(stage1, stage2, cleaner, stamp)
    result = _write_report(variant, stamp, time.perf_counter() - started)
    log(f"dev best_macro={result['selections']['best_macro_iou']['document_macro_iou']:.4f}")
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
        _score_pending(context, pending, requests, stage1, stage2, cleaner, stamp)
    finally:
        release_drawing(context)
    log(f"dev scored {image} queries={len(pending)} seconds={time.perf_counter() - started:.1f} rss_gb={current_gb():.2f}")


def _score_pending(context, pending, requests, stage1, stage2, cleaner, stamp: int) -> None:
    for index, query in enumerate(pending):
        gated = map_request(context, index, requests[index], stage1, stage2, cleaner)
        sampled = np.asarray(gated[query.y, query.x], dtype=np.float32)
        del gated
        _check_sample(query, sampled)
        _write(query.id, sampled, stamp)


def _write_report(variant: str, stamp: int, seconds: float) -> dict:
    scores = {query.id: _require_scores(query, stamp) for query in development_queries()}
    name = f"retrain7_{CLEANER}_{variant}_color_w16_linear_0.6"
    result = report(name, scores, DEV_REPORT)
    result["recall_ge_0_95"] = recall_point(result["points"], 0.95)
    result["recall_ge_0_98"] = recall_point(result["points"], 0.98)
    result["variant"] = variant
    result["cleaner"] = CLEANER
    result["model_stamp"] = stamp
    result["seconds"] = seconds
    result["rule"] = "max development macro IoU with development recall >= 0.95; ties by fewer FP"
    write_json(DEV_REPORT, result)
    return result


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
    from runtime import atomic_savez

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
    require(bool(query_id) and "/" not in query_id and "\\" not in query_id, "unsafe id")
    return DEV_CACHE / f"{query_id}.npz"
