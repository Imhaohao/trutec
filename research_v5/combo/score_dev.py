"""Score the development split with one drawing in memory. Labels stay in this file."""
from __future__ import annotations

import bootstrap  # noqa: F401

import argparse
import json
import time
from collections import defaultdict

import numpy as np

from bootstrap import (
    DEV_CACHE,
    STRUCTURE_GATED,
    STRUCTURE_MODEL,
    atomic_savez,
    current_gb,
    guard_memory,
    log,
    peak_gb,
)
from devharness import development_queries, report
from evaluate_dev import assert_development_unseen, load_original
from inference_v7 import load_stage2
from pipeline import cleaner_by_name, map_request, prepare_drawing, release_drawing

CACHE_VERSION = 1
HERE_RUNTIME = DEV_CACHE.parent / "score_runtime.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cleaner", required=True)
    parser.add_argument("--selection-note", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    stage1, metadata = load_original()
    assert_development_unseen(metadata)
    stage2 = load_stage2(STRUCTURE_MODEL)
    names = _names(args.cleaner)
    gap = _score_all(names, stage1, stage2)
    reports = {name: _write_report(name) for name in names}
    payload = {
        "cleaner": args.cleaner,
        "selection_note": args.selection_note,
        "also_scored": names,
        "seconds": time.perf_counter() - started,
        "peak_gb": peak_gb(),
        "current_gb": current_gb(),
        "identity_structure_max_abs_gap": gap,
        "queries": len(development_queries()),
        "reports": {name: str(_report_path(name)) for name in reports},
    }
    HERE_RUNTIME.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log(f"score done seconds={payload['seconds']:.1f} peak_gb={payload['peak_gb']:.2f} structure_gap={gap}")


def _names(cleaner: str) -> list[str]:
    if cleaner == "identity":
        return ["identity"]
    return [cleaner, "identity"]


def _score_all(names: list[str], stage1, stage2) -> float:
    worst = 0.0
    grouped: dict[str, list] = defaultdict(list)
    for query in development_queries():
        grouped[query.image].append(query)
    for image, queries in _by_area(grouped):
        gap = _score_image(image, queries, names, stage1, stage2)
        worst = max(worst, gap)
        guard_memory()
    return worst


def _by_area(grouped: dict) -> list:
    return sorted(grouped.items(), key=lambda item: int(item[1][0].width) * int(item[1][0].height))


def _score_image(image, queries, names, stage1, stage2) -> float:
    pending = [name for name in names if any(_read(name, query.id) is None for query in queries)]
    if not pending:
        log(f"dev cached {image}")
        return _cached_gap(queries)
    started = time.perf_counter()
    requests = [_request(query) for query in queries]
    context = prepare_drawing(requests, _data_root())
    worst = 0.0
    try:
        worst = _score_pending(context, queries, requests, pending, stage1, stage2)
    finally:
        release_drawing(context)
    log(
        f"dev scored {image} queries={len(queries)} cleaners={','.join(pending)} "
        f"seconds={time.perf_counter() - started:.1f} rss_gb={current_gb():.2f}"
    )
    return max(worst, _cached_gap(queries))


def _score_pending(context, queries, requests, pending, stage1, stage2) -> float:
    worst = 0.0
    cleaners = {name: cleaner_by_name(name) for name in pending}
    for index, (query, request) in enumerate(zip(queries, requests)):
        worst = max(worst, _score_query(context, index, query, request, cleaners, stage1, stage2))
    return worst


def _score_query(context, index, query, request, cleaners, stage1, stage2) -> float:
    worst = 0.0
    for name, cleaner in cleaners.items():
        if _read(name, query.id) is not None:
            continue
        gated = map_request(context, index, request, stage1, stage2, cleaner)
        sampled = np.asarray(gated[query.y, query.x], dtype=np.float32)
        del gated
        _check_sample(query, sampled)
        _write(name, query.id, sampled)
        if name == "identity":
            gap = _structure_gap(query.id, sampled)
            if gap != 0.0:
                raise RuntimeError(f"{query.id} identity/structure gap {gap}")
            worst = max(worst, gap)
    return worst


def _cached_gap(queries) -> float:
    worst = 0.0
    for query in queries:
        cached = _read("identity", query.id)
        if cached is not None:
            worst = max(worst, _structure_gap(query.id, cached))
    return worst


def _structure_gap(query_id: str, scores: np.ndarray) -> float:
    reference = np.load(STRUCTURE_GATED / f"{query_id}.npy")
    if reference.shape != scores.shape:
        raise RuntimeError(f"{query_id} structure cache length {reference.shape} != {scores.shape}")
    return float(np.max(np.abs(reference.astype(np.float64) - scores.astype(np.float64))))


def _write_report(name: str) -> dict:
    scores = {}
    for query in development_queries():
        values = _read(name, query.id)
        if values is None or len(values) != len(query.y):
            raise RuntimeError(f"missing scores {name} {query.id}")
        scores[query.id] = values
    result = report(f"combo_{name}_structure_hardneg_0.75_color_w16_linear_0.6", scores, _report_path(name))
    best = result["selections"]["best_macro_iou"]
    log(f"{name} best_macro_iou={best['document_macro_iou']:.4f}")
    return result


def _request(query) -> dict:
    return {
        "id": query.id,
        "image": query.image,
        "width": query.width,
        "height": query.height,
        "query_box": list(query.query_box),
        "context_boxes": [list(box) for box in query.context_boxes],
    }


def _check_sample(query, sampled: np.ndarray) -> None:
    if sampled.shape != query.y.shape or not np.isfinite(sampled).all():
        raise RuntimeError(f"bad sample {query.id}")


def _write(name: str, query_id: str, values: np.ndarray) -> None:
    atomic_savez(_cache_path(name, query_id), scores=values, version=np.int32(CACHE_VERSION))


def _read(name: str, query_id: str):
    path = _cache_path(name, query_id)
    if not path.is_file():
        return None
    with np.load(path) as stored:
        if int(stored["version"]) != CACHE_VERSION:
            return None
        return np.array(stored["scores"], dtype=np.float32)


def _cache_path(name: str, query_id: str):
    return DEV_CACHE / name / f"{query_id}.npz"


def _report_path(name: str):
    from bootstrap import HERE

    return HERE / f"dev_report_{name}.json"


def _data_root():
    import devharness

    return devharness.DATA_ROOT


if __name__ == "__main__":
    main()
