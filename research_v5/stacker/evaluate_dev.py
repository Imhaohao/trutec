"""Score the frozen second stage on the development split."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import bootstrap  # noqa: F401

import numpy as np

from bootstrap import atomic_save_npy
from constants import GRID_STRIDE, MODEL_PATH, MODEL_STRIDE, STACKER, log, require
from devharness import HGB_CACHE, cached_scores, development_queries, report
from drawing import consume_volumes
from inference_texture import verify_model_metadata
from model import scores_at_coordinates
from texture_learning import load_texture_model


def load_original():
    model, metadata = load_texture_model(MODEL_PATH)
    verify_model_metadata(metadata)
    require(metadata.get("query_descriptor_source") == "crop", "query source")
    require(int(metadata.get("stride", -1)) == MODEL_STRIDE, "model stride")
    return model, metadata


def assert_development_unseen(metadata: dict) -> None:
    trained = set(metadata.get("training_documents", []))
    for query in development_queries():
        require(query.document_id not in trained, f"development document in texture model: {query.document_id}")


def score_development(stage2, model_stamp: float) -> dict[str, np.ndarray]:
    stage1, _metadata = load_original()
    grouped: dict[str, list] = defaultdict(list)
    for query in development_queries():
        grouped[query.image].append(query)
    scores: dict[str, np.ndarray] = {}
    for image, queries in grouped.items():
        _score_or_load(image, queries, stage1, stage2, model_stamp, scores)
    _require_complete(scores)
    return scores


def per_query_recall(scores: dict[str, np.ndarray], threshold: float) -> list[dict]:
    rows = [_one_query(query, scores[query.id], threshold) for query in development_queries()]
    rows.sort(key=_recall_sort_key)
    return rows


def write_report(name: str, scores: dict[str, np.ndarray], path: Path) -> dict:
    return report(name, scores, path)


def baseline_report() -> dict:
    return report("original_crop_hgb", cached_scores(HGB_CACHE), STACKER / "baseline_hgb.json")


def _score_or_load(image, queries, stage1, stage2, model_stamp, scores) -> None:
    if all(_fresh(query.id, model_stamp) for query in queries):
        for query in queries:
            scores[query.id] = np.load(_score_path(query.id))
        log(f"dev cached {image}")
        return
    spec = {
        "id": queries[0].id,
        "image": image,
        "width": queries[0].width,
        "height": queries[0].height,
    }
    coords = [(query.id, query.y, query.x) for query in queries]

    def consume(index: int, volume, coords=coords) -> None:
        ident, y, x = coords[index]
        scores[ident] = scores_at_coordinates(stage2, volume, y, x)

    consume_volumes(spec, [query.query_box for query in queries], [stage1] * len(queries), consume)
    for query in queries:
        _save_score(query.id, scores[query.id], len(query.y))
    log(f"dev scored {image} queries={len(queries)}")


def _one_query(query, values: np.ndarray, threshold: float) -> dict:
    decision = np.asarray(values) >= threshold
    positive = np.asarray(query.positive, dtype=bool)
    require(decision.shape == positive.shape, f"score shape {query.id}")
    true_positive = int(np.count_nonzero(decision & positive))
    false_negative = int(np.count_nonzero(~decision & positive))
    denominator = true_positive + false_negative
    recall = None if denominator == 0 else true_positive / denominator
    return {
        "id": query.id,
        "document_id": query.document_id,
        "recall": recall,
        "positives": int(np.count_nonzero(positive)),
        "true_positive": true_positive,
        "false_negative": false_negative,
    }


def _save_score(query_id: str, values: np.ndarray, expected: int) -> None:
    require(len(values) == expected, f"score length {query_id}")
    atomic_save_npy(_score_path(query_id), np.asarray(values, dtype=np.float32))


def _fresh(query_id: str, model_stamp: float) -> bool:
    path = _score_path(query_id)
    return path.is_file() and path.stat().st_mtime >= model_stamp


def _score_path(query_id: str) -> Path:
    return STACKER / "cache" / "dev_scores" / f"{query_id}.npy"


def _require_complete(scores: dict[str, np.ndarray]) -> None:
    expected = {query.id for query in development_queries()}
    require(set(scores) == expected, "development score keys")


def _recall_sort_key(row: dict) -> tuple:
    if row["recall"] is None:
        return (1, 0.0)
    return (0, float(row["recall"]))


def dump_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
