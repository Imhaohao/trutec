"""Cross-fit the texture HGB with train_texture.py's sampler and estimator."""
from __future__ import annotations

import json
import time
from pathlib import Path

import bootstrap  # noqa: F401

import joblib
import numpy as np

import hatch_texture
from bootstrap import atomic_savez
from constants import (
    CACHE,
    DATA_ROOT,
    EXPECTED_STAGE1_NEGATIVE,
    EXPECTED_STAGE1_POSITIVE,
    EXPECTED_STAGE1_SAMPLES,
    FEATURE_SET,
    MODEL_STRIDE,
    QUERY_SOURCE,
    SAMPLES_PER_CLASS,
    STAGE1_MAX_ITER,
    STAGE1_SEED,
    log,
    require,
)
from folds import indexed_training_order
from inference import read_drawing
from train_texture import extract_query_samples, train_classifier


def sample_path(query_id: str) -> Path:
    _safe_id(query_id)
    return CACHE / "stage1" / f"{query_id}.npz"


def fold_model_path(fold: int) -> Path:
    return CACHE / "folds" / f"fold_{fold}.joblib"


def collect_stage1_samples(rows: list[dict]) -> dict:
    indexed = indexed_training_order(rows)
    grouped: dict[str, list[tuple[int, dict]]] = {}
    for index, row in indexed:
        grouped.setdefault(row["image"], []).append((index, row))
    for image, group in grouped.items():
        _collect_image(image, group)
    summary = summarize_stage1(rows)
    summary["seconds"] = _sum_image_seconds()
    log(
        f"stage1 samples={summary['samples']} pos={summary['positive']} "
        f"neg={summary['negative']} seconds={summary['seconds']:.1f}"
    )
    _require_original_coverage(summary)
    return summary


def summarize_stage1(rows: list[dict]) -> dict:
    positive = 0
    negative = 0
    files = 0
    for row in rows:
        path = sample_path(row["id"])
        require(path.is_file(), f"missing stage1 sample {row['id']}")
        with np.load(path) as stored:
            labels = np.array(stored["y"])
        positive += int(np.count_nonzero(labels == 1))
        negative += int(np.count_nonzero(labels == 0))
        files += 1
    return {
        "files": files,
        "samples": positive + negative,
        "positive": positive,
        "negative": negative,
    }


def train_fold_models(rows: list[dict], document_fold: dict[str, int]) -> list[dict]:
    records = []
    for fold in sorted(set(document_fold.values())):
        records.append(_train_one_fold(rows, document_fold, fold))
    return records


def load_fold_payloads() -> dict[int, dict]:
    payloads = {}
    for path in sorted((CACHE / "folds").glob("fold_*.joblib")):
        payload = joblib.load(path)
        payloads[int(payload["held_out_fold"])] = payload
    require(payloads, "no fold models")
    return payloads


def _collect_image(image: str, group: list[tuple[int, dict]]) -> None:
    if all(sample_path(row["id"]).is_file() for _, row in group):
        log(f"stage1 cached {image}")
        return
    log(f"stage1 start {image} queries={len(group)}")
    started = time.perf_counter()
    spec = _slim(group[0][1])
    gray = read_drawing(DATA_ROOT, spec)
    features = hatch_texture.texture_features(gray, stride=MODEL_STRIDE, feature_set=FEATURE_SET)
    for index, row in group:
        _collect_query(index, row, gray, features)
    del gray, features
    seconds = time.perf_counter() - started
    _append_image_time(image, len(group), seconds)
    log(f"stage1 image {image} queries={len(group)} seconds={seconds:.1f}")


def _collect_query(index: int, row: dict, gray, features) -> None:
    path = sample_path(row["id"])
    if path.is_file():
        return
    require(tuple(gray.shape) == (row["height"], row["width"]), f"shape {row['id']}")
    pairs, labels, weights = extract_query_samples(
        row,
        features,
        gray.shape,
        DATA_ROOT,
        stride=MODEL_STRIDE,
        samples_per_class=SAMPLES_PER_CLASS,
        query_seed=STAGE1_SEED + index,
        gray=gray,
        query_source=QUERY_SOURCE,
        feature_set=FEATURE_SET,
    )
    _save_sample(path, pairs, labels, weights)


def _save_sample(path: Path, pairs, labels, weights) -> None:
    atomic_savez(path, pairs=pairs, y=labels, w=weights)


def _train_one_fold(rows: list[dict], document_fold: dict[str, int], fold: int) -> dict:
    path = fold_model_path(fold)
    if path.is_file():
        payload = joblib.load(path)
        log(f"stage1 fold {fold} cached")
        return payload["record"]
    started = time.perf_counter()
    pairs, labels, weights, documents = _load_fold_matrix(rows, document_fold, fold)
    _require_held_out(document_fold, fold, documents)
    model = train_classifier(pairs, labels, weights, max_iter=STAGE1_MAX_ITER, seed=STAGE1_SEED)
    seconds = time.perf_counter() - started
    record = {
        "held_out_fold": fold,
        "training_documents": len(documents),
        "samples": int(len(labels)),
        "positive": int(np.count_nonzero(labels == 1)),
        "negative": int(np.count_nonzero(labels == 0)),
        "seconds": seconds,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "held_out_fold": fold,
            "training_documents": sorted(documents),
            "record": record,
        },
        path,
    )
    del pairs, labels, weights
    log(f"stage1 fold {fold} samples={record['samples']} seconds={seconds:.1f}")
    return record


def _load_fold_matrix(rows, document_fold, fold):
    pairs, labels, weights, documents = [], [], [], set()
    for _index, row in indexed_training_order(rows):
        _append_training_row(row, document_fold, fold, pairs, labels, weights, documents)
    require(pairs, f"fold {fold} has no training rows")
    return (
        np.concatenate(pairs, axis=0),
        np.concatenate(labels, axis=0),
        np.concatenate(weights, axis=0),
        documents,
    )


def _append_training_row(row, document_fold, fold, pairs, labels, weights, documents) -> None:
    document = row["document_id"]
    if document_fold[document] == fold:
        return
    with np.load(sample_path(row["id"])) as stored:
        target = np.array(stored["y"])
        if len(target) == 0:
            return
        pairs.append(np.array(stored["pairs"]))
        labels.append(target)
        weights.append(np.array(stored["w"]))
    documents.add(document)


def _require_held_out(document_fold, fold, documents) -> None:
    held = {doc for doc, assigned in document_fold.items() if assigned == fold}
    require(held.isdisjoint(documents), f"fold {fold} leaked a held-out document")


def _require_original_coverage(summary: dict) -> None:
    require(summary["samples"] == EXPECTED_STAGE1_SAMPLES, f"sample count {summary['samples']}")
    require(summary["positive"] == EXPECTED_STAGE1_POSITIVE, f"positive count {summary['positive']}")
    require(summary["negative"] == EXPECTED_STAGE1_NEGATIVE, f"negative count {summary['negative']}")


def _append_image_time(image: str, queries: int, seconds: float) -> None:
    path = CACHE / "stage1" / "images.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"image": image, "queries": queries, "seconds": seconds})
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _sum_image_seconds() -> float:
    path = CACHE / "stage1" / "images.jsonl"
    if not path.is_file():
        return 0.0
    latest: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        latest[row["image"]] = float(row["seconds"])
    return float(sum(latest.values()))


def _slim(row: dict) -> dict:
    return {key: row[key] for key in ("id", "image", "width", "height")}


def _safe_id(query_id: str) -> None:
    require(bool(query_id) and "/" not in query_id and "\\" not in query_id, "unsafe id")
    require(not query_id.startswith("."), "unsafe id")
