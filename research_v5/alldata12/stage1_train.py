"""Cross-fit the texture HGB on fit plus development, using the stacker recipe."""
from __future__ import annotations

import gc
import json
import os
import time

import runtime  # noqa: F401

import cv2
import joblib
import numpy as np

import hatch_texture
from bootstrap import atomic_savez
from constants import (
    DATA_ROOT,
    FEATURE_SET,
    MODEL_STRIDE,
    QUERY_SOURCE,
    SAMPLES_PER_CLASS,
    STAGE1_MAX_ITER,
    STAGE1_SEED,
)
from folds import indexed_training_order
from inference import read_drawing
from manifests import fit_ids, index_of
from runtime import (
    FOLD_DIR,
    SAMPLE_DIR,
    STACKER_SAMPLES,
    STAGE1_PATH,
    guard_rss,
    inside,
    log,
    require,
    sha256_file,
    write_json,
)
from texture_learning import pair_feature_names
from train_texture import extract_query_samples, train_classifier

cv2.setNumThreads(4)

PAIR_WIDTH = len(pair_feature_names(hatch_texture.texture_feature_names(FEATURE_SET), QUERY_SOURCE))
SEED_RULE = "STAGE1_SEED + index from indexed_training_order(fit rows then development rows)"


def ensure_samples(rows: list[dict]) -> dict:
    require(PAIR_WIDTH == 184, f"pair width {PAIR_WIDTH}")
    owned = fit_ids()
    grouped = _groups(rows)
    for image, group in grouped.items():
        _collect_image(image, group, owned)
    summary = summarize_samples(rows, owned)
    log(
        f"stage1 samples={summary['samples']} pos={summary['positive']} "
        f"neg={summary['negative']} fit={summary['fit']['samples']} dev={summary['development']['samples']}"
    )
    return summary


def train_stage1(rows: list[dict], document_fold: dict[str, int]) -> dict:
    fold_records = [_train_fold(rows, document_fold, fold) for fold in range(4)]
    final = _train_final(rows, document_fold)
    _require_coverage(document_fold, fold_records, final)
    return {"folds": fold_records, "final": final, "seed_rule": SEED_RULE, "pair_width": PAIR_WIDTH}


def summarize_samples(rows: list[dict], owned: set[str] | None = None) -> dict:
    owned = fit_ids() if owned is None else owned
    buckets = {"fit": _empty_bucket(), "development": _empty_bucket()}
    for row in rows:
        labels = _labels(_sample_path(row, owned))
        split = "fit" if row["id"] in owned else "development"
        _tally(buckets[split], labels)
    return _summary_payload(buckets)


def _groups(rows: list[dict]) -> dict[str, list]:
    grouped: dict[str, list] = {}
    for index, row in indexed_training_order(rows):
        grouped.setdefault(row["image"], []).append((index, row))
    return grouped


def _collect_image(image: str, group: list, owned: set[str]) -> None:
    pending = [(index, row) for index, row in group if not _sample_path(row, owned).is_file()]
    if not pending:
        log(f"stage1 cached {image}")
        return
    require(all(row["id"] not in owned for _index, row in pending), f"missing fit sample {image}")
    log(f"stage1 start {image} queries={len(pending)}")
    started = time.perf_counter()
    spec = {key: pending[0][1][key] for key in ("id", "image", "width", "height")}
    gray = read_drawing(DATA_ROOT, spec)
    features = hatch_texture.texture_features(gray, stride=MODEL_STRIDE, feature_set=FEATURE_SET)
    try:
        _collect_pending(pending, gray, features)
    finally:
        del gray, features
        gc.collect()
    seconds = time.perf_counter() - started
    _append_image_time(image, len(pending), seconds)
    log(f"stage1 image {image} queries={len(pending)} seconds={seconds:.1f}")
    guard_rss()


def _collect_pending(pending, gray, features) -> None:
    for index, row in pending:
        _collect_query(index, row, gray, features)


def _collect_query(index: int, row: dict, gray, features) -> None:
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
    _require_pairs(row["id"], pairs, labels, weights)
    require(index >= 357, f"development sample index {index}")
    _save_sample(row["id"], pairs, labels, weights)
    log(f"stage1 save {row['id']} index={index} seed={STAGE1_SEED + index} rows={len(labels)}")


def _require_pairs(query_id: str, pairs, labels, weights) -> None:
    require(len(labels) == len(weights), f"weights {query_id}")
    if len(labels) == 0:
        require(pairs.shape[1] == PAIR_WIDTH, f"empty width {query_id}")
        return
    require(pairs.shape == (len(labels), PAIR_WIDTH), f"pair shape {query_id}")


def _save_sample(query_id: str, pairs, labels, weights) -> None:
    path = SAMPLE_DIR / f"{query_id}.npz"
    _require_safe_id(query_id)
    inside(path)
    atomic_savez(path, pairs=pairs, y=labels, w=weights)


def _train_fold(rows, document_fold, fold: int) -> dict:
    path = FOLD_DIR / f"fold_{fold}.joblib"
    signature = _signature("fold", fold)
    if _fresh(path, signature):
        log(f"stage1 fold {fold} cached")
        return json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    started = time.perf_counter()
    pairs, labels, weights, documents = _load_matrix(rows, document_fold, fold)
    _require_held_out(document_fold, fold, documents)
    model = _fit_estimator(pairs, labels, weights)
    record = _record(signature, fold, documents, labels, time.perf_counter() - started, model)
    del pairs, labels, weights
    gc.collect()
    _store(path, model, fold, documents, record)
    del model
    gc.collect()
    saved = _with_hash(path, record)
    log(f"stage1 fold {fold} samples={saved['samples']} seconds={saved['seconds']:.1f}")
    return saved


def _train_final(rows, document_fold) -> dict:
    signature = _signature("final", None)
    if _fresh(STAGE1_PATH, signature):
        log("stage1 final cached")
        return json.loads(STAGE1_PATH.with_suffix(".json").read_text(encoding="utf-8"))
    started = time.perf_counter()
    pairs, labels, weights, documents = _load_matrix(rows, document_fold, None)
    model = _fit_estimator(pairs, labels, weights)
    record = _record(signature, None, documents, labels, time.perf_counter() - started, model)
    del pairs, labels, weights
    gc.collect()
    _store(STAGE1_PATH, model, None, documents, record)
    del model
    gc.collect()
    saved = _with_hash(STAGE1_PATH, record)
    log(f"stage1 final samples={saved['samples']} seconds={saved['seconds']:.1f}")
    return saved


def _fit_estimator(pairs, labels, weights):
    model = train_classifier(pairs, labels, weights, max_iter=STAGE1_MAX_ITER, seed=STAGE1_SEED)
    _require_estimator(model)
    require(int(model.n_features_in_) == PAIR_WIDTH, "stage1 width")
    require(list(map(int, model.classes_)) == [0, 1], "stage1 classes")
    return model


def _require_estimator(model) -> None:
    params = model.get_params()
    require(params["learning_rate"] == 0.08, "learning rate")
    require(params["max_leaf_nodes"] == 31, "leaves")
    require(params["l2_regularization"] == 1.0, "l2")
    require(params["max_iter"] == STAGE1_MAX_ITER, "max iter")
    require(params["early_stopping"] is False, "early stopping")
    require(params["random_state"] == STAGE1_SEED, "seed")
    require(params["min_samples_leaf"] == 20, "min samples leaf")


def _load_matrix(rows, document_fold, held_out):
    pairs, labels, weights, documents = [], [], [], set()
    owned = fit_ids()
    for _index, row in indexed_training_order(rows):
        _append_row(row, document_fold, held_out, owned, pairs, labels, weights, documents)
    require(pairs, f"fold {held_out} has no training rows")
    return (
        np.concatenate(pairs, axis=0),
        np.concatenate(labels, axis=0),
        np.concatenate(weights, axis=0),
        documents,
    )


def _append_row(row, document_fold, held_out, owned, pairs, labels, weights, documents) -> None:
    document = row["document_id"]
    if held_out is not None and document_fold[document] == held_out:
        return
    path = _sample_path(row, owned)
    require(path.is_file(), f"missing stage1 sample {row['id']}")
    with np.load(path) as stored:
        target = np.array(stored["y"])
        if len(target) == 0:
            return
        block = np.array(stored["pairs"])
        weight = np.array(stored["w"])
    require(block.shape == (len(target), PAIR_WIDTH), f"pair shape {row['id']}")
    require(len(weight) == len(target), f"weights {row['id']}")
    pairs.append(block)
    labels.append(target)
    weights.append(weight)
    documents.add(document)


def _require_held_out(document_fold, fold: int, documents: set) -> None:
    held = {doc for doc, assigned in document_fold.items() if assigned == fold}
    require(held.isdisjoint(documents), f"fold {fold} leaked a held-out document")
    require(len(held & _development_documents()) == 3, f"fold {fold} development holdout")


def _development_documents() -> set[str]:
    from manifests import development_rows

    return {row["document_id"] for row in development_rows()}


def _require_coverage(document_fold, fold_records, final) -> None:
    final_docs = set(final["training_documents"])
    for record in fold_records:
        require(set(record["training_documents"]).issubset(final_docs), "fold docs missing from final")
        held = {doc for doc, fold in document_fold.items() if fold == record["held_out_fold"]}
        require(held.isdisjoint(record["training_documents"]), "held-out document in a fold model")
    require(_development_documents() <= final_docs, "development document missing from the final stage1 model")
    require(len(final_docs) == len(document_fold), f"final documents {len(final_docs)}")


def _signature(kind: str, fold) -> dict:
    return {
        "kind": kind,
        "held_out_fold": fold,
        "query_source": QUERY_SOURCE,
        "feature_set": FEATURE_SET,
        "stride": MODEL_STRIDE,
        "samples_per_class": SAMPLES_PER_CLASS,
        "seed": STAGE1_SEED,
        "seed_rule": SEED_RULE,
        "max_iter": STAGE1_MAX_ITER,
        "pair_width": PAIR_WIDTH,
        "learning_rate": 0.08,
        "max_leaf_nodes": 31,
        "l2_regularization": 1.0,
        "early_stopping": False,
        "min_samples_leaf": 20,
    }


def _record(signature, fold, documents, labels, seconds, model) -> dict:
    names = pair_feature_names(hatch_texture.texture_feature_names(FEATURE_SET), QUERY_SOURCE)
    return {
        "signature": signature,
        "held_out_fold": fold,
        "training_documents": sorted(documents),
        "training_document_count": len(documents),
        "samples": int(len(labels)),
        "positive": int(np.count_nonzero(labels == 1)),
        "negative": int(np.count_nonzero(labels == 0)),
        "seconds": seconds,
        "feature_count": int(model.n_features_in_),
        "feature_names": names,
    }


def _store(path, model, fold, documents, record) -> None:
    inside(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    inside(temporary)
    joblib.dump(
        {
            "model": model,
            "held_out_fold": fold,
            "training_documents": sorted(documents),
            "record": record,
        },
        temporary,
    )
    os.replace(temporary, path)


def _with_hash(path, record) -> dict:
    payload = dict(record)
    payload["sha256"] = sha256_file(path)
    payload["path"] = str(path)
    write_json(path.with_suffix(".json"), payload)
    return payload


def _fresh(path, signature) -> bool:
    sidecar = path.with_suffix(".json")
    if not path.is_file() or not sidecar.is_file():
        return False
    saved = json.loads(sidecar.read_text(encoding="utf-8"))
    return saved.get("signature") == signature and saved.get("sha256") == sha256_file(path)


def _sample_path(row: dict, owned: set[str]):
    if row["id"] in owned:
        return STACKER_SAMPLES / f"{row['id']}.npz"
    _require_safe_id(row["id"])
    return SAMPLE_DIR / f"{row['id']}.npz"


def _labels(path):
    require(path.is_file(), f"missing stage1 sample {path.name}")
    with np.load(path) as stored:
        return np.array(stored["y"])


def _tally(bucket: dict, labels) -> None:
    bucket["files"] += 1
    bucket["positive"] += int(np.count_nonzero(labels == 1))
    bucket["negative"] += int(np.count_nonzero(labels == 0))


def _empty_bucket() -> dict:
    return {"files": 0, "positive": 0, "negative": 0}


def _summary_payload(buckets: dict) -> dict:
    positive = buckets["fit"]["positive"] + buckets["development"]["positive"]
    negative = buckets["fit"]["negative"] + buckets["development"]["negative"]
    return {
        "files": buckets["fit"]["files"] + buckets["development"]["files"],
        "samples": positive + negative,
        "positive": positive,
        "negative": negative,
        "fit": {**buckets["fit"], "samples": buckets["fit"]["positive"] + buckets["fit"]["negative"]},
        "development": {
            **buckets["development"],
            "samples": buckets["development"]["positive"] + buckets["development"]["negative"],
        },
        "seed_rule": SEED_RULE,
        "index_check": _index_check(),
    }


def _index_check() -> dict:
    from manifests import combined_rows

    rows = combined_rows()
    indexes = index_of(rows)
    owned = fit_ids()
    development = [indexes[row["id"]] for row in rows if row["id"] not in owned]
    return {"fit_queries": len(owned), "development_index_min": min(development), "development_index_max": max(development)}


def _append_image_time(image: str, queries: int, seconds: float) -> None:
    path = SAMPLE_DIR / "images.jsonl"
    inside(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"image": image, "queries": queries, "seconds": seconds})
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _require_safe_id(query_id: str) -> None:
    require(bool(query_id) and "/" not in query_id and "\\" not in query_id, "unsafe id")
    require(not query_id.startswith("."), "unsafe id")
