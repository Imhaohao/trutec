"""Cross-fit the texture HGB on fit, development, and synthetic rows."""
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
    FEATURE_SET,
    MODEL_STRIDE,
    QUERY_SOURCE,
    SAMPLES_PER_CLASS,
    STAGE1_MAX_ITER,
    STAGE1_SEED,
)
from folds import indexed_training_order
from inference import read_drawing
from manifests import (
    combined_rows,
    data_root_for,
    development_ids,
    development_rows,
    fit_ids,
    fit_rows,
    index_of,
    source_of,
    synthetic_ids,
    synthetic_rows,
)
from runtime import (
    ALLDATA12_SAMPLES,
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
SEED_RULE = (
    "STAGE1_SEED + index from indexed_training_order"
    "(fit rows then development rows then synthetic rows)"
)
SPLITS = ("fit", "development", "synthetic")
FIT_SAMPLES = (331422, 163578, 167844)
DEVELOPMENT_SAMPLES = (45064, 19154, 25910)


def ensure_samples(rows: list[dict]) -> dict:
    require(PAIR_WIDTH == 184, f"pair width {PAIR_WIDTH}")
    _prove_synthetic_read(rows)
    owned = fit_ids()
    summarize_samples(fit_rows() + development_rows(), owned)
    log("reused fit and development stage1 samples match v12 counts")
    for image, group in _groups(rows).items():
        _collect_image(image, group, owned)
    summary = summarize_samples(rows, owned)
    log(
        f"stage1 samples={summary['samples']} pos={summary['positive']} "
        f"neg={summary['negative']} synthetic={summary['synthetic']['samples']}"
    )
    return summary


def train_stage1(rows: list[dict], document_fold: dict[str, int]) -> dict:
    _require_sampled_documents(rows, document_fold)
    fold_records = [_train_fold(rows, document_fold, fold) for fold in range(4)]
    final = _train_final(rows, document_fold)
    _require_coverage(document_fold, fold_records, final)
    return {"folds": fold_records, "final": final, "seed_rule": SEED_RULE, "pair_width": PAIR_WIDTH}


def summarize_samples(rows: list[dict], owned: set[str] | None = None) -> dict:
    owned = fit_ids() if owned is None else owned
    dev_ids = development_ids()
    buckets = {name: _empty_bucket() for name in SPLITS}
    origins = {"stacker": 0, "alldata12": 0, "synth13": 0}
    for row in rows:
        labels = _labels(_sample_path(row, owned))
        _tally(buckets[source_of(row)], labels)
        origins[_origin(row, owned)] += 1
    summary = _summary_payload(buckets, origins)
    _require_known_sample_counts(summary)
    return summary


def _prove_synthetic_read(rows: list[dict]) -> None:
    from evaluate import reviewed_domains

    row = next(item for item in rows if item["id"] in synthetic_ids())
    root = data_root_for(row)
    positive, known, _blank, _meta = reviewed_domains(row, root)
    gray = read_drawing(root, row)
    require(positive.any() and known.any(), f"synthetic labels empty {row['id']}")
    require(tuple(gray.shape) == (row["height"], row["width"]), f"synthetic shape {row['id']}")
    del positive, known, gray
    gc.collect()
    log(f"synthetic root ok {row['id']} root={root}")


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
    root = _one_root(pending)
    log(f"stage1 start {image} queries={len(pending)} root={root.name}")
    started = time.perf_counter()
    spec = {key: pending[0][1][key] for key in ("id", "image", "width", "height")}
    gray = read_drawing(root, spec)
    features = hatch_texture.texture_features(gray, stride=MODEL_STRIDE, feature_set=FEATURE_SET)
    try:
        _collect_pending(pending, gray, features, root)
    finally:
        del gray, features
        gc.collect()
    seconds = time.perf_counter() - started
    _append_image_time(image, len(pending), seconds)
    log(f"stage1 image {image} queries={len(pending)} seconds={seconds:.1f}")
    guard_rss()


def _one_root(pending: list):
    roots = {data_root_for(row) for _index, row in pending}
    require(len(roots) == 1, "image has mixed data roots")
    return next(iter(roots))


def _collect_pending(pending, gray, features, root) -> None:
    for index, row in pending:
        _collect_query(index, row, gray, features, root)


def _collect_query(index: int, row: dict, gray, features, root) -> None:
    require(tuple(gray.shape) == (row["height"], row["width"]), f"shape {row['id']}")
    pairs, labels, weights = extract_query_samples(
        row,
        features,
        gray.shape,
        root,
        stride=MODEL_STRIDE,
        samples_per_class=SAMPLES_PER_CLASS,
        query_seed=STAGE1_SEED + index,
        gray=gray,
        query_source=QUERY_SOURCE,
        feature_set=FEATURE_SET,
    )
    _require_pairs(row["id"], pairs, labels, weights)
    require(index >= 357, f"new sample index {index}")
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
    require(len(held & _synthetic_documents()) == 150, f"fold {fold} synthetic holdout")


def _development_documents() -> set[str]:
    return {row["document_id"] for row in development_rows()}


def _synthetic_documents() -> set[str]:
    return {row["document_id"] for row in synthetic_rows()}


def _require_coverage(document_fold, fold_records, final) -> None:
    final_docs = set(final["training_documents"])
    for record in fold_records:
        _require_fold_subset(document_fold, record, final_docs)
    require(_development_documents() <= final_docs, "development document missing from the final stage1 model")
    require(_synthetic_documents() <= final_docs, "synthetic document missing from the final stage1 model")
    require(len(final_docs) == len(document_fold), f"final documents {len(final_docs)}")


def _require_fold_subset(document_fold, record, final_docs) -> None:
    require(set(record["training_documents"]).issubset(final_docs), "fold docs missing from final")
    held = {doc for doc, fold in document_fold.items() if fold == record["held_out_fold"]}
    require(held.isdisjoint(record["training_documents"]), "held-out document in a fold model")


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
        "sources": list(SPLITS),
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
    local = SAMPLE_DIR / f"{row['id']}.npz"
    if local.is_file():
        return local
    shared = _shared_development_sample(row)
    if shared is not None:
        return shared
    return local


def _shared_development_sample(row: dict):
    if row["id"] not in development_ids():
        return None
    shared = ALLDATA12_SAMPLES / f"{row['id']}.npz"
    if shared.is_file():
        return shared
    return None


def _origin(row: dict, owned: set[str]) -> str:
    path = _sample_path(row, owned).resolve()
    if path.is_relative_to(STACKER_SAMPLES.resolve()):
        return "stacker"
    if path.is_relative_to(ALLDATA12_SAMPLES.resolve()):
        return "alldata12"
    return "synth13"


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


def _summary_payload(buckets: dict, origins: dict) -> dict:
    positive = sum(buckets[name]["positive"] for name in SPLITS)
    negative = sum(buckets[name]["negative"] for name in SPLITS)
    payload = {
        "files": sum(buckets[name]["files"] for name in SPLITS),
        "samples": positive + negative,
        "positive": positive,
        "negative": negative,
        "seed_rule": SEED_RULE,
        "index_check": _index_check(),
        "sample_origins": origins,
    }
    for name in SPLITS:
        payload[name] = {**buckets[name], "samples": buckets[name]["positive"] + buckets[name]["negative"]}
    return payload


def _require_known_sample_counts(summary: dict) -> None:
    fit = summary["fit"]
    development = summary["development"]
    require((fit["samples"], fit["positive"], fit["negative"]) == FIT_SAMPLES, f"fit samples {fit}")
    require(
        (development["samples"], development["positive"], development["negative"]) == DEVELOPMENT_SAMPLES,
        f"development samples {development}",
    )
    require(summary["sample_origins"]["stacker"] == 357, "fit samples were not read from stacker")
    require(summary["sample_origins"]["alldata12"] == 55, "development samples were not read from alldata12")
    if summary["synthetic"]["files"] == 0:
        return
    require(summary["synthetic"]["files"] == 842, f"synthetic sample files {summary['synthetic']['files']}")
    require(summary["sample_origins"]["synth13"] == 842, "synthetic samples were not written in synth13")


def _require_sampled_documents(rows: list[dict], document_fold: dict[str, int]) -> None:
    owned = fit_ids()
    found = {row["document_id"] for row in rows if len(_labels(_sample_path(row, owned)))}
    missing = sorted(set(document_fold) - found)
    require(not missing, f"documents with no stage1 samples: {missing[:20]} count={len(missing)}")


def _index_check() -> dict:
    rows = combined_rows()
    indexes = index_of(rows)
    development = [indexes[row["id"]] for row in rows if row["id"] in development_ids()]
    synthetic = [indexes[row["id"]] for row in rows if row["id"] in synthetic_ids()]
    return {
        "fit_queries": len(fit_ids()),
        "development_index_min": min(development),
        "development_index_max": max(development),
        "synthetic_index_min": min(synthetic),
        "synthetic_index_max": max(synthetic),
        "synthetic_queries": len(synthetic),
    }


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
