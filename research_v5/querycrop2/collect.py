"""Fit and development score collection. One drawing in memory at a time."""
from __future__ import annotations

import gc
import json
import time
from collections import defaultdict

import bootstrap  # noqa: F401

import joblib
import numpy as np

from bootstrap import (
    DEV_CACHE,
    FIT_CACHE,
    FIT_MANIFEST,
    FIT_REAL_MANIFEST,
    FOLDS_PATH,
    FROZEN_STAGE2,
    IMAGE_LOG,
    STAGE1_FOLD_DIR,
    STAGE2_FOLD_DIR,
    guard_memory,
    log,
)
from constants import DATA_ROOT
import devharness
from evaluate import reviewed_domains
from evaluate_dev import assert_development_unseen, load_original
from folds import load_document_folds
from guards import CACHE_VERSION, VARIANTS
from volume import prepare, score_jobs


def collect_fit() -> list[dict]:
    folds = load_document_folds(FOLDS_PATH)
    jobs = [_fit_job(row) for row in real_fit_rows()]
    for fold, members in _by_fold(jobs, folds).items():
        payload, stage2 = _load_fold(fold)
        log(f"fit fold {fold} drawings={len({job['image'] for job in members})}")
        _score_members(members, payload["model"], stage2, _fit_coordinates, payload)
        del payload, stage2
        gc.collect()
        guard_memory()
    missing = [job["id"] for job in jobs if not cache_ready(job["cache_path"])]
    if missing:
        raise RuntimeError(f"fit cache missing {len(missing)} queries")
    log(f"fit cache complete queries={len(jobs)}")
    return jobs


def collect_dev() -> list[dict]:
    stage1, metadata = load_original()
    assert_development_unseen(metadata)
    stage2 = joblib.load(FROZEN_STAGE2)["model"]
    jobs = [_dev_job(query) for query in devharness.development_queries()]
    _score_members(jobs, stage1, stage2, _dev_coordinates, None)
    del stage1, stage2
    gc.collect()
    guard_memory()
    missing = [job["id"] for job in jobs if not cache_ready(job["cache_path"])]
    if missing:
        raise RuntimeError(f"dev cache missing {len(missing)} queries")
    log(f"dev cache complete queries={len(jobs)}")
    return jobs


def real_fit_rows() -> list[dict]:
    fit_ids = {row["id"] for row in _rows(FIT_MANIFEST)}
    return [row for row in _rows(FIT_REAL_MANIFEST) if row["id"] in fit_ids]


def cache_ready(path) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path) as stored:
            if int(stored["version"]) != CACHE_VERSION:
                return False
            length = int(stored["positive"].shape[0])
            return all(stored[name].shape == (length,) for name in VARIANTS)
    except (OSError, KeyError, ValueError):
        return False


def _score_members(jobs, stage1, stage2, coordinates, stage1_payload) -> None:
    for image, group in _by_image(jobs).items():
        pending = [job for job in group if not cache_ready(job["cache_path"])]
        if not pending:
            log(f"cached {image}")
            continue
        _require_one_document(pending, stage1_payload)
        log(f"start {image} queries={len(pending)}")
        started = time.perf_counter()
        spec = _spec(pending[0])
        context, descriptors = prepare(spec, [job["query_box"] for job in pending])
        try:
            score_jobs(context, pending, descriptors, stage1, stage2, coordinates)
        finally:
            del context, descriptors
            gc.collect()
        seconds = time.perf_counter() - started
        _append_log({"image": image, "queries": len(pending), "seconds": seconds})
        log(f"scored {image} queries={len(pending)} seconds={seconds:.1f}")
        guard_memory()


def _fit_coordinates(job: dict):
    positive, known, _blank, _meta = reviewed_domains(job["row"], DATA_ROOT)
    ys, xs = np.nonzero(known)
    labels = np.ascontiguousarray(positive[ys, xs])
    del positive, known
    return ys.astype(np.int32), xs.astype(np.int32), labels.astype(bool)


def _dev_coordinates(job: dict):
    query = job["query"]
    return query.y, query.x, np.asarray(query.positive).astype(bool)


def _fit_job(row: dict) -> dict:
    return _job(row["id"], row["document_id"], row["image"], row["width"], row["height"], row["query_box"], FIT_CACHE, row=row)


def _dev_job(query) -> dict:
    return _job(
        query.id,
        query.document_id,
        query.image,
        query.width,
        query.height,
        query.query_box,
        DEV_CACHE,
        query=query,
    )


def _job(query_id, document_id, image, width, height, box, cache, row=None, query=None) -> dict:
    return {
        "id": query_id,
        "document_id": document_id,
        "image": image,
        "width": int(width),
        "height": int(height),
        "query_box": tuple(int(value) for value in box),
        "cache_path": cache / f"{query_id}.npz",
        "row": row,
        "query": query,
    }


def _load_fold(fold: int):
    stage1_payload = joblib.load(STAGE1_FOLD_DIR / f"fold_{fold}.joblib")
    stage2_payload = joblib.load(STAGE2_FOLD_DIR / f"fold_{fold}.joblib")
    if int(stage1_payload["held_out_fold"]) != fold or int(stage2_payload["held_out_fold"]) != fold:
        raise RuntimeError(f"fold file does not match fold {fold}")
    return stage1_payload, stage2_payload["model"]


def _require_one_document(jobs, stage1_payload) -> None:
    documents = {job["document_id"] for job in jobs}
    if len(documents) != 1:
        raise RuntimeError("one drawing spans documents")
    if stage1_payload is None:
        return
    document = documents.pop()
    if document in set(stage1_payload["training_documents"]):
        raise RuntimeError(f"stage-1 fold saw document {document}")


def _by_fold(jobs, folds) -> dict[int, list]:
    grouped = defaultdict(list)
    for job in jobs:
        document = job["document_id"]
        if document not in folds:
            raise RuntimeError(f"document missing from folds.json: {document}")
        grouped[int(folds[document])].append(job)
    return dict(sorted(grouped.items()))


def _by_image(jobs) -> dict[str, list]:
    grouped = defaultdict(list)
    for job in jobs:
        grouped[job["image"]].append(job)
    return dict(grouped)


def _spec(job: dict) -> dict:
    return {"id": job["id"], "image": job["image"], "width": job["width"], "height": job["height"]}


def _rows(path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["examples"] if isinstance(payload, dict) else payload


def _append_log(payload: dict) -> None:
    IMAGE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with IMAGE_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")
