"""Per-query caches. Fit files hold targets; development files hold reviewed samples for the harness."""
from __future__ import annotations

import limits  # noqa: F401

import numpy as np

from bootstrap import atomic_savez
from limits import DEV_CACHE, FIT_CACHE, VERSION
from scoremap import feature_names


def fit_path(query_id: str):
    return FIT_CACHE / f"{query_id}.npz"


def dev_path(query_id: str):
    return DEV_CACHE / f"{query_id}.npz"


def fit_ready(query_id: str) -> bool:
    return _ready(fit_path(query_id), ("pos_scores", "neg_scores", "target_iou"))


def dev_ready(query_id: str) -> bool:
    return _ready(dev_path(query_id), ("reviewed_scores",))


def save_fit(query_id, features, pos, neg, target_iou, target_recall, document_id, fold) -> None:
    atomic_savez(
        fit_path(query_id),
        features=np.asarray(features, dtype=np.float32),
        pos_scores=np.asarray(pos, dtype=np.float32),
        neg_scores=np.asarray(neg, dtype=np.float32),
        target_iou=np.float64(target_iou),
        target_recall98=np.float64(target_recall),
        document_id=np.array(document_id),
        fold=np.int16(fold),
        version=np.int16(VERSION),
    )


def save_dev(query_id, features, reviewed) -> None:
    atomic_savez(
        dev_path(query_id),
        features=np.asarray(features, dtype=np.float32),
        reviewed_scores=np.asarray(reviewed, dtype=np.float32),
        version=np.int16(VERSION),
    )


def read_fit_header(query_id: str) -> dict:
    with np.load(fit_path(query_id)) as stored:
        _check(stored)
        return {
            "features": np.array(stored["features"], dtype=np.float64),
            "target_iou": float(stored["target_iou"]),
            "target_recall98": float(stored["target_recall98"]),
            "document_id": _text(stored["document_id"]),
            "fold": int(stored["fold"]),
        }


def read_pos_neg(query_id: str):
    with np.load(fit_path(query_id)) as stored:
        _check(stored)
        return np.array(stored["pos_scores"], dtype=np.float32), np.array(stored["neg_scores"], dtype=np.float32)


def read_dev(query_id: str) -> dict:
    with np.load(dev_path(query_id)) as stored:
        _check(stored)
        return {
            "features": np.array(stored["features"], dtype=np.float64),
            "reviewed_scores": np.array(stored["reviewed_scores"], dtype=np.float32),
        }


def _ready(path, keys: tuple[str, ...]) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path) as stored:
            shapes_ok = stored["features"].shape == (len(feature_names()),)
            return int(stored["version"]) == VERSION and shapes_ok and all(key in stored for key in keys)
    except (OSError, ValueError, KeyError):
        return False


def _check(stored) -> None:
    if int(stored["version"]) != VERSION:
        raise RuntimeError("stale qthreshold cache")
    if stored["features"].shape != (len(feature_names()),):
        raise RuntimeError("feature width drifted")


def _text(value) -> str:
    if hasattr(value, "item"):
        return str(value.item())
    return str(value)
