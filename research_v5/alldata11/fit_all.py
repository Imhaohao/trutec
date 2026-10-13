"""Fit hardneg_0.75 on every fit cell plus every development cell. No cross-validation."""
from __future__ import annotations

import gc
import hashlib
import json
import os

import runtime  # noqa: F401

import joblib
import numpy as np

from runtime import (
    CELL_DIR,
    FIT_CELL_DIR,
    MODEL_PATH,
    STRUCTURE,
    guard_rss,
    log,
    require,
)
from structure_features import FEATURE_VERSION
from train import Table, fit_setting, recipe_params
from weights import HARD_WEIGHT, SETTINGS, hard_negative_fraction
from cells import joined_names


def fit_all() -> dict:
    setting = _hardneg_setting()
    names = joined_names()
    table, counts = _load_table(names)
    guard_rss()
    if _matches_saved(counts, setting):
        del table
        gc.collect()
        log("model cached")
        return json.loads(MODEL_PATH.with_suffix(".json").read_text(encoding="utf-8"))
    model, seconds = fit_setting(table, setting)
    record = _record(table, counts, setting, seconds)
    del table
    gc.collect()
    _dump(model, record)
    del model
    gc.collect()
    saved = _with_hash(record)
    log(f"model rows={saved['rows']} sha256={saved['sha256']} seconds={saved['seconds']:.1f}")
    return saved


def _hardneg_setting() -> dict:
    found = [item for item in SETTINGS if item["name"] == "hardneg_0.75"]
    require(len(found) == 1, "hardneg_0.75 setting")
    require(found[0]["cutoff"] == 0.75, "hardneg cutoff")
    require(HARD_WEIGHT == 3.0, "hard weight")
    recipe = recipe_params()
    require(recipe["name"] == "recipe", "recipe name")
    return found[0]


def _load_table(names: list[str]) -> tuple[Table, dict]:
    codes: dict[str, int] = {}
    blocks, labels, docs = [], [], []
    counts = {
        "fit": _empty_bucket(),
        "development": _empty_bucket(),
    }
    _consume_dir(FIT_CELL_DIR, "fit", names, codes, blocks, labels, docs, counts)
    _consume_dir(CELL_DIR, "development", names, codes, blocks, labels, docs, counts)
    _check_counts(counts)
    table = Table(
        x=np.concatenate(blocks, axis=0),
        y=np.concatenate(labels, axis=0),
        doc=np.concatenate(docs, axis=0),
        fold=np.zeros(sum(len(item) for item in labels), dtype=np.int8),
        pop=np.ones(sum(len(item) for item in labels), dtype=np.float64),
    )
    require(int(np.count_nonzero(table.y == 1)) == counts["fit"]["positive"] + counts["development"]["positive"], "pos")
    require(int(np.count_nonzero(table.y == 0)) == counts["fit"]["negative"] + counts["development"]["negative"], "neg")
    log(f"table rows={len(table.y)} features={table.x.shape[1]} documents={len(codes)}")
    return table, counts


def _consume_dir(directory, split: str, names, codes, blocks, labels, docs, counts) -> None:
    files = sorted(directory.glob("*.npz"))
    require(files, f"no cells in {split}")
    for path in files:
        features, target, document = _read_cell(path, names)
        _tally(counts[split], target, document)
        if len(target) == 0:
            continue
        code = codes.setdefault(document, len(codes))
        blocks.append(features)
        labels.append(target)
        docs.append(np.full(len(target), code, dtype=np.int32))


def _read_cell(path, names):
    with np.load(path) as stored:
        require(int(stored["version"]) == FEATURE_VERSION, f"stale cache {path.name}")
        features = np.array(stored["X"], dtype=np.float32)
        target = np.array(stored["y"], dtype=np.int8)
        document = str(stored["document"].item())
    require(features.shape[1] == len(names) or len(target) == 0, f"feature width {path.name}")
    return features, target, document


def _tally(bucket: dict, target, document: str) -> None:
    bucket["queries"] += 1
    bucket["cells"] += int(len(target))
    bucket["positive"] += int(np.count_nonzero(target == 1))
    bucket["negative"] += int(np.count_nonzero(target == 0))
    bucket["documents"].add(document)


def _empty_bucket() -> dict:
    return {"queries": 0, "cells": 0, "positive": 0, "negative": 0, "documents": set()}


def _check_counts(counts: dict) -> None:
    fit = counts["fit"]
    development = counts["development"]
    expected = json.loads((STRUCTURE / "cache" / "cell_summary.json").read_text(encoding="utf-8"))
    require(fit["queries"] == int(expected["queries"]), f"fit queries {fit['queries']}")
    require(fit["cells"] == int(expected["cells"]), f"fit cells {fit['cells']}")
    require(fit["positive"] == int(expected["positive"]), f"fit positive {fit['positive']}")
    require(fit["negative"] == int(expected["negative"]), f"fit negative {fit['negative']}")
    require(development["queries"] == 55, f"development queries {development['queries']}")
    require(fit["documents"].isdisjoint(development["documents"]), "fit and development documents overlap")
    log(
        f"fit cells={fit['cells']} pos={fit['positive']} neg={fit['negative']} "
        f"dev cells={development['cells']} pos={development['positive']} neg={development['negative']}"
    )


def _record(table: Table, counts: dict, setting: dict, seconds: float) -> dict:
    positive = counts["fit"]["positive"] + counts["development"]["positive"]
    negative = counts["fit"]["negative"] + counts["development"]["negative"]
    return {
        "signature": _signature(setting),
        "seconds": seconds,
        "rows": int(len(table.y)),
        "positive": positive,
        "negative": negative,
        "fit_rows": counts["fit"]["cells"],
        "fit_positive": counts["fit"]["positive"],
        "fit_negative": counts["fit"]["negative"],
        "development_rows": counts["development"]["cells"],
        "development_positive": counts["development"]["positive"],
        "development_negative": counts["development"]["negative"],
        "fit_documents": len(counts["fit"]["documents"]),
        "development_documents": len(counts["development"]["documents"]),
        "hard_negative_fraction": hard_negative_fraction(table.y, table.x[:, 0], setting["cutoff"]),
        "feature_names": joined_names(),
    }


def _signature(setting: dict) -> dict:
    return {
        "feature_version": FEATURE_VERSION,
        "setting": {"name": setting["name"], "cutoff": setting["cutoff"]},
        "hard_weight": HARD_WEIGHT,
        "recipe": recipe_params(),
        "rows": "all cached fit cells plus all development cells",
    }


def _dump(model, record: dict) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = MODEL_PATH.with_name(f".{MODEL_PATH.name}.{os.getpid()}.partial")
    joblib.dump({"model": model, "record": record}, temporary)
    os.replace(temporary, MODEL_PATH)


def _with_hash(record: dict) -> dict:
    stored = joblib.load(MODEL_PATH)
    require(stored["record"]["feature_names"] == joined_names(), "saved feature names")
    require(int(stored["model"].n_features_in_) == len(joined_names()), "saved width")
    payload = dict(record)
    payload["sha256"] = _sha256(MODEL_PATH)
    MODEL_PATH.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _matches_saved(counts: dict, setting: dict) -> bool:
    path = MODEL_PATH.with_suffix(".json")
    if not MODEL_PATH.is_file() or not path.is_file():
        return False
    saved = json.loads(path.read_text(encoding="utf-8"))
    positive = counts["fit"]["positive"] + counts["development"]["positive"]
    negative = counts["fit"]["negative"] + counts["development"]["negative"]
    return (
        saved.get("signature") == _signature(setting)
        and saved.get("rows") == counts["fit"]["cells"] + counts["development"]["cells"]
        and saved.get("positive") == positive
        and saved.get("negative") == negative
        and saved.get("sha256") == _sha256(MODEL_PATH)
    )


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
