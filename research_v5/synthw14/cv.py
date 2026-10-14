"""Grouped 4-fold CV of the v13 stage 2 with synthetic document weight scaled by w.

Train on folds != k (real plus synthetic, synthetic weights times w), score held-out
real documents in fold k. Selection rule fixed before running: largest real-document
macro IoU at threshold 0.25; ties within 0.002 go to the smaller w. Public validation
is never read.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "synth13"))

import runtime  # noqa: E402,F401

import gc  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from cells import joined_names  # noqa: E402
from dataset import _population_weights  # noqa: E402
from model import document_balanced_weights, positive_probability  # noqa: E402
from runtime import CELL_DEV, CELL_FIT, CELL_SYN, log  # noqa: E402
from structure_features import FEATURE_VERSION  # noqa: E402
from train import make_classifier  # noqa: E402
from weights import apply_hard_negative  # noqa: E402

SYNTHETIC_WEIGHTS = (0.0, 0.1, 0.3, 1.0)
THRESHOLD = 0.25
CUTOFF = 0.75
FOLDS = 4
OUT = HERE / "cv_results.json"


def load() -> dict:
    names = joined_names()
    parts = {"x": [], "y": [], "doc": [], "fold": [], "pop": [], "syn": []}
    codes: dict[str, int] = {}
    for directory, synthetic in ((CELL_FIT, False), (CELL_DEV, False), (CELL_SYN, True)):
        for path in sorted(directory.glob("*.npz")):
            _append(path, names, synthetic, codes, parts)
    table = {key: np.concatenate(value) for key, value in parts.items()}
    log(f"cv table rows={len(table['y'])} documents={len(codes)} synthetic_rows={int(table['syn'].sum())}")
    return table


def _append(path, names, synthetic, codes, parts) -> None:
    with np.load(path) as stored:
        assert int(stored["version"]) == FEATURE_VERSION, path.name
        labels = np.array(stored["y"], dtype=np.int8)
        if len(labels) == 0:
            return
        features = np.array(stored["X"], dtype=np.float32)
        document = str(stored["document"].item())
        fold = int(stored["fold"])
        n_pos, n_neg = int(stored["n_pos_domain"]), int(stored["n_neg_domain"])
    assert features.shape[1] == len(names), path.name
    code = codes.setdefault(document, len(codes))
    parts["x"].append(features)
    parts["y"].append(labels)
    parts["doc"].append(np.full(len(labels), code, dtype=np.int32))
    parts["fold"].append(np.full(len(labels), fold, dtype=np.int8))
    parts["pop"].append(_population_weights(labels, n_pos, n_neg))
    parts["syn"].append(np.full(len(labels), synthetic, dtype=bool))


def training_weights(table: dict, synthetic_weight: float) -> np.ndarray:
    base = document_balanced_weights(table["doc"], table["y"])
    weights = apply_hard_negative(base, table["y"], table["x"][:, 0], table["doc"], CUTOFF)
    return np.where(table["syn"], weights * synthetic_weight, weights)


def held_out_counts(table: dict, test: np.ndarray, scores: np.ndarray) -> dict[int, list[float]]:
    predicted = scores >= THRESHOLD
    y, pop, doc = table["y"][test], table["pop"][test], table["doc"][test]
    counts: dict[int, list[float]] = {}
    for code in np.unique(doc):
        mask = doc == code
        p, t, w = predicted[mask], y[mask] == 1, pop[mask]
        counts[int(code)] = [float(w[p & t].sum()), float(w[p & ~t].sum()), float(w[~p & t].sum()), float(w[~p & ~t].sum())]
    return counts


def summarize(counts: dict[int, list[float]]) -> dict:
    ious = [tp / (tp + fp + fn) for tp, fp, fn, _tn in counts.values() if tp + fn > 0]
    tp, fp, fn, tn = (sum(item[i] for item in counts.values()) for i in range(4))
    return {
        "documents": len(ious),
        "macro_iou": float(np.mean(ious)),
        "precision": tp / (tp + fp),
        "recall": tp / (tp + fn),
        "fpr": fp / (fp + tn),
    }


def run_weight(table: dict, synthetic_weight: float) -> dict:
    weights = training_weights(table, synthetic_weight)
    counts: dict[int, list[float]] = {}
    for fold in range(FOLDS):
        train = (table["fold"] != fold) & (weights > 0)
        test = (table["fold"] == fold) & ~table["syn"]
        started = time.perf_counter()
        model = make_classifier()
        model.fit(table["x"][train], table["y"][train], sample_weight=weights[train])
        counts.update(held_out_counts(table, test, positive_probability(model, table["x"][test])))
        del model
        gc.collect()
        log(f"w={synthetic_weight} fold={fold} seconds={time.perf_counter() - started:.1f}")
    result = {"synthetic_weight": synthetic_weight, **summarize(counts)}
    log(json.dumps(result))
    return result


def choose(results: list[dict]) -> dict:
    best = max(item["macro_iou"] for item in results)
    eligible = [item for item in results if item["macro_iou"] >= best - 0.002]
    return min(eligible, key=lambda item: item["synthetic_weight"])


def main() -> None:
    table = load()
    results = []
    for synthetic_weight in SYNTHETIC_WEIGHTS:
        results.append(run_weight(table, synthetic_weight))
        OUT.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
    chosen = choose(results)
    OUT.write_text(json.dumps({"results": results, "chosen": chosen, "rule": __doc__}, indent=2), encoding="utf-8")
    log(f"chosen {json.dumps(chosen)}")


if __name__ == "__main__":
    main()
