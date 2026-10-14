"""Screen a larger stage-1 HGB on v12's stage-1 samples, judged on held-out real documents.

For each of v12's 4 document folds, compare v12's own fold model (recipe: 200 rounds,
31 leaves) with larger fold models trained on exactly the same rows, on held-out real
(doc-) document samples: weighted ROC AUC and average precision. This is a screen only;
adoption still requires the full stage-2 CV rule on real documents. Public validation is
never read.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "alldata12"))

import runtime  # noqa: E402,F401

import json  # noqa: E402
import time  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from folds import indexed_training_order  # noqa: E402
from manifests import combined_rows, fit_ids  # noqa: E402
from runtime import FOLD_DIR, FOLDS_PATH, log  # noqa: E402
from stage1_train import _load_matrix, _sample_path  # noqa: E402

CONFIGS = (
    {"name": "iter600_leaf63", "max_iter": 600, "max_leaf_nodes": 63, "learning_rate": 0.08},
    {"name": "iter800_leaf127_lr05", "max_iter": 800, "max_leaf_nodes": 127, "learning_rate": 0.05},
)
OUT = HERE / "screen.json"


def held_out_real(rows, document_fold, fold: int):
    owned = fit_ids()
    pairs, labels, weights = [], [], []
    for _index, row in indexed_training_order(rows):
        if document_fold[row["document_id"]] != fold or not row["document_id"].startswith("doc-"):
            continue
        with np.load(_sample_path(row, owned)) as stored:
            if len(stored["y"]) == 0:
                continue
            pairs.append(np.array(stored["pairs"]))
            labels.append(np.array(stored["y"]))
            weights.append(np.array(stored["w"]))
    return np.concatenate(pairs), np.concatenate(labels), np.concatenate(weights)


def classifier(config: dict) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=config["learning_rate"],
        max_leaf_nodes=config["max_leaf_nodes"],
        l2_regularization=1.0,
        max_iter=config["max_iter"],
        min_samples_leaf=20,
        early_stopping=False,
        random_state=42,
    )


def scores(model, x, y, w) -> dict:
    probability = model.predict_proba(x)[:, 1]
    return {"auc": float(roc_auc_score(y, probability, sample_weight=w)), "ap": float(average_precision_score(y, probability, sample_weight=w))}


def main() -> None:
    rows = combined_rows()
    document_fold = {key: int(value) for key, value in json.loads(FOLDS_PATH.read_text(encoding="utf-8"))["document_fold"].items()}
    results = []
    for fold in range(4):
        x_test, y_test, w_test = held_out_real(rows, document_fold, fold)
        recipe = joblib.load(FOLD_DIR / f"fold_{fold}.joblib")["model"]
        entry = {"fold": fold, "test_rows": int(len(y_test)), "recipe": scores(recipe, x_test, y_test, w_test)}
        x_train, y_train, w_train, _documents = _load_matrix(rows, document_fold, fold)
        for config in CONFIGS:
            started = time.perf_counter()
            model = classifier(config).fit(x_train, y_train, sample_weight=w_train)
            entry[config["name"]] = {**scores(model, x_test, y_test, w_test), "seconds": time.perf_counter() - started}
        results.append(entry)
        log(json.dumps(entry))
        OUT.write_text(json.dumps({"folds": results}, indent=2), encoding="utf-8")
    names = ["recipe"] + [config["name"] for config in CONFIGS]
    mean = {name: {metric: float(np.mean([entry[name][metric] for entry in results])) for metric in ("auc", "ap")} for name in names}
    OUT.write_text(json.dumps({"folds": results, "mean": mean, "rule": __doc__}, indent=2), encoding="utf-8")
    log(f"mean {json.dumps(mean)}")


if __name__ == "__main__":
    main()
