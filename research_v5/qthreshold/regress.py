"""Fit-split regressors. Document-grouped cross-validation chooses the cutoff model."""
from __future__ import annotations

import limits  # noqa: F401

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from devharness import fit_rows
from experiment import RIDGE_ALPHAS, TREE_SPECS
from fit_metrics import score_many
from folds import load_document_folds
from limits import CHOSEN_PATH, FOLDS_PATH, MODEL_CACHE, dump_joblib, log, write_json
from rules import clip_column, clipped_name, fraction_name, rule_columns
from scoremap import feature_names
from store import read_fit_header

FOLD_COUNT = 4


class ScaledRidge:
    def __init__(self, alpha: float):
        self.alpha = float(alpha)
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=self.alpha)

    def fit(self, features, target, sample_weight):
        self.scaler.fit(features, sample_weight=sample_weight)
        self.model.fit(self.scaler.transform(features), target, sample_weight=sample_weight)
        return self

    def predict(self, features):
        return self.model.predict(self.scaler.transform(features))


class TreeCutoff:
    def __init__(self, leaves: int, iters: int):
        self.model = HistGradientBoostingRegressor(
            learning_rate=0.08,
            max_leaf_nodes=int(leaves),
            max_iter=int(iters),
            min_samples_leaf=5,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=42,
        )

    def fit(self, features, target, sample_weight):
        self.model.fit(features, target, sample_weight=sample_weight)
        return self

    def predict(self, features):
        return self.model.predict(features)


def candidate_specs() -> list[dict]:
    specs = []
    for target in ("iou", "recall98"):
        specs.extend(_ridge_specs(target))
        specs.extend(_tree_specs(target))
    return specs


def select_and_fit() -> dict:
    table = load_table()
    predicted = {spec["name"]: _oof_predict(table, spec) for spec in candidate_specs()}
    scored = score_many(table["id"].tolist(), table["document"].tolist(), _columns(table, predicted))
    ranked = _rank(table, predicted, scored)
    _refit_all(table)
    fraction = _choose_fraction(scored)
    payload = _payload(table, ranked, fraction, scored)
    write_json(CHOSEN_PATH, payload)
    log(f"fit chose {payload['chosen_model']} fraction {payload['median_fraction']}")
    return payload


def load_table() -> dict:
    folds = load_document_folds(FOLDS_PATH)
    built = [_one_row(row, folds) for row in fit_rows()]
    return {
        "id": np.array([item["id"] for item in built]),
        "document": np.array([item["document_id"] for item in built]),
        "fold": np.array([item["fold"] for item in built], dtype=np.int16),
        "x": np.stack([item["features"] for item in built]),
        "iou": np.array([item["target_iou"] for item in built], dtype=np.float64),
        "recall98": np.array([item["target_recall98"] for item in built], dtype=np.float64),
    }


def make_regressor(spec: dict):
    if spec["kind"] == "ridge":
        return ScaledRidge(float(spec["alpha"]))
    return TreeCutoff(int(spec["leaves"]), int(spec["iters"]))


def document_weights(documents: np.ndarray) -> np.ndarray:
    counts: dict[str, int] = {}
    labels = documents.tolist()
    for document in labels:
        counts[document] = counts.get(document, 0) + 1
    return np.array([1.0 / counts[document] for document in labels], dtype=np.float64)


def _ridge_specs(target: str) -> list[dict]:
    return [_spec(f"ridge_a{alpha:.1f}_{target}", "ridge", target, alpha, 0, 0) for alpha in RIDGE_ALPHAS]


def _tree_specs(target: str) -> list[dict]:
    rows = []
    for leaves, iters in TREE_SPECS:
        rows.append(_spec(f"hgb_leaves{leaves}_{target}", "hgb", target, 0.0, leaves, iters))
    return rows


def _spec(name: str, kind: str, target: str, alpha: float, leaves: int, iters: int) -> dict:
    return {"name": name, "kind": kind, "target": target, "alpha": alpha, "leaves": leaves, "iters": iters}


def _oof_predict(table: dict, spec: dict) -> np.ndarray:
    target = table[spec["target"]]
    predicted = np.empty(len(target), dtype=np.float64)
    for fold in range(FOLD_COUNT):
        _predict_fold(table, spec, target, fold, predicted)
    if not np.isfinite(predicted).all():
        raise RuntimeError(f"non-finite out-of-fold prediction for {spec['name']}")
    log(f"oof {spec['name']} mae={_mae(predicted, target):.4f}")
    return predicted


def _predict_fold(table, spec, target, fold: int, predicted: np.ndarray) -> None:
    train = (table["fold"] != fold) & np.isfinite(target)
    test = table["fold"] == fold
    if not np.any(train) or not np.any(test):
        raise RuntimeError(f"empty fold {fold} for {spec['name']}")
    model = make_regressor(spec)
    model.fit(table["x"][train], target[train], document_weights(table["document"][train]))
    predicted[test] = model.predict(table["x"][test])


def _columns(table: dict, predicted: dict[str, np.ndarray]) -> dict[str, list[float]]:
    columns = {name: values.tolist() for name, values in predicted.items()}
    for name, values in predicted.items():
        columns[clipped_name(name)] = clip_column(values).tolist()
    columns.update(rule_columns(table["x"]))
    return columns


def _rank(table, predicted, scored) -> list[dict]:
    rows = [_candidate_row(spec, predicted, scored, table) for spec in candidate_specs()]
    rows.sort(key=_rank_key)
    return rows


def _candidate_row(spec, predicted, scored, table) -> dict:
    target = table[spec["target"]]
    return {
        "name": spec["name"],
        "target": spec["target"],
        "kind": spec["kind"],
        "oof": scored[spec["name"]],
        "oof_clipped": scored[clipped_name(spec["name"])],
        "mae": _mae(predicted[spec["name"]], target),
        "correlation": _correlation(predicted[spec["name"]], target),
    }


def _rank_key(row: dict) -> tuple:
    macro = row["oof"]["document_macro_iou"]
    precision = row["oof"]["global_precision"]
    return (
        -(macro if macro is not None else -1.0),
        -(precision if precision is not None else -1.0),
        row["name"],
    )


def _choose_fraction(scored: dict) -> dict:
    ranked = [_fraction_row(fraction, scored) for fraction in _fractions()]
    ranked.sort(key=_fraction_key)
    winner = ranked[0]
    return {"name": winner["name"], "fraction": winner["fraction"], "ranked": ranked}


def _fractions():
    from experiment import MEDIAN_FRACTIONS

    return MEDIAN_FRACTIONS


def _fraction_row(fraction: float, scored: dict) -> dict:
    name = fraction_name(fraction)
    return {"name": name, "fraction": fraction, "fit": scored[name]}


def _fraction_key(row: dict) -> tuple:
    macro = row["fit"]["document_macro_iou"]
    precision = row["fit"]["global_precision"]
    return (
        -(macro if macro is not None else -1.0),
        -(precision if precision is not None else -1.0),
        -row["fraction"],
    )


def _refit_all(table: dict) -> None:
    for spec in candidate_specs():
        _refit_one(table, spec)


def _refit_one(table: dict, spec: dict) -> None:
    target = table[spec["target"]]
    train = np.isfinite(target)
    if not np.any(train):
        raise RuntimeError(f"no trainable rows for {spec['name']}")
    model = make_regressor(spec)
    model.fit(table["x"][train], target[train], document_weights(table["document"][train]))
    dump_joblib(
        MODEL_CACHE / f"{spec['name']}.joblib",
        {"model": model, "spec": spec, "feature_names": feature_names()},
    )


def _payload(table, ranked, fraction, scored) -> dict:
    return {
        "selection": (
            "Fit document-grouped out-of-fold document-macro IoU on the raw predicted cutoff. "
            "Tie break is higher pooled precision, then name. The clip is not used for selection."
        ),
        "fraction_selection": (
            "Full fit document-macro IoU of fraction times query-box median. "
            "Tie break is higher pooled precision, then the larger fraction."
        ),
        "chosen_model": ranked[0]["name"],
        "chosen_target": ranked[0]["target"],
        "model_names": [row["name"] for row in ranked],
        "candidates": ranked,
        "median_fraction": fraction["fraction"],
        "median_fraction_name": fraction["name"],
        "fraction_ranked": fraction["ranked"],
        "otsu_fit": scored["otsu"],
        "target_iou": _distribution(table["iou"]),
        "target_recall98": _distribution(table["recall98"]),
        "queries": int(len(table["id"])),
    }


def _one_row(row: dict, folds: dict[str, int]) -> dict:
    header = read_fit_header(row["id"])
    expected = int(folds[row["document_id"]])
    if header["fold"] != expected or header["document_id"] != row["document_id"]:
        raise RuntimeError(f"cache fold mismatch for {row['id']}")
    header["id"] = row["id"]
    return header


def _distribution(values: np.ndarray) -> dict:
    mask = np.isfinite(values)
    if not np.any(mask):
        return {"count": 0, "mean": None, "std": None}
    held = values[mask]
    return {"count": int(mask.sum()), "mean": float(np.mean(held)), "std": float(np.std(held))}


def _mae(predicted: np.ndarray, target: np.ndarray) -> float | None:
    mask = np.isfinite(target)
    if not np.any(mask):
        return None
    return float(np.mean(np.abs(predicted[mask] - target[mask])))


def _correlation(predicted: np.ndarray, target: np.ndarray) -> float | None:
    mask = np.isfinite(target)
    if int(mask.sum()) < 3:
        return None
    left = predicted[mask]
    right = target[mask]
    if float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])
