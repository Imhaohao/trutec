"""Score every cutoff rule on the development split. Choices were already saved on fit."""
from __future__ import annotations

import math

import limits  # noqa: F401

import joblib
import numpy as np

from cutoffs import best_cutoffs
from devharness import THRESHOLDS, development_queries, report, summarize_decisions
from experiment import CLIP_HIGH, CLIP_LOW, GLOBAL_V6_THRESHOLD, MEDIAN_FRACTIONS
from fit_metrics import SUMMARY_KEYS
from limits import DEV_REPORT_PATH, MODEL_CACHE, PUBLISHED_V6, log, read_json, write_json
from rules import clipped_name, fraction_name
from scoremap import feature_index, feature_names
from store import read_dev

HERE_SWEEP = DEV_REPORT_PATH.with_name("dev_sweep.json")
PREDICTIONS_PATH = DEV_REPORT_PATH.with_name("dev_predictions.json")
STATED_ORACLE = {
    "document_macro_iou": 0.9159,
    "global_recall": 0.973,
    "global_precision": 0.589,
    "source": "stated in the experiment prompt; not a number computed by this run",
}


def score_development(chosen: dict) -> dict:
    queries = development_queries()
    features, scores = _load(queries)
    oracle_1001 = _oracle_values(queries, scores, 1001)
    oracle_101 = _oracle_values(queries, scores, 101)
    variants = []
    variants.extend(_learned_variants(queries, features, scores, chosen, oracle_1001))
    variants.extend(_rule_variants(queries, features, scores, chosen))
    variants.extend(_oracle_variants(queries, scores, oracle_1001, oracle_101))
    variants.append(_constant_variant("global_0.44", scores, _harness_threshold(), "global"))
    payload = {
        "variants": variants,
        "reproduction": _reproduction(scores),
        "sweep_selections": _sweep(scores),
        "stated_oracle_from_prompt": STATED_ORACLE,
        "queries": len(queries),
    }
    write_json(DEV_REPORT_PATH, payload)
    _write_predictions(queries, features, chosen, oracle_1001)
    log(f"development variants={len(variants)} reproduction_match={payload['reproduction']['counts_match']}")
    return payload


def _load(queries):
    features = {}
    scores = {}
    for query in queries:
        stored = read_dev(query.id)
        if len(stored["reviewed_scores"]) != len(query.y):
            raise RuntimeError(f"reviewed length {query.id}")
        features[query.id] = stored["features"]
        scores[query.id] = stored["reviewed_scores"]
    return features, scores


def _learned_variants(queries, features, scores, chosen, oracle_values) -> list[dict]:
    matrix = np.stack([features[query.id] for query in queries])
    rows = []
    for name in chosen["model_names"]:
        rows.extend(_one_model(name, matrix, queries, scores, chosen, oracle_values))
    return rows


def _one_model(name, matrix, queries, scores, chosen, oracle_values) -> list[dict]:
    model, spec = _load_model(name)
    raw = np.asarray(model.predict(matrix), dtype=np.float64)
    clipped = np.clip(raw, CLIP_LOW, CLIP_HIGH)
    selected = name == chosen["chosen_model"]
    raw_row = _variant(name, spec["target"], False, queries, scores, raw, selected)
    clip_row = _variant(clipped_name(name), spec["target"], True, queries, scores, clipped, False)
    raw_row["mae_vs_oracle_1001"] = _mae(raw, oracle_values)
    clip_row["mae_vs_oracle_1001"] = _mae(clipped, oracle_values)
    clip_row["selected_blend"] = selected
    return [raw_row, clip_row]


def _rule_variants(queries, features, scores, chosen) -> list[dict]:
    rows = [_feature_variant("otsu", queries, features, scores, "otsu", 1.0, False)]
    for fraction in MEDIAN_FRACTIONS:
        name = fraction_name(fraction)
        selected = name == chosen["median_fraction_name"]
        rows.append(_feature_variant(name, queries, features, scores, "query_median", fraction, selected))
    return rows


def _feature_variant(name, queries, features, scores, feature_name, scale, selected) -> dict:
    index = feature_index(feature_name)
    values = np.array([scale * float(features[query.id][index]) for query in queries], dtype=np.float64)
    row = _variant(name, "rule", False, queries, scores, values, selected)
    row["family"] = "rule"
    return row


def _oracle_variants(queries, scores, oracle_1001, oracle_101) -> list[dict]:
    fine = _variant("oracle_grid_1001", "oracle_not_deployable", False, queries, scores, oracle_1001, False)
    coarse = _variant("oracle_grid_101", "oracle_not_deployable", False, queries, scores, oracle_101, False)
    return [fine, coarse]


def _constant_variant(name, scores, threshold: float, family: str) -> dict:
    values = np.full(len(scores), threshold, dtype=np.float64)
    queries = development_queries()
    row = _variant(name, family, False, queries, scores, values, False)
    row["family"] = family
    return row


def _variant(name, family, clipped, queries, scores, values, selected) -> dict:
    if len(values) != len(queries) or not np.isfinite(values).all():
        raise RuntimeError(f"bad cutoffs for {name}")
    cutoffs = {query.id: float(value) for query, value in zip(queries, values)}
    return {
        "name": name,
        "family": family,
        "clipped": clipped,
        "selected_on_fit": selected,
        "cutoff_mean": float(np.mean(values)),
        "cutoff_min": float(np.min(values)),
        "cutoff_max": float(np.max(values)),
        "summary": _summarize(scores, cutoffs),
    }


def _summarize(scores, cutoffs) -> dict:
    decisions = {key: scores[key] >= cutoffs[key] for key in scores}
    summary = summarize_decisions(decisions)
    return {key: summary[key] for key in SUMMARY_KEYS}


def _oracle_values(queries, scores, steps: int) -> np.ndarray:
    values = []
    for query in queries:
        labels = np.asarray(query.positive, dtype=bool)
        iou_cutoff, _recall_cutoff = best_cutoffs(scores[query.id][labels], scores[query.id][~labels], steps)
        if not math.isfinite(iou_cutoff):
            raise RuntimeError(f"no oracle cutoff for {query.id}")
        values.append(iou_cutoff)
    return np.asarray(values, dtype=np.float64)


def _load_model(name: str):
    payload = joblib.load(MODEL_CACHE / f"{name}.joblib")
    if list(payload["feature_names"]) != feature_names():
        raise RuntimeError(f"feature names drifted for {name}")
    return payload["model"], payload["spec"]


def _reproduction(scores) -> dict:
    threshold = _harness_threshold()
    ours = _constant_variant("reproduction", scores, threshold, "global")["summary"]
    published = _published_point(threshold)
    keys = ("tp", "fp", "fn", "tn")
    return {
        "threshold": threshold,
        "ours": ours,
        "published": {key: published[key] for key in (*keys, *SUMMARY_KEYS)},
        "counts_match": all(int(ours[key]) == int(published[key]) for key in keys),
    }


def _sweep(scores) -> dict:
    result = report("qthreshold_v6_grids", scores, HERE_SWEEP)
    return result["selections"]


def _published_point(threshold: float) -> dict:
    points = read_json(PUBLISHED_V6)["points"]
    return min(points, key=lambda row: abs(float(row["threshold"]) - threshold))


def _harness_threshold() -> float:
    index = int(np.argmin(np.abs(np.asarray(THRESHOLDS) - GLOBAL_V6_THRESHOLD)))
    return float(THRESHOLDS[index])


def _write_predictions(queries, features, chosen, oracle_values) -> None:
    model, _spec = _load_model(chosen["chosen_model"])
    matrix = np.stack([features[query.id] for query in queries])
    raw = np.asarray(model.predict(matrix), dtype=np.float64)
    median_index = feature_index("query_median")
    otsu_index = feature_index("otsu")
    rows = []
    for index, query in enumerate(queries):
        rows.append(
            {
                "id": query.id,
                "document_id": query.document_id,
                "chosen_cutoff": float(raw[index]),
                "chosen_clipped": float(np.clip(raw[index], CLIP_LOW, CLIP_HIGH)),
                "otsu": float(features[query.id][otsu_index]),
                "query_median": float(features[query.id][median_index]),
                "median_rule": float(chosen["median_fraction"] * features[query.id][median_index]),
                "oracle_1001": float(oracle_values[index]),
            }
        )
    write_json(PREDICTIONS_PATH, rows)


def _mae(predicted: np.ndarray, oracle_values: np.ndarray) -> float:
    return float(np.mean(np.abs(predicted - oracle_values)))
