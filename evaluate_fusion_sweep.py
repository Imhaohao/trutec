"""Cached, development-only fusion sweep over reviewed probability vectors.

The texture HGB and adapted CNN probabilities are already computed.  This
script only validates those sparse artifacts, combines their decisions, and
scores known pixels with the official evaluator functions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np

from evaluate import EvaluationError, aggregate, load_manifest, metrics_from_counts, reviewed_domains
from evaluate_texture_sweep import validate_split_safety


TEXTURE_GLOBAL_THRESHOLDS = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70)
TEXTURE_QUERY_SELF_RULES = (
    ("q10_scale_0.3_offset_-0.2", "q10", 0.3, -0.2, "clip(query_self.q10 * 0.3 - 0.2, 0.01, 0.99)"),
    ("q10_scale_0.5_offset_0", "q10", 0.5, 0.0, "clip(query_self.q10 * 0.5 + 0, 0.01, 0.99)"),
    ("q10_scale_0.7_offset_-0.2", "q10", 0.7, -0.2, "clip(query_self.q10 * 0.7 - 0.2, 0.01, 0.99)"),
    ("q10_scale_0.7_offset_-0.1", "q10", 0.7, -0.1, "clip(query_self.q10 * 0.7 - 0.1, 0.01, 0.99)"),
    ("median_scale_0.7_offset_0", "median", 0.7, 0.0, "clip(query_self.median * 0.7, 0.01, 0.99)"),
    ("median_scale_0.9_offset_-0.2", "median", 0.9, -0.2, "clip(query_self.median * 0.9 - 0.2, 0.01, 0.99)"),
)
CNN_THRESHOLDS = (0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
OPERATORS = ("AND", "OR")
REQUIRED_QUANTILES = ("median", "q10", "q25", "q75")
RECALL_GATE = 0.95


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--texture-probability-dir", required=True, type=Path)
    parser.add_argument("--cnn-probability-dir", required=True, type=Path)
    parser.add_argument("--query-self", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(None if argv is None else list(argv))


def build_texture_candidates() -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = [
        {
            "candidate_id": f"texture-global-{threshold:g}",
            "kind": "global",
            "formula": f"texture_probability >= {threshold:g}",
            "threshold": threshold,
            "quantile": None,
            "scale": None,
            "offset": None,
            "clip": None,
        }
        for threshold in TEXTURE_GLOBAL_THRESHOLDS
    ]
    candidates.extend(
        {
            "candidate_id": f"texture-query-self-{name}",
            "kind": "query_self",
            "formula": formula,
            "threshold": None,
            "quantile": quantile,
            "scale": scale,
            "offset": offset,
            "clip": [0.01, 0.99],
        }
        for name, quantile, scale, offset, formula in TEXTURE_QUERY_SELF_RULES
    )
    return candidates


def build_fusion_candidates() -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for texture in build_texture_candidates():
        for cnn_threshold in CNN_THRESHOLDS:
            for operator in OPERATORS:
                candidate_id = f"{operator.lower()}|{texture['candidate_id']}|cnn-{cnn_threshold:g}"
                candidates.append({
                    "candidate_id": candidate_id, "operator": operator, "texture": texture,
                    "cnn": {"threshold": cnn_threshold, "formula": f"cnn_probability >= {cnn_threshold:g}"},
                    "selection": f"{texture['formula']} {operator} cnn_probability >= {cnn_threshold:g}",
                })
    return candidates


def load_query_self(path: Path) -> dict[str, dict[str, float]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"Cannot read query-self report {path}: {exc}") from exc
    entries = payload.get("queries") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        raise EvaluationError("Query-self report must contain a queries list")
    result: dict[str, dict[str, float]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            raise EvaluationError(f"Invalid query-self entry in {path}")
        ident = entry["id"]
        if ident in result:
            raise EvaluationError(f"Duplicate query-self entry: {ident}")
        values = entry.get("query_self")
        if not isinstance(values, dict):
            raise EvaluationError(f"Query-self entry lacks query_self values: {ident}")
        parsed: dict[str, float] = {}
        for quantile in REQUIRED_QUANTILES:
            value = values.get(quantile)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise EvaluationError(f"{ident} {quantile} must be numeric")
            parsed_value = float(value)
            if not np.isfinite(parsed_value) or not 0.0 <= parsed_value <= 1.0:
                raise EvaluationError(f"{ident} {quantile} must be within [0, 1]")
            parsed[quantile] = parsed_value
        result[ident] = parsed
    if not result:
        raise EvaluationError(f"Query-self report contains no entries: {path}")
    return result


def _load_npz(
    path: Path,
    expected_positive: np.ndarray,
    expected_blank: np.ndarray,
    expected_y: np.ndarray,
    expected_x: np.ndarray,
    source: str,
    query_id: str,
) -> np.ndarray:
    if not path.is_file():
        raise EvaluationError(f"Missing {source} probability artifact for {query_id}: {path}")
    try:
        with np.load(path, allow_pickle=False) as data:
            if not {"probability", "positive", "blank"}.issubset(data.files):
                raise EvaluationError(f"Incomplete {source} probability artifact: {path}")
            probability = np.asarray(data["probability"])
            positive = np.asarray(data["positive"])
            blank = np.asarray(data["blank"])
            coordinate_keys = {"y", "x"} & set(data.files)
            coordinates = None
            if coordinate_keys:
                if coordinate_keys != {"y", "x"}:
                    raise EvaluationError(f"{source} coordinates must include y and x: {path}")
                coordinates = (np.asarray(data["y"]), np.asarray(data["x"]))
    except EvaluationError:
        raise
    except (OSError, ValueError, EOFError) as exc:
        raise EvaluationError(f"Cannot read {source} probability artifact {path}: {exc}") from exc

    _validate_npz(
        probability,
        positive,
        blank,
        expected_positive,
        expected_blank,
        coordinates,
        expected_y,
        expected_x,
        path,
        source,
    )
    return probability


def _validate_npz(
    probability: np.ndarray,
    positive: np.ndarray,
    blank: np.ndarray,
    expected_positive: np.ndarray,
    expected_blank: np.ndarray,
    coordinates: tuple[np.ndarray, np.ndarray] | None,
    expected_y: np.ndarray,
    expected_x: np.ndarray,
    path: Path,
    source: str,
) -> None:
    if probability.dtype != np.float32 or probability.ndim != 1:
        raise EvaluationError(f"{source} probabilities must be a float32 vector: {path}")
    if positive.dtype != np.bool_ or blank.dtype != np.bool_:
        raise EvaluationError(f"{source} labels must be boolean vectors: {path}")
    if positive.ndim != 1 or blank.ndim != 1 or probability.shape != expected_positive.shape:
        raise EvaluationError(f"{source} vector shapes do not match known pixels: {path}")
    if positive.shape != expected_positive.shape or blank.shape != expected_blank.shape:
        raise EvaluationError(f"{source} label shapes do not match known pixels: {path}")
    if not np.array_equal(positive, expected_positive) or not np.array_equal(blank, expected_blank):
        raise EvaluationError(f"{source} labels do not match reviewed domains: {path}")
    if not np.all(np.isfinite(probability)) or np.any((probability < 0.0) | (probability > 1.0)):
        raise EvaluationError(f"{source} probabilities are outside [0, 1]: {path}")
    if coordinates is not None:
        _validate_coordinates(coordinates, expected_y, expected_x, path, source)


def _validate_coordinates(
    coordinates: tuple[np.ndarray, np.ndarray],
    expected_y: np.ndarray,
    expected_x: np.ndarray,
    path: Path,
    source: str,
) -> None:
    y, x = coordinates
    if y.dtype.kind not in "iu" or x.dtype.kind not in "iu":
        raise EvaluationError(f"{source} coordinates must be integer vectors: {path}")
    if y.shape != expected_y.shape or x.shape != expected_x.shape:
        raise EvaluationError(f"{source} coordinate shapes do not match known pixels: {path}")
    if not np.array_equal(y, expected_y) or not np.array_equal(x, expected_x):
        raise EvaluationError(f"{source} coordinates do not match known row-major pixels: {path}")


def load_cached_queries(
    rows: list[dict[str, Any]],
    data_root: Path,
    texture_dir: Path,
    cnn_dir: Path,
    query_self: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    ids = {str(row["id"]) for row in rows}
    if ids != set(query_self):
        raise EvaluationError(
            f"Query-self IDs do not match manifest: missing={sorted(ids - set(query_self))[:5]}, "
            f"extra={sorted(set(query_self) - ids)[:5]}"
        )
    records: list[dict[str, Any]] = []
    for row in rows:
        positive, known, blank, metadata = reviewed_domains(row, data_root)
        y, x = np.nonzero(known)
        positive_vector = np.asarray(positive[known], dtype=bool)
        blank_vector = np.asarray(blank[known], dtype=bool)
        texture = _load_npz(texture_dir / f"{row['id']}.npz", positive_vector, blank_vector, y, x, "texture", str(row["id"]))
        cnn = _load_npz(cnn_dir / f"{row['id']}.npz", positive_vector, blank_vector, y, x, "cnn", str(row["id"]))
        if texture.shape != cnn.shape or len(texture) != len(y):
            raise EvaluationError(f"Probability vectors do not match known pixels: {row['id']}")
        records.append({"id": row["id"], "document_id": row["document_id"], "kind": row["kind"],
                        "texture": texture, "cnn": cnn, "positive": positive_vector,
                        "blank": blank_vector, "metadata": metadata,
                        "query_self": query_self[str(row["id"])]})
    return records


def _texture_threshold(texture: dict[str, Any], query_self: dict[str, float]) -> float:
    if texture["kind"] == "global":
        return float(texture["threshold"])
    return float(np.clip(
        texture["scale"] * query_self[texture["quantile"]] + texture["offset"], 0.01, 0.99
    ))


def _metric(
    record: dict[str, Any], selected: np.ndarray, texture_threshold: float, config: dict[str, Any]
) -> dict[str, Any]:
    positive = record["positive"]
    blank = record["blank"]
    positive_count = int(positive.sum())
    tp = int(np.count_nonzero(selected & positive))
    fp = int(np.count_nonzero(selected & ~positive))
    fn = positive_count - tp
    tn = len(positive) - positive_count - fp
    return {"id": record["id"], "document_id": record["document_id"], "kind": record["kind"],
            **record["metadata"], "texture_threshold": texture_threshold,
            "cnn_threshold": config["cnn"]["threshold"], "operator": config["operator"],
            "metrics": metrics_from_counts(tp, fp, fn, tn,
                                            int(np.count_nonzero(selected & blank)), int(blank.sum()))}


def score_candidates(
    records: list[dict[str, Any]], configs: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {config["candidate_id"]: [] for config in configs}
    textures = {config["texture"]["candidate_id"]: config["texture"] for config in configs}
    for record in records:
        texture_masks = {
            ident: record["texture"] >= _texture_threshold(texture, record["query_self"])
            for ident, texture in textures.items()
        }
        cnn_masks = {f"{threshold:g}": record["cnn"] >= threshold for threshold in CNN_THRESHOLDS}
        for config in configs:
            texture = texture_masks[config["texture"]["candidate_id"]]
            cnn = cnn_masks[f"{config['cnn']['threshold']:g}"]
            selected = texture & cnn if config["operator"] == "AND" else texture | cnn
            output[config["candidate_id"]].append(_metric(
                record, selected, _texture_threshold(config["texture"], record["query_self"]), config))
    return output


def _rank(summary: dict[str, Any], candidate_id: str) -> tuple[float, float, float, str]:
    def value(key: str) -> float:
        item = summary.get(key)
        return float(item) if isinstance(item, (int, float)) else float("-inf")

    return (value("document_macro_iou"), value("global_recall"), value("global_precision"), candidate_id)


def choose_best_configs(
    configs: list[dict[str, Any]], summaries: dict[str, dict[str, Any]]
) -> tuple[str, str | None]:
    best = max(configs, key=lambda config: _rank(summaries[config["candidate_id"]], config["candidate_id"]))
    eligible = [
        config
        for config in configs
        if isinstance(summaries[config["candidate_id"]].get("global_recall"), (int, float))
        and summaries[config["candidate_id"]]["global_recall"] >= RECALL_GATE
    ]
    gated = max(eligible, key=lambda config: _rank(summaries[config["candidate_id"]], config["candidate_id"])) if eligible else None
    return best["candidate_id"], gated["candidate_id"] if gated else None


def _hash_file(path: Path) -> str:
    if not path.is_file():
        raise EvaluationError(f"Missing input artifact: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash_dir(path: Path) -> str:
    if not path.is_dir():
        raise EvaluationError(f"Missing probability directory: {path}")
    files = sorted(path.glob("*.npz"))
    if not files:
        raise EvaluationError(f"No .npz probability artifacts found: {path}")
    digest = hashlib.sha256()
    for file_path in files:
        digest.update(file_path.name.encode("utf-8"))
        digest.update(file_path.read_bytes())
    return digest.hexdigest()


def _write_query_records(
    path: Path,
    records: list[dict[str, Any]],
    scored: dict[str, list[dict[str, Any]]],
    configs: dict[str, dict[str, Any]],
    selected_ids: list[str],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(records):
            scores = {}
            for candidate_id in selected_ids:
                query_record = scored[candidate_id][index]
                scores[candidate_id] = {"config": configs[candidate_id],
                    "texture_threshold": query_record["texture_threshold"],
                    "cnn_threshold": query_record["cnn_threshold"], "operator": query_record["operator"],
                    "metrics": query_record["metrics"]}
            handle.write(json.dumps({"id": record["id"], "document_id": record["document_id"],
                "kind": record["kind"], "scores": scores}, sort_keys=True) + "\n")


def build_report(
    rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
    configs: list[dict[str, Any]],
    summaries: dict[str, dict[str, Any]],
    best_id: str,
    gated_id: str | None,
    args: argparse.Namespace,
    elapsed_seconds: float,
) -> dict[str, Any]:
    hashes = {
        "manifest_sha256": _hash_file(args.manifest),
        "texture_probability_dir_sha256": _hash_dir(args.texture_probability_dir),
        "cnn_probability_dir_sha256": _hash_dir(args.cnn_probability_dir),
        "query_self_sha256": _hash_file(args.query_self),
        "implementation_sha256": _hash_file(Path(__file__)),
    }
    config_map = {config["candidate_id"]: config for config in configs}
    totals = {"known_pixels": 0, "positive_pixels": 0, "negative_pixels": 0, "blank_pixels": 0}
    for record in records:
        known = len(record["positive"])
        positive = int(record["positive"].sum())
        blank = int(record["blank"].sum())
        totals["known_pixels"] += known
        totals["positive_pixels"] += positive
        totals["negative_pixels"] += known - positive
        totals["blank_pixels"] += blank
    documents = len({str(row["document_id"]) for row in rows})
    eligibility = {
        **totals,
        "queries": len(rows),
        "documents": documents,
        "real_queries": sum(row.get("kind") == "real" for row in rows),
        "real_documents": len({str(row["document_id"]) for row in rows if row.get("kind") == "real"}),
        "candidate_texture_rules": len(build_texture_candidates()),
        "candidate_cnn_thresholds": len(CNN_THRESHOLDS),
        "fusion_operators": list(OPERATORS),
        "fusion_configs": len(configs),
        "recall_gate": RECALL_GATE,
        "recall_eligible_configs": sum(
            isinstance(summaries[config["candidate_id"]].get("global_recall"), (int, float))
            and summaries[config["candidate_id"]]["global_recall"] >= RECALL_GATE
            for config in configs
        ),
    }
    exposure = {
        **totals,
        "scoring_domain": "all reviewed_domains known pixels after query/context exclusion",
        "negative_domain": "known pixels minus reviewed positive pixels",
        "blank_domain": "explicit blank pixels intersected with known pixels",
        "unscored_domain": "unknown pixels and supplied query/context rectangles",
    }
    threshold_formulas = {
        "texture_global": "texture_probability >= global_threshold",
        "texture_query_self": "texture_probability >= clip(query_self.quantile * scale + offset, 0.01, 0.99)",
        "cnn": "cnn_probability >= cnn_threshold",
        "operators": {"AND": "texture_selected and cnn_selected", "OR": "texture_selected or cnn_selected"},
        "query_self_candidates": [
            {"quantile": quantile, "scale": scale, "offset": offset, "formula": formula}
            for _, quantile, scale, offset, formula in TEXTURE_QUERY_SELF_RULES
        ],
        "adaptive_clip": [0.01, 0.99],
        "texture_global_thresholds": list(TEXTURE_GLOBAL_THRESHOLDS),
        "cnn_thresholds": list(CNN_THRESHOLDS),
    }
    return {
        "schema": "hatch-matching-challenge-fusion-sweep/v1",
        "method": "cached-probability-fusion-sweep-v1",
        "partition": "training-development only",
        "development_only": True,
        "whole_sheet_accuracy_certified": False,
        "inference_claim": None,
        "official_metric_sources": {
            "reviewed_domains": "evaluate.reviewed_domains",
            "per_query": "evaluate.metrics_from_counts",
            "aggregate": "evaluate.aggregate",
        },
        "hashes": hashes,
        **hashes,
        "source_manifest_sha256": hashes["manifest_sha256"],
        "queries": len(rows),
        "documents": documents,
        "real_document_count": eligibility["real_documents"],
        "elapsed_seconds": elapsed_seconds,
        "candidate_eligibility": eligibility,
        "eligible_counts": eligibility,
        "exposure_domains": exposure,
        "threshold_formulas": threshold_formulas,
        "candidate_configs": config_map,
        "summaries": summaries,
        "all_summaries": summaries,
        "best_overall": {
            "candidate_id": best_id,
            "selection_metric": "document_macro_iou",
            "config": config_map[best_id],
            "summary": summaries[best_id],
        },
        "best_recall_at_least_0.95": (
            {
                "candidate_id": gated_id,
                "selection_metric": "document_macro_iou",
                "config": config_map[gated_id],
                "summary": summaries[gated_id],
            }
            if gated_id is not None
            else None
        ),
        "recall_constraint": {"metric": "global_recall", "minimum": RECALL_GATE},
        "query_records": "query-records.jsonl",
        "exact_known_scoring_only": True,
        "production_inference_distinction": (
            "This cached experiment thresholds probabilities at reviewed pixels only. "
            "It does not produce complete native masks or make a full-sheet inference claim."
        ),
        "timing": {
            "elapsed_seconds": elapsed_seconds,
            "timing_coverage": "cached NPZ loading, validation, threshold combinations, official metrics, hashing, and report writing",
        },
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        rows = load_manifest(args.manifest, require_labels=True)
        validate_split_safety(args.manifest, args.data_root, rows)
        query_self = load_query_self(args.query_self)
        started = time.monotonic()
        records = load_cached_queries(
            rows,
            args.data_root,
            args.texture_probability_dir,
            args.cnn_probability_dir,
            query_self,
        )
        configs = build_fusion_candidates()
        scored = score_candidates(records, configs)
        summaries = {
            candidate_id: aggregate(query_records)
            for candidate_id, query_records in scored.items()
        }
        best_id, gated_id = choose_best_configs(configs, summaries)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        config_map = {config["candidate_id"]: config for config in configs}
        selected_ids = [best_id] + ([gated_id] if gated_id and gated_id != best_id else [])
        _write_query_records(
            args.output_dir / "query-records.jsonl", records, scored, config_map, selected_ids
        )
        report = build_report(
            rows,
            records,
            configs,
            summaries,
            best_id,
            gated_id,
            args,
            time.monotonic() - started,
        )
        (args.output_dir / "sweep.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "best_overall": report["best_overall"],
                    "best_recall_at_least_0.95": report["best_recall_at_least_0.95"],
                },
                indent=2,
                sort_keys=True,
                allow_nan=False,
            ),
            flush=True,
        )
    except (EvaluationError, OSError, ValueError, TypeError) as exc:
        build_parser().exit(2, "Fusion sweep failed: " + str(exc) + "\n")


if __name__ == "__main__":
    main()
