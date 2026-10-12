"""Development-only query-self calibration over saved texture probabilities."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Sequence, cast

import numpy as np

from evaluate import EvaluationError, aggregate, load_manifest, metrics_from_counts, reviewed_domains
from evaluate_texture_sweep import DEFAULT_THRESHOLDS, validate_split_safety
import hatch_texture
from inference import read_drawing
from inference_texture import (
    _nearest_query_cell,
    _predict_probabilities,
    query_self_quantiles,
    threshold_from_query_self,
    verify_model_metadata,
)
from texture_learning import load_texture_model

SCALES: tuple[float, ...] = (0.3, 0.5, 0.7, 0.9, 1.0, 1.1)
OFFSETS: tuple[float, ...] = (-0.2, -0.1, 0.0, 0.1)
QUANTILES: tuple[str, ...] = ("median", "q10", "q25", "q75")
CLIP_LOW = 0.01
CLIP_HIGH = 0.99


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--probability-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(None if argv is None else list(argv))


def run_calibration(
    rows: list[dict[str, Any]],
    data_root: Path,
    model: Any,
    metadata: dict[str, Any],
    probability_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    """Compute query-self summaries and official baseline/calibration records."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["image"])].append(row)
    stride = int(metadata.get("stride", 4))
    source = hatch_texture.descriptor_source_from_metadata(metadata)
    self_rows: list[dict[str, Any]] = []
    baseline_records: dict[str, list[dict[str, Any]]] = {str(t): [] for t in DEFAULT_THRESHOLDS}
    calibration_records: dict[str, list[dict[str, Any]]] = {}
    for quantile in QUANTILES:
        for scale in SCALES:
            for offset in OFFSETS:
                calibration_records[_config_key(quantile, scale, offset)] = []

    for query_group in grouped.values():
        gray = read_drawing(data_root, query_group[0])
        features = hatch_texture.texture_features(gray, stride=stride)
        for row in query_group:
            query_started = time.monotonic()
            if gray.shape != (row["height"], row["width"]):
                raise EvaluationError(f"Inconsistent image dimensions for query {row['id']}")
            positive, known, blank, domain_meta = reviewed_domains(row, data_root)
            probabilities = _load_saved_probabilities(
                probability_dir / f"{row['id']}.npz", positive, known, blank, row
            )
            positive_vector = np.asarray(positive[known], dtype=bool)
            blank_vector = np.asarray(blank[known], dtype=bool)
            positive_count = int(positive_vector.sum())
            blank_count = int(blank_vector.sum())
            query_box = _query_box(row)
            query_desc = hatch_texture.descriptor_for_query(gray, features, query_box, stride, source)
            self_stats = query_self_quantiles(
                model, features, query_desc, query_box, gray.shape, stride
            )
            self_rows.append({
                "id": row["id"],
                "document_id": row["document_id"],
                "kind": row["kind"],
                "known_pixels": int(known.sum()),
                "query_self": {key: self_stats[key] for key in QUANTILES},
                "query_self_cell_count": int(self_stats["count"]),
                "probability_file": str((probability_dir / f"{row['id']}.npz").name),
            })
            for threshold in DEFAULT_THRESHOLDS:
                baseline_records[str(threshold)].append(
                    _metric_record(
                        row, domain_meta, probabilities, positive_vector, blank_vector,
                        positive_count, blank_count, threshold
                    )
                )
            for quantile in QUANTILES:
                for scale in SCALES:
                    for offset in OFFSETS:
                        key = _config_key(quantile, scale, offset)
                        threshold = threshold_from_query_self(self_stats, quantile, scale, offset)
                        calibration_records[key].append(
                            _metric_record(
                                row, domain_meta, probabilities, positive_vector, blank_vector,
                                positive_count, blank_count, threshold
                            )
                        )
            print(
                json.dumps({"id": row["id"], "seconds": round(time.monotonic() - query_started, 4)}),
                flush=True,
            )
    return self_rows, baseline_records, calibration_records


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_manifest(args.manifest, require_labels=True)
    validate_split_safety(args.manifest, args.data_root, rows)
    model, metadata = load_texture_model(args.model)
    verify_model_metadata(metadata)
    started = time.monotonic()
    self_rows, baseline_records, calibration_records = run_calibration(
        rows, args.data_root, model, metadata, args.probability_dir
    )
    elapsed_seconds = time.monotonic() - started
    print(json.dumps({"elapsed_seconds": round(elapsed_seconds, 4), "queries": len(rows)}), flush=True)
    hashes = _hashes(args.manifest, args.model, args.probability_dir)
    formula = "clip(scale * query_self_quantile + offset, 0.01, 0.99)"
    query_report = {
        "schema": "hatch-matching-challenge-texture-query-self/v1",
        "method": "learned-texture-query-self-calibration-v1",
        "partition": "training-development only",
        "model_selection_scope": "development only",
        "hashes": hashes,
        "queries": self_rows,
        "quantiles": list(QUANTILES),
        "threshold_formula": formula,
        "scales": list(SCALES),
        "offsets": list(OFFSETS),
        "whole_sheet_accuracy_certified": False,
        "inference_claim": None,
        "elapsed_seconds": elapsed_seconds,
    }
    (args.output_dir / "query-self.json").write_text(
        json.dumps(query_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    calibration_report = {
        "schema": "hatch-matching-challenge-texture-calibration/v1",
        "method": "learned-texture-query-self-calibration-v1",
        "partition": "training-development only",
        "model_selection_scope": "development only",
        "hashes": hashes,
        "threshold_formula": formula,
        "scales": list(SCALES),
        "offsets": list(OFFSETS),
        "quantiles": list(QUANTILES),
        "baseline": {key: aggregate(value) for key, value in baseline_records.items()},
        "calibrated": {
            key: {
                **_decode_key(key),
                "summary": aggregate(records),
            }
            for key, records in calibration_records.items()
        },
        "whole_sheet_accuracy_certified": False,
        "inference_claim": None,
        "elapsed_seconds": elapsed_seconds,
    }
    (args.output_dir / "calibration-sweep.json").write_text(
        json.dumps(calibration_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _query_box(row: dict[str, Any]) -> tuple[int, int, int, int]:
    return cast(tuple[int, int, int, int], tuple(row["query_box"]))


def _load_saved_probabilities(
    path: Path,
    positive: np.ndarray,
    known: np.ndarray,
    blank: np.ndarray,
    row: dict[str, Any],
) -> np.ndarray:
    if not path.is_file():
        raise EvaluationError(f"Missing saved probability artifact for {row['id']}: {path}")
    with np.load(path) as data:
        if set(("probability", "positive", "blank")) - set(data.files):
            raise EvaluationError(f"Saved probability artifact is incomplete: {path}")
        probabilities = np.asarray(data["probability"])
        saved_positive = np.asarray(data["positive"])
        saved_blank = np.asarray(data["blank"])
    expected_positive = positive[known]
    expected_blank = blank[known]
    if probabilities.dtype != np.float32 or probabilities.ndim != 1:
        raise EvaluationError(f"Saved probabilities must be float32 vector: {path}")
    if saved_positive.dtype != np.bool_ or saved_blank.dtype != np.bool_:
        raise EvaluationError(f"Saved labels must be boolean vectors: {path}")
    if probabilities.shape != expected_positive.shape:
        raise EvaluationError(f"Saved probability length does not match reviewed pixels: {path}")
    if not np.array_equal(saved_positive, expected_positive) or not np.array_equal(saved_blank, expected_blank):
        raise EvaluationError(f"Saved positive/blank vectors do not match reviewed domains: {path}")
    if not bool(np.all(np.isfinite(probabilities))) or bool(np.any((probabilities < 0) | (probabilities > 1))):
        raise EvaluationError(f"Saved probabilities are outside [0, 1]: {path}")
    return probabilities


def _metric_record(
    row: dict[str, Any], metadata: dict[str, Any], probabilities: np.ndarray,
    positive: np.ndarray, blank: np.ndarray, positive_count: int, blank_count: int,
    threshold: float,
) -> dict[str, Any]:
    selected = probabilities >= threshold
    tp = int(np.count_nonzero(selected & positive))
    fp = int(np.count_nonzero(selected & ~positive))
    fn = positive_count - tp
    tn = int(len(positive) - positive_count - fp)
    return {
        "id": row["id"], "document_id": row["document_id"], "kind": row["kind"],
        **metadata,
        "metrics": metrics_from_counts(
            tp, fp, fn, tn, int(np.count_nonzero(selected & blank)), blank_count
        ),
    }


def _calibrated_threshold(value: float, scale: float, offset: float) -> float:
    """Compatibility wrapper for callers of the former local helper."""
    return threshold_from_query_self({"median": value}, "median", scale, offset)


def _config_key(quantile: str, scale: float, offset: float) -> str:
    return f"{quantile}|scale={scale:g}|offset={offset:g}"


def _decode_key(key: str) -> dict[str, Any]:
    quantile, scale, offset = key.split("|")
    return {"quantile": quantile, "scale": float(scale.split("=")[1]), "offset": float(offset.split("=")[1])}


def _hashes(manifest: Path, model: Path, probability_dir: Path) -> dict[str, str]:
    digest = hashlib.sha256()
    files = sorted(probability_dir.glob("*.npz"))
    if not files:
        raise EvaluationError(f"No .npz probability artifacts found: {probability_dir}")
    for path in files:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return {
        "manifest_sha256": _sha256(manifest),
        "model_sha256": _sha256(model),
        "probability_dir_sha256": digest.hexdigest(),
        "implementation_sha256": _sha256(Path(__file__)),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
