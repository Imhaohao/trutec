"""Validation-informed sparse diagnostics for the frozen native pipeline.

This module deliberately runs after the unchanged native validation evaluation.
The public validation labels and official result influence the diagnostics and
candidate ranking produced here.  They never enter texture descriptors, query
descriptors, query-self calibration, CNN inputs, or the sparse model scorers.

The sparse arrays contain one row-major entry for every official reviewed
coordinate.  They are a diagnostic representation of full native inference;
they are not submission masks and cannot certify whole-sheet accuracy.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, cast

import numpy as np

from evaluate import (
    EvaluationError,
    aggregate,
    evaluate,
    load_binary_png,
    load_manifest,
    metrics_from_counts,
    reviewed_domains,
)
from evaluate_hatchfinder_sweep import score_known_probabilities as score_cnn_sparse
from evaluate_texture_sweep import score_known_probabilities as score_texture_sparse
import hatch_texture
from inference import read_drawing, read_requests
from inference_hatchfinder import load_hatchfinder, read_rgb_drawing
from inference_texture import (
    query_self_quantiles,
    threshold_from_query_self,
    verify_model_metadata,
)
from texture_learning import load_texture_model


EXPECTED_VALIDATION_QUERIES = 57
DEFAULT_TILE_SIZE = 896
DEFAULT_OVERLAP = 128
DEFAULT_DEVICE = "mps"
FROZEN_QUANTILE = "q10"
FROZEN_SCALE = 0.7
FROZEN_OFFSET = -0.1
FROZEN_CNN_THRESHOLD = 0.4
FROZEN_CANDIDATE_ID = "and|q10_scale_0.7_offset_-0.1|cnn_0.4"
IOU_GATE = 0.75
RECALL_GATE = 0.95


@dataclass(frozen=True)
class TextureCandidate:
    """One global query-self calibration formula."""

    candidate_id: str
    scale: float
    offset: float
    formula: str


TEXTURE_CANDIDATES: tuple[TextureCandidate, ...] = (
    TextureCandidate(
        "q10_scale_0.7_offset_-0.1", 0.7, -0.1,
        "clip(query_self.q10 * 0.7 - 0.1, 0.01, 0.99)",
    ),
    TextureCandidate(
        "q10_scale_0.7_offset_-0.2", 0.7, -0.2,
        "clip(query_self.q10 * 0.7 - 0.2, 0.01, 0.99)",
    ),
    TextureCandidate(
        "q10_scale_0.5_offset_0", 0.5, 0.0,
        "clip(query_self.q10 * 0.5 + 0, 0.01, 0.99)",
    ),
    TextureCandidate(
        "q10_scale_0.3_offset_-0.2", 0.3, -0.2,
        "clip(query_self.q10 * 0.3 - 0.2, 0.01, 0.99)",
    ),
)
CNN_THRESHOLDS: tuple[float, ...] = (0.01, 0.05, 0.1, 0.2, 0.3, 0.4)

PUBLIC_VALIDATION_DISCLOSURE = {
    "labels_influenced_diagnostics": True,
    "results_influenced_selection": True,
    "selection_scope": "public validation labels and official native results",
    "selection_warning": (
        "This report is validation-informed and must not be presented as an "
        "unbiased estimate on an unseen partition."
    ),
}


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for the bounded diagnostic."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--official-metrics", required=True, type=Path)
    parser.add_argument("--predictions-dir", required=True, type=Path)
    parser.add_argument("--texture-model", required=True, type=Path)
    parser.add_argument("--cnn-model", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", choices=("mps", "cpu"), default=DEFAULT_DEVICE)
    parser.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments without starting model inference."""
    return build_parser().parse_args(None if argv is None else list(argv))


def _read_json_object(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise EvaluationError(f"{name} is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"{name} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise EvaluationError(f"{name} must be a JSON object: {path}")
    return payload


def _row_ids(rows: Sequence[Mapping[str, Any]], name: str) -> list[str]:
    ids = [row.get("id") for row in rows]
    if any(not isinstance(identifier, str) or not identifier for identifier in ids):
        raise EvaluationError(f"{name} contains a row without a valid id")
    if len(set(ids)) != len(ids):
        raise EvaluationError(f"{name} contains duplicate ids")
    return [str(identifier) for identifier in ids]


def load_official_metrics(
    path: Path,
    validation_rows: Sequence[Mapping[str, Any]],
    *,
    expected_count: int = EXPECTED_VALIDATION_QUERIES,
) -> dict[str, Any]:
    """Load and structurally verify the native official metrics report.

    This function checks that the report covers exactly the validation IDs and
    that each example exposes an official reviewed-pixel count.  When the
    prediction directory is available, :func:`verify_official_metrics` also
    re-runs the unchanged evaluator to detect edits to this JSON file.
    """
    report = _read_json_object(path, "official metrics report")
    expected_ids = _row_ids(validation_rows, "validation manifest")
    if len(expected_ids) != expected_count:
        raise EvaluationError(
            f"validation manifest must contain {expected_count} queries, got {len(expected_ids)}"
        )
    if report.get("schema") != "hatch-matching-challenge-metrics/v1":
        raise EvaluationError("official metrics report has the wrong schema")
    if report.get("whole_sheet_accuracy_certified") is not False:
        raise EvaluationError("official metrics must be native reviewed-pixel metrics")
    examples = report.get("examples")
    if not isinstance(examples, list) or len(examples) != expected_count:
        raise EvaluationError(
            f"official metrics must contain exactly {expected_count} examples"
        )
    report_ids = _row_ids(examples, "official metrics examples")
    if set(report_ids) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(report_ids))
        extra = sorted(set(report_ids) - set(expected_ids))
        raise EvaluationError(f"official metrics IDs mismatch; missing={missing}, extra={extra}")
    by_id = {str(example["id"]): example for example in examples}
    for row in validation_rows:
        example = by_id[str(row["id"])]
        _validate_official_example(example, row)
    summary = report.get("summary")
    if not isinstance(summary, dict) or summary.get("examples") != expected_count:
        raise EvaluationError("official metrics summary does not cover all validation queries")
    return report


def _validate_official_example(example: Mapping[str, Any], row: Mapping[str, Any]) -> None:
    """Validate the fields needed to align official labels and sparse outputs."""
    if example.get("document_id") != row.get("document_id"):
        raise EvaluationError(f"official metrics document mismatch for {row['id']}")
    metrics = example.get("metrics")
    if not isinstance(metrics, dict):
        raise EvaluationError(f"official metrics missing metrics for {row['id']}")
    reviewed = metrics.get("reviewed_pixels")
    if type(reviewed) is not int or reviewed <= 0:
        raise EvaluationError(f"official metrics reviewed-pixel count is invalid for {row['id']}")
    for key in ("tp", "fp", "fn", "tn"):
        if type(metrics.get(key)) is not int or metrics[key] < 0:
            raise EvaluationError(f"official metrics count {key} is invalid for {row['id']}")


def verify_official_metrics(
    report_path: Path,
    manifest_path: Path,
    data_root: Path,
    predictions_dir: Path,
    validation_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Require an unedited report by comparing it with native evaluator output."""
    report = load_official_metrics(report_path, validation_rows)
    try:
        evaluated = evaluate(manifest_path, data_root, predictions_dir)
    except (EvaluationError, OSError, ValueError) as exc:
        raise EvaluationError("native official evaluation could not be verified") from exc
    if evaluated != report:
        raise EvaluationError(
            "official-metrics.json differs from unchanged native evaluate.py output"
        )
    return report


def validate_input_alignment(
    validation_rows: Sequence[Mapping[str, Any]], input_rows: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Ensure label-free full-native inputs align exactly with validation rows."""
    validation_ids = _row_ids(validation_rows, "validation manifest")
    input_ids = _row_ids(input_rows, "label-free inputs")
    if set(validation_ids) != set(input_ids):
        raise EvaluationError("label-free inputs and validation manifest IDs mismatch")
    input_by_id = {str(row["id"]): dict(row) for row in input_rows}
    for row in validation_rows:
        request = input_by_id[str(row["id"])]
        for key in ("image", "width", "height", "query_box", "context_boxes"):
            if request.get(key) != row.get(key):
                raise EvaluationError(f"label-free input mismatch for {row['id']}: {key}")
    return input_by_id


def _reviewed_vectors(
    row: Mapping[str, Any], data_root: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    positive, known, blank, metadata = reviewed_domains(dict(row), data_root)
    y, x = np.nonzero(known)
    coordinates = np.stack((y, x), axis=1).astype(np.int32, copy=False)
    return coordinates, np.asarray(positive[known], dtype=bool), np.asarray(
        blank[known], dtype=bool
    ), known, metadata


def _validate_sparse_vectors(
    row: Mapping[str, Any],
    coordinates: np.ndarray,
    texture_probability: np.ndarray,
    cnn_probability: np.ndarray,
    positive: np.ndarray,
    blank: np.ndarray,
) -> None:
    """Validate row-major coordinate/probability/label alignment before saving."""
    expected = len(coordinates)
    arrays = (texture_probability, cnn_probability, positive, blank)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise EvaluationError("sparse coordinates must have shape (N, 2)")
    if any(len(array) != expected for array in arrays):
        raise EvaluationError("sparse probabilities and labels must share one length")
    height, width = int(row["height"]), int(row["width"])
    if expected and (
        np.any(coordinates[:, 0] < 0)
        or np.any(coordinates[:, 0] >= height)
        or np.any(coordinates[:, 1] < 0)
        or np.any(coordinates[:, 1] >= width)
    ):
        raise EvaluationError("sparse coordinate lies outside the native image")
    linear = coordinates[:, 0].astype(np.int64) * width + coordinates[:, 1]
    if len(linear) and (np.any(linear[1:] <= linear[:-1])):
        raise EvaluationError("sparse coordinates must be unique row-major reviewed pixels")
    for name, values in (("texture", texture_probability), ("CNN", cnn_probability)):
        if not np.issubdtype(np.asarray(values).dtype, np.floating):
            raise EvaluationError(f"{name} sparse probabilities must be floating point")
        if not bool(np.all(np.isfinite(values))) or bool(np.any((values < 0) | (values > 1))):
            raise EvaluationError(f"{name} sparse probabilities are invalid")


def save_sparse_outputs(
    destination: Path,
    row: Mapping[str, Any],
    coordinates: np.ndarray,
    texture_probability: np.ndarray,
    cnn_probability: np.ndarray,
    positive: np.ndarray,
    blank: np.ndarray,
    metadata: Mapping[str, Any],
    official_reviewed_pixels: int,
) -> Path:
    """Atomically save sparse native probabilities aligned to reviewed labels."""
    _validate_sparse_vectors(
        row, coordinates, texture_probability, cnn_probability, positive, blank
    )
    if len(coordinates) != official_reviewed_pixels:
        raise EvaluationError(
            f"computed reviewed pixels for {row['id']} do not match official count"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".partial.npz")
    np.savez_compressed(
        temporary,
        y=np.asarray(coordinates[:, 0], dtype=np.int32),
        x=np.asarray(coordinates[:, 1], dtype=np.int32),
        texture_probability=np.asarray(texture_probability, dtype=np.float32),
        cnn_probability=np.asarray(cnn_probability, dtype=np.float32),
        positive=np.asarray(positive, dtype=bool),
        blank=np.asarray(blank, dtype=bool),
        id=np.asarray(str(row["id"])),
        document_id=np.asarray(str(row["document_id"])),
        kind=np.asarray(str(row["kind"])),
        height=np.asarray(int(row["height"]), dtype=np.int32),
        width=np.asarray(int(row["width"]), dtype=np.int32),
        official_reviewed_pixels=np.asarray(official_reviewed_pixels, dtype=np.int64),
        original_known_pixels=np.asarray(int(metadata["original_known_pixels"]), dtype=np.int64),
        excluded_known_pixels=np.asarray(int(metadata["excluded_known_pixels"]), dtype=np.int64),
        unscored_pixels=np.asarray(int(metadata["unscored_pixels"]), dtype=np.int64),
    )
    temporary.replace(destination)
    return destination


def load_sparse_outputs(path: Path) -> dict[str, np.ndarray]:
    """Load and minimally validate one sparse output artifact."""
    if not path.is_file():
        raise EvaluationError(f"sparse output is missing: {path}")
    try:
        with np.load(path, allow_pickle=False) as data:
            required = {
                "y", "x", "texture_probability", "cnn_probability", "positive", "blank",
            }
            if not required.issubset(data.files):
                raise EvaluationError("sparse output is missing required arrays")
            return {key: np.asarray(data[key]) for key in data.files}
    except (OSError, ValueError) as exc:
        raise EvaluationError(f"cannot read sparse output: {path}") from exc


def _candidate_thresholds(query_self: Mapping[str, Any]) -> dict[str, float]:
    return {
        candidate.candidate_id: threshold_from_query_self(
            query_self, FROZEN_QUANTILE, candidate.scale, candidate.offset
        )
        for candidate in TEXTURE_CANDIDATES
    }


def _score_sparse_metrics(
    row: Mapping[str, Any],
    selected: np.ndarray,
    positive: np.ndarray,
    blank: np.ndarray,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    if len(selected) != len(positive) or len(selected) != len(blank):
        raise EvaluationError(f"candidate alignment mismatch for {row['id']}")
    positive_count = int(np.count_nonzero(positive))
    tp = int(np.count_nonzero(selected & positive))
    fp = int(np.count_nonzero(selected & ~positive))
    fn = positive_count - tp
    tn = int(len(selected) - positive_count - fp)
    metrics = metrics_from_counts(
        tp, fp, fn, tn, int(np.count_nonzero(selected & blank)), int(blank.sum())
    )
    return {
        "id": row["id"],
        "document_id": row["document_id"],
        "kind": row["kind"],
        **dict(metadata),
        "metrics": metrics,
    }


def compare_frozen_candidate(
    sparse_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    official_report: Mapping[str, Any],
    predictions_dir: Path,
    *,
    require_all: bool = True,
) -> dict[str, Any]:
    """Compare frozen sparse decisions to every official native PNG coordinate."""
    expected = {str(row["id"]): row for row in validation_rows}
    official = {
        str(example["id"]): example for example in official_report["examples"]
    }
    mismatches: dict[str, int] = {}
    official_counts: dict[str, int] = {}
    frozen_records: list[dict[str, Any]] = []
    for sparse in sparse_rows:
        identifier = str(sparse["id"])
        if identifier not in expected or identifier not in official:
            raise EvaluationError(f"frozen comparison has unknown query id {identifier}")
        row = expected[identifier]
        coords = np.asarray(sparse["coordinates"])
        selected = np.asarray(sparse["frozen_selected"], dtype=bool)
        if len(coords) != len(selected):
            raise EvaluationError(f"frozen sparse decision length mismatch for {identifier}")
        native = load_binary_png(
            predictions_dir / f"{identifier}.png", int(row["width"]), int(row["height"])
        )
        native_selected = native[coords[:, 0], coords[:, 1]]
        difference = int(np.count_nonzero(selected != native_selected))
        if difference:
            mismatches[identifier] = difference
        official_counts[identifier] = int(official[identifier]["metrics"]["reviewed_pixels"])
        if len(coords) != official_counts[identifier]:
            raise EvaluationError(f"official reviewed-pixel count mismatch for {identifier}")
        if {"row", "positive", "blank", "metadata"}.issubset(sparse):
            frozen_records.append(
                _score_sparse_metrics(
                    sparse["row"], selected, sparse["positive"], sparse["blank"], sparse["metadata"]
                )
            )
    expected_set = set(expected)
    seen = {str(row["id"]) for row in sparse_rows}
    if require_all and seen != expected_set:
        raise EvaluationError("frozen comparison did not cover all validation IDs")
    summary_matches: bool | None = None
    if frozen_records and (require_all or seen == expected_set):
        if len(frozen_records) != len(sparse_rows):
            raise EvaluationError("frozen comparison has incomplete label alignment")
        summary_matches = aggregate(frozen_records) == official_report.get("summary")
        if not summary_matches:
            mismatches["__aggregate_summary__"] = 1
    matches = not mismatches
    return {
        "verified": matches,
        "candidate": (
            "clip(query_self.q10 * 0.7 - 0.1, 0.01, 0.99) "
            "AND cnn_probability >= 0.4"
        ),
        "queries": len(sparse_rows),
        "mismatches": mismatches,
        "summary_matches_official": summary_matches,
        "official_reviewed_pixels": official_counts,
    }


def _candidate_id(texture: TextureCandidate, cnn_threshold: float) -> str:
    return f"and|{texture.candidate_id}|cnn_{cnn_threshold:g}"


def _texture_only_id(texture: TextureCandidate) -> str:
    return f"texture-only|{texture.candidate_id}"


def _cnn_only_id(cnn_threshold: float) -> str:
    return f"cnn-only|{cnn_threshold:g}"


def _record_candidate(
    records: dict[str, list[dict[str, Any]]],
    formulas: dict[str, dict[str, Any]],
    identifier: str,
    formula_details: dict[str, Any],
    row: Mapping[str, Any],
    selected: np.ndarray,
    positive: np.ndarray,
    blank: np.ndarray,
    metadata: Mapping[str, Any],
) -> None:
    records[identifier].append(_score_sparse_metrics(row, selected, positive, blank, metadata))
    formulas[identifier] = formula_details


def _build_candidate_summaries(
    sparse_rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    formulas: dict[str, dict[str, Any]] = {}
    for sparse in sparse_rows:
        row = sparse["row"]
        texture = np.asarray(sparse["texture_probability"], dtype=np.float32)
        cnn = np.asarray(sparse["cnn_probability"], dtype=np.float32)
        positive = np.asarray(sparse["positive"], dtype=bool)
        blank = np.asarray(sparse["blank"], dtype=bool)
        thresholds = sparse["texture_thresholds"]
        for texture_candidate in TEXTURE_CANDIDATES:
            texture_id = texture_candidate.candidate_id
            texture_selected = texture >= float(thresholds[texture_id])
            _record_candidate(
                records,
                formulas,
                _texture_only_id(texture_candidate),
                {
                    "formula": texture_candidate.formula,
                    "texture_formula": texture_candidate.formula,
                    "cnn_threshold": None,
                    "operator": "texture-only",
                },
                row,
                texture_selected,
                positive,
                blank,
                sparse["metadata"],
            )
            for cnn_threshold in CNN_THRESHOLDS:
                identifier = _candidate_id(texture_candidate, cnn_threshold)
                _record_candidate(
                    records,
                    formulas,
                    identifier,
                    {
                        "formula": (
                            f"{texture_candidate.formula} AND "
                            f"cnn_probability >= {cnn_threshold:g}"
                        ),
                        "texture_formula": texture_candidate.formula,
                        "cnn_threshold": cnn_threshold,
                        "operator": "AND",
                    },
                    row,
                    texture_selected & (cnn >= cnn_threshold),
                    positive,
                    blank,
                    sparse["metadata"],
                )
        for cnn_threshold in CNN_THRESHOLDS:
            _record_candidate(
                records,
                formulas,
                _cnn_only_id(cnn_threshold),
                {
                    "formula": f"cnn_probability >= {cnn_threshold:g}",
                    "texture_formula": None,
                    "cnn_threshold": cnn_threshold,
                    "operator": "cnn-only",
                },
                row,
                cnn >= cnn_threshold,
                positive,
                blank,
                sparse["metadata"],
            )
    summaries = {}
    for identifier, candidate_records in records.items():
        summaries[identifier] = {
            **formulas[identifier],
            "summary": aggregate(candidate_records),
            "per_query": candidate_records,
            "reviewed_pixels": sum(
                int(record["metrics"]["reviewed_pixels"]) for record in candidate_records
            ),
        }
    return summaries


def _gate_status(
    candidate_id: str, summary: Mapping[str, Any], verification: Mapping[str, Any]
) -> dict[str, Any]:
    """Report a gate only after both native checks have passed."""
    metrics = summary["summary"]
    native_verified = bool(
        verification.get("native_final_masks_verified")
        and candidate_id == verification.get("native_verified_candidate_id")
    )
    official_verified = bool(
        verification.get("official_evaluation_unchanged")
        and candidate_id == verification.get("official_verified_candidate_id")
    )
    meets_metrics = bool(
        metrics.get("document_macro_iou") is not None
        and metrics.get("document_macro_iou") >= IOU_GATE
        and metrics.get("global_recall") is not None
        and metrics.get("global_recall") >= RECALL_GATE
    )
    return {
        "document_macro_iou_gate": IOU_GATE,
        "global_recall_gate": RECALL_GATE,
        "native_final_masks_verified": native_verified,
        "official_evaluation_unchanged": official_verified,
        "metrics_meet_gates": meets_metrics,
        "gate_met": bool(native_verified and official_verified and meets_metrics),
    }


def _score_validation_rows(
    validation_rows: Sequence[Mapping[str, Any]],
    input_by_id: Mapping[str, Mapping[str, Any]],
    data_root: Path,
    official_by_id: Mapping[str, Mapping[str, Any]],
    texture_model: Any,
    texture_metadata: Mapping[str, Any],
    cnn_model: Any,
    output_dir: Path,
    device: str,
    tile_size: int,
    overlap: int,
    texture_scorer: Callable[..., np.ndarray],
    cnn_scorer: Callable[..., np.ndarray],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in validation_rows:
        grouped[str(input_by_id[str(row["id"])] ["image"])].append(row)
    stride = int(texture_metadata.get("stride", 4))
    source = hatch_texture.descriptor_source_from_metadata(dict(texture_metadata))
    feature_set = hatch_texture.feature_set_from_metadata(dict(texture_metadata))
    sparse_rows: list[dict[str, Any]] = []
    probability_dir = output_dir / "probabilities"
    processed = 0
    for group in grouped.values():
        input_row = input_by_id[str(group[0]["id"])]
        gray = read_drawing(data_root, dict(input_row))
        rgb = read_rgb_drawing(data_root, dict(input_row))
        features = hatch_texture.texture_features(gray, stride=stride, feature_set=feature_set)
        for row in group:
            sparse_row = _score_one_row(
                row,
                input_by_id[str(row["id"])],
                data_root,
                official_by_id[str(row["id"])],
                gray,
                rgb,
                features,
                texture_model,
                source,
                feature_set,
                stride,
                cnn_model,
                device,
                tile_size,
                overlap,
                probability_dir,
                texture_scorer,
                cnn_scorer,
            )
            sparse_rows.append(sparse_row)
            processed += 1
            print(
                json.dumps(
                    {
                        "progress": "validation_sparse_query",
                        "id": row["id"],
                        "processed_queries": processed,
                        "total_queries": len(validation_rows),
                    }
                ),
                flush=True,
            )
        del features, rgb, gray
    return sparse_rows


def _score_one_row(
    row: Mapping[str, Any],
    input_row: Mapping[str, Any],
    data_root: Path,
    official: Mapping[str, Any],
    gray: np.ndarray,
    rgb: np.ndarray,
    features: np.ndarray,
    texture_model: Any,
    source: str,
    feature_set: str,
    stride: int,
    cnn_model: Any,
    device: str,
    tile_size: int,
    overlap: int,
    probability_dir: Path,
    texture_scorer: Callable[..., np.ndarray],
    cnn_scorer: Callable[..., np.ndarray],
) -> dict[str, Any]:
    if gray.shape != (int(input_row["height"]), int(input_row["width"])):
        raise EvaluationError(f"drawing dimensions mismatch for {row['id']}")
    coordinates, positive, blank, known, metadata = _reviewed_vectors(row, data_root)
    official_reviewed = int(official["metrics"]["reviewed_pixels"])
    if len(coordinates) != official_reviewed:
        raise EvaluationError(f"official reviewed-pixel count mismatch for {row['id']}")
    query_box = cast(
        tuple[int, int, int, int], tuple(int(value) for value in input_row["query_box"])
    )
    query_desc = hatch_texture.descriptor_for_query(
        gray, features, query_box, stride, source, feature_set=feature_set
    )
    query_self = query_self_quantiles(
        texture_model, features, query_desc, query_box, gray.shape, stride
    )
    texture_probability = texture_scorer(
        texture_model, features, query_desc, known, gray.shape[0], gray.shape[1], stride
    )
    cnn_probability = cnn_scorer(
        cnn_model, rgb, query_box, known, tile_size, overlap, device
    )
    texture_thresholds = _candidate_thresholds(query_self)
    _validate_sparse_vectors(
        row, coordinates, texture_probability, cnn_probability, positive, blank
    )
    frozen_selected = (
        texture_probability >= texture_thresholds[TEXTURE_CANDIDATES[0].candidate_id]
    ) & (cnn_probability >= FROZEN_CNN_THRESHOLD)
    destination = probability_dir / f"{row['id']}.npz"
    save_sparse_outputs(
        destination,
        row,
        coordinates,
        texture_probability,
        cnn_probability,
        positive,
        blank,
        metadata,
        official_reviewed,
    )
    return {
        "id": str(row["id"]),
        "row": dict(row),
        "coordinates": coordinates,
        "texture_probability": np.asarray(texture_probability, dtype=np.float32),
        "cnn_probability": np.asarray(cnn_probability, dtype=np.float32),
        "positive": positive,
        "blank": blank,
        "metadata": metadata,
        "query_self": {key: float(query_self[key]) for key in ("median", "q10", "q25", "q75")},
        "query_self_cell_count": int(query_self["count"]),
        "texture_thresholds": texture_thresholds,
        "frozen_selected": np.asarray(frozen_selected, dtype=bool),
        "probability_path": str(destination),
    }


def run_analysis(
    validation_rows: Sequence[Mapping[str, Any]],
    input_rows: Sequence[Mapping[str, Any]],
    data_root: Path,
    official_report: Mapping[str, Any],
    predictions_dir: Path,
    texture_model: Any,
    texture_metadata: Mapping[str, Any],
    cnn_model: Any,
    output_dir: Path,
    *,
    device: str = DEFAULT_DEVICE,
    tile_size: int = DEFAULT_TILE_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    official_evaluation_verified: bool = False,
    texture_scorer: Callable[..., np.ndarray] = score_texture_sparse,
    cnn_scorer: Callable[..., np.ndarray] = score_cnn_sparse,
) -> dict[str, Any]:
    """Run sparse validation diagnostics after official-report verification."""
    input_by_id = validate_input_alignment(validation_rows, input_rows)
    official_by_id = {str(example["id"]): example for example in official_report["examples"]}
    sparse_rows = _score_validation_rows(
        validation_rows,
        input_by_id,
        data_root,
        official_by_id,
        texture_model,
        texture_metadata,
        cnn_model,
        output_dir,
        device,
        tile_size,
        overlap,
        texture_scorer,
        cnn_scorer,
    )
    frozen = compare_frozen_candidate(
        sparse_rows, validation_rows, official_report, predictions_dir
    )
    summaries = _build_candidate_summaries(sparse_rows)
    verification = {
        "native_final_masks_verified": bool(frozen["verified"]),
        "native_verified_candidate_id": (
            FROZEN_CANDIDATE_ID if frozen["verified"] else None
        ),
        "official_evaluation_unchanged": bool(official_evaluation_verified),
        "official_verified_candidate_id": (
            FROZEN_CANDIDATE_ID if official_evaluation_verified else None
        ),
        "native_verification_scope": "frozen candidate only",
    }
    for identifier, candidate in summaries.items():
        candidate["gate_status"] = _gate_status(identifier, candidate, verification)
    best_id = _select_recall_eligible_candidate(summaries)
    reviewed_counts = {
        str(row["id"]): int(official_by_id[str(row["id"])]["metrics"]["reviewed_pixels"])
        for row in validation_rows
    }
    return {
        "schema": "hatch-matching-validation-analysis/v1",
        "validation_disclosure": PUBLIC_VALIDATION_DISCLOSURE,
        "queries": len(validation_rows),
        "official_reviewed_pixels": reviewed_counts,
        "candidate_formulas": [
            {
                "candidate_id": identifier,
                "formula": candidate["formula"],
                "texture_formula": candidate["texture_formula"],
                "cnn_threshold": candidate["cnn_threshold"],
                "operator": candidate["operator"],
            }
            for identifier, candidate in summaries.items()
        ],
        "candidates": summaries,
        "selected_candidate": best_id,
        "selection_rule": (
            "Highest document-macro IoU among candidates with global recall >= 0.95; "
            "ties prefer fewer reviewed false positives."
        ),
        "frozen_equivalence": frozen,
        "verification": verification,
        "official_summary": official_report["summary"],
        "whole_sheet_accuracy_certified": False,
    }


def _select_recall_eligible_candidate(
    summaries: Mapping[str, Mapping[str, Any]],
) -> str | None:
    """Choose the best validation-informed candidate only after the recall filter."""
    eligible = [
        identifier
        for identifier, candidate in summaries.items()
        if candidate["summary"].get("global_recall") is not None
        and candidate["summary"]["global_recall"] >= RECALL_GATE
    ]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda identifier: (
            summaries[identifier]["summary"].get("document_macro_iou") or float("-inf"),
            -summaries[identifier]["summary"].get("fp", 0),
        ),
    )


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    """Verify official native evaluation, then run public-label diagnostics."""
    args = parse_args(argv)
    validation_rows = load_manifest(args.manifest, require_labels=True)
    input_rows = read_requests(args.inputs)
    official = verify_official_metrics(
        args.official_metrics,
        args.manifest,
        args.data_root,
        args.predictions_dir,
        validation_rows,
    )
    texture_model, texture_metadata = load_texture_model(args.texture_model)
    verify_model_metadata(texture_metadata)
    cnn_model = load_hatchfinder(args.cnn_model, args.device)
    report = run_analysis(
        validation_rows,
        input_rows,
        args.data_root,
        official,
        args.predictions_dir,
        texture_model,
        texture_metadata,
        cnn_model,
        args.output_dir,
        device=args.device,
        tile_size=args.tile_size,
        overlap=args.overlap,
        official_evaluation_verified=True,
    )
    report["official_metrics_sha256"] = hashlib.sha256(
        args.official_metrics.read_bytes()
    ).hexdigest()
    _write_report(args.output_dir / "analysis.json", report)
    print(
        json.dumps(
            {"selected_candidate": report["selected_candidate"], "queries": len(validation_rows)}
        )
    )


if __name__ == "__main__":
    main()
