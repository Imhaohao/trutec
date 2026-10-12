"""Compare sparse development probability caches on the reviewed native domain."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from evaluate import EvaluationError, aggregate, load_manifest, metrics_from_counts, reviewed_domains
from evaluate_texture_sweep import validate_split_safety

EXPECTED_QUERIES = 55
EXPECTED_DOCUMENTS = 12
THRESHOLDS = np.linspace(0.0, 1.0, 101, dtype=np.float64)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--cnn-cache", required=True, type=Path)
    parser.add_argument(
        "--texture-cache", action="append", type=_named_path, required=True,
        metavar="NAME=DIR", help="Sparse probability cache for one texture model; repeatable.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-queries", type=int, default=EXPECTED_QUERIES)
    parser.add_argument("--expected-documents", type=int, default=EXPECTED_DOCUMENTS)
    parser.add_argument("--fit-manifest", type=Path)
    parser.add_argument(
        "--training-metadata", action="append", type=_named_path, default=[], metavar="NAME=JSON",
    )
    return parser


def _named_path(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    return name, Path(path)


def _validate_manifest(
    manifest_path: Path,
    data_root: Path,
    rows: list[dict[str, Any]],
    expected_queries: int,
    expected_documents: int,
) -> None:
    validate_split_safety(manifest_path, data_root, rows)
    if len(rows) != expected_queries:
        raise EvaluationError(f"development manifest must contain {expected_queries} queries")
    if len({row["document_id"] for row in rows}) != expected_documents:
        raise EvaluationError(f"development manifest must contain {expected_documents} documents")
    if any(row.get("kind") != "real" for row in rows):
        raise EvaluationError("development-real manifest must contain only real queries")


def _manifest_rows(path: Path) -> list[dict[str, Any]]:
    rows = load_manifest(path)
    return rows


def _validate_fit_metadata(
    fit_manifest: Path | None,
    development_rows: Sequence[Mapping[str, Any]],
    model_metadata: Mapping[str, Path],
    data_root: Path,
) -> None:
    if not model_metadata:
        return
    if fit_manifest is None or not fit_manifest.is_file():
        raise EvaluationError("--fit-manifest is required when --training-metadata is supplied")
    fit_rows = load_manifest(fit_manifest)
    validate_split_safety(fit_manifest, data_root, fit_rows)
    fit_ids = {str(row["id"]) for row in fit_rows}
    fit_documents = {str(row["document_id"]) for row in fit_rows}
    dev_ids = {str(row["id"]) for row in development_rows}
    dev_documents = {str(row["document_id"]) for row in development_rows}
    if fit_ids & dev_ids or fit_documents & dev_documents:
        raise EvaluationError("fit and development manifests share query or document IDs")
    fit_query_documents = {str(row["id"]): str(row["document_id"]) for row in fit_rows}
    for name, path in model_metadata.items():
        payload = _read_json(path, f"{name} training metadata")
        if _has_trainer_metadata(payload):
            if {"training_ids", "training_documents"} & payload.keys():
                raise EvaluationError(f"{name} training metadata mixes canonical and trainer schemas")
            _validate_trainer_metadata(
                payload, name, fit_manifest, fit_ids, fit_documents,
                fit_query_documents, dev_ids, dev_documents,
            )
        else:
            _validate_canonical_metadata(payload, name, fit_ids, fit_documents, dev_ids, dev_documents)


def _has_trainer_metadata(payload: Mapping[str, Any]) -> bool:
    trainer_fields = {"selected_query_ids", "selected_document_ids", "fit_document_ids"}
    return bool(trainer_fields & payload.keys())


def _validate_canonical_metadata(
    payload: Mapping[str, Any], name: str, fit_ids: set[str], fit_documents: set[str],
    dev_ids: set[str], dev_documents: set[str],
) -> None:
    training_ids = _string_set(payload.get("training_ids"), f"{name} training_ids")
    training_documents = _string_set(
        payload.get("training_documents"), f"{name} training_documents"
    )
    if not training_ids <= fit_ids or not training_documents <= fit_documents:
        raise EvaluationError(f"{name} training metadata is outside the fit manifest")
    if training_ids & dev_ids or training_documents & dev_documents:
        raise EvaluationError(f"{name} training metadata overlaps development")


def _validate_trainer_metadata(
    payload: Mapping[str, Any], name: str, fit_manifest: Path,
    fit_ids: set[str], fit_documents: set[str], fit_query_documents: Mapping[str, str],
    dev_ids: set[str], dev_documents: set[str],
) -> None:
    required = {"selected_query_ids", "selected_document_ids", "fit_document_ids", "manifest_sha256"}
    if not required <= payload.keys():
        missing = sorted(required - payload.keys())
        raise EvaluationError(f"{name} trainer metadata is missing fields: {missing}")
    expected_hash = _sha256_file(fit_manifest)
    if payload["manifest_sha256"] != expected_hash:
        raise EvaluationError(f"{name} trainer manifest_sha256 does not match --fit-manifest")
    recorded_fit_documents = _string_set(payload["fit_document_ids"], f"{name} fit_document_ids")
    if recorded_fit_documents != fit_documents:
        raise EvaluationError(f"{name} trainer fit_document_ids do not match the fit manifest")
    query_ids = _string_list(payload["selected_query_ids"], f"{name} selected_query_ids")
    document_ids = _string_list(payload["selected_document_ids"], f"{name} selected_document_ids")
    if not query_ids or len(query_ids) != len(document_ids):
        raise EvaluationError(f"{name} trainer selected query/document IDs must be nonempty and aligned")
    if any(fit_query_documents.get(query_id) != document_id for query_id, document_id in zip(query_ids, document_ids)):
        raise EvaluationError(f"{name} trainer selected query/document pair is outside the fit manifest")
    if set(query_ids) & dev_ids or set(document_ids) & dev_documents:
        raise EvaluationError(f"{name} trainer metadata overlaps development")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"cannot read {name}: {path}") from exc
    if not isinstance(value, dict):
        raise EvaluationError(f"{name} must be a JSON object")
    return value


def _string_set(value: object, name: str) -> set[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise EvaluationError(f"{name} must be a list of strings")
    result = set(value)
    if len(result) != len(value):
        raise EvaluationError(f"{name} contains duplicates")
    return result


def _string_list(value: object, name: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise EvaluationError(f"{name} must be a list of nonempty strings")
    return value


def _expected_cache_ids(cache_dir: Path, ids: set[str], name: str) -> None:
    observed = {path.stem for path in cache_dir.glob("*.npz")}
    if observed != ids:
        missing, extra = sorted(ids - observed), sorted(observed - ids)
        raise EvaluationError(f"{name} cache IDs mismatch; missing={missing[:5]}, extra={extra[:5]}")


def _load_cache_array(
    path: Path,
    row: Mapping[str, Any],
    data_root: Path,
    probability_name: str,
    source: str,
) -> dict[str, np.ndarray]:
    positive_mask, known_mask, blank_mask, _ = reviewed_domains(row, data_root)
    expected_y, expected_x = np.nonzero(known_mask)
    expected_positive = np.asarray(positive_mask[known_mask], dtype=np.bool_)
    expected_blank = np.asarray(blank_mask[known_mask], dtype=np.bool_)
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: np.asarray(archive[key]) for key in archive.files}
    except (OSError, ValueError, EOFError) as exc:
        raise EvaluationError(f"cannot read {source} sparse cache: {path}") from exc
    probability = _probability_vector(arrays, probability_name, path, source)
    _validate_cache_labels(arrays, expected_positive, expected_blank, probability, path, source)
    _validate_optional_metadata(arrays, row, len(expected_y), path, source)
    _validate_optional_coordinates(arrays, expected_y, expected_x, path, source)
    return {
        "probability": probability,
        "positive": expected_positive,
        "blank": expected_blank,
    }


def _probability_vector(
    arrays: Mapping[str, np.ndarray], key: str, path: Path, source: str
) -> np.ndarray:
    if key not in arrays and "probability" in arrays:
        key = "probability"
    if key not in arrays:
        raise EvaluationError(f"{source} cache lacks {key}: {path}")
    probability = arrays[key]
    if probability.dtype != np.float32 or probability.ndim != 1:
        raise EvaluationError(f"{source} probabilities must be a float32 vector: {path}")
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise EvaluationError(f"{source} probabilities must be finite and within [0, 1]: {path}")
    return probability


def _validate_cache_labels(
    arrays: Mapping[str, np.ndarray],
    expected_positive: np.ndarray,
    expected_blank: np.ndarray,
    probability: np.ndarray,
    path: Path,
    source: str,
) -> None:
    for name, expected in (("positive", expected_positive), ("blank", expected_blank)):
        if name not in arrays:
            raise EvaluationError(f"{source} cache lacks {name} labels: {path}")
        actual = arrays[name]
        if actual.dtype != np.bool_ or actual.shape != expected.shape:
            raise EvaluationError(f"{source} {name} labels have invalid shape or dtype: {path}")
        if not np.array_equal(actual, expected):
            raise EvaluationError(f"{source} {name} labels do not match reviewed coordinates: {path}")
    if probability.shape != expected_positive.shape:
        raise EvaluationError(f"{source} probability length does not match known pixels: {path}")


def _validate_optional_metadata(
    arrays: Mapping[str, np.ndarray], row: Mapping[str, Any], count: int, path: Path, source: str
) -> None:
    expected = {
        "id": str(row["id"]),
        "document_id": str(row["document_id"]),
        "kind": str(row["kind"]),
        "width": int(row["width"]),
        "height": int(row["height"]),
        "official_reviewed_pixels": count,
    }
    for key, value in expected.items():
        if key in arrays and arrays[key].item() != value:
            raise EvaluationError(f"{source} {key} does not match manifest: {path}")


def _validate_optional_coordinates(
    arrays: Mapping[str, np.ndarray], y: np.ndarray, x: np.ndarray, path: Path, source: str
) -> None:
    has_y, has_x = "y" in arrays, "x" in arrays
    if has_y != has_x:
        raise EvaluationError(f"{source} coordinates must include both y and x: {path}")
    if not has_y:
        return
    actual_y, actual_x = arrays["y"], arrays["x"]
    if actual_y.dtype.kind not in "iu" or actual_x.dtype.kind not in "iu":
        raise EvaluationError(f"{source} coordinates must be integer arrays: {path}")
    if not np.array_equal(actual_y, y) or not np.array_equal(actual_x, x):
        raise EvaluationError(f"{source} coordinates do not match reviewed row-major pixels: {path}")


def load_model_cache(
    rows: Sequence[Mapping[str, Any]], data_root: Path, cache_dir: Path,
    probability_name: str, source: str,
) -> list[dict[str, Any]]:
    ids = {str(row["id"]) for row in rows}
    _expected_cache_ids(cache_dir, ids, source)
    records = []
    for row in rows:
        values = _load_cache_array(
            cache_dir / f"{row['id']}.npz", row, data_root, probability_name, source
        )
        records.append({**values, "id": row["id"], "document_id": row["document_id"]})
    return records


def _rank_values(values: np.ndarray) -> np.ndarray:
    # Match NumPy's array >= Python-float threshold behavior used by the sweeper.
    # NumPy compares float32 arrays against weak Python scalars at float32 precision.
    thresholds = THRESHOLDS.astype(values.dtype, copy=False)
    return np.searchsorted(thresholds, values, side="right")


def _suffix_counts(histogram: np.ndarray, dimensions: int) -> np.ndarray:
    reversed_histogram = histogram[(slice(None, None, -1),) * dimensions]
    cumulative = reversed_histogram
    for axis in range(dimensions):
        cumulative = np.cumsum(cumulative, axis=axis)
    return cumulative[(slice(None, None, -1),) * dimensions][(slice(1, None),) * dimensions]


def _threshold_counts(
    probability: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    ranks = _rank_values(probability)
    hist = np.bincount(ranks[labels], minlength=len(THRESHOLDS) + 1)
    selected = np.cumsum(hist[::-1])[::-1][1:]
    blank = np.zeros(len(selected), dtype=np.int64)
    return selected.astype(np.int64), blank


def _and_counts(
    texture: np.ndarray, cnn: np.ndarray, labels: np.ndarray, blank: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    texture_ranks, cnn_ranks = _rank_values(texture), _rank_values(cnn)
    shape = (len(THRESHOLDS) + 1, len(THRESHOLDS) + 1)
    positive_hist = np.zeros(shape, dtype=np.int64)
    negative_hist = np.zeros(shape, dtype=np.int64)
    blank_hist = np.zeros(shape, dtype=np.int64)
    np.add.at(positive_hist, (texture_ranks[labels], cnn_ranks[labels]), 1)
    np.add.at(negative_hist, (texture_ranks[~labels], cnn_ranks[~labels]), 1)
    np.add.at(blank_hist, (texture_ranks[blank], cnn_ranks[blank]), 1)
    return (
        _suffix_counts(positive_hist, 2),
        _suffix_counts(negative_hist, 2),
        _suffix_counts(blank_hist, 2),
    )


def _texture_counts(
    probability: np.ndarray, positive: np.ndarray, blank: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    pos, _ = _threshold_counts(probability, positive)
    bfp, _ = _threshold_counts(probability, blank)
    return pos, bfp


def _aggregate_selection(
    rows: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    tp_values: np.ndarray,
    fp_values: np.ndarray,
    blank_fp_values: np.ndarray,
) -> dict[str, Any]:
    metrics = []
    for row, record, tp, fp, blank_fp in zip(rows, records, tp_values, fp_values, blank_fp_values):
        positives = int(record["positive"].sum())
        negatives = len(record["positive"]) - positives
        counts = metrics_from_counts(
            int(tp), int(fp), positives - int(tp), negatives - int(fp),
            int(blank_fp), int(record["blank"].sum()),
        )
        metrics.append({
            "id": row["id"],
            "document_id": row["document_id"],
            "positive_only": positives > 0 and positives == len(record["positive"]),
            "metrics": counts,
        })
    return aggregate(metrics)


def _candidate(
    rows: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    tp: np.ndarray,
    fp: np.ndarray,
    blank_fp: np.ndarray,
    texture_index: int,
    cnn_index: int | None,
) -> dict[str, Any]:
    summary = _aggregate_selection(
        rows, records, tp[:, texture_index] if cnn_index is None else tp[:, texture_index, cnn_index],
        fp[:, texture_index] if cnn_index is None else fp[:, texture_index, cnn_index],
        blank_fp[:, texture_index] if cnn_index is None else blank_fp[:, texture_index, cnn_index],
    )
    return {
        "texture_threshold": float(THRESHOLDS[texture_index]),
        "cnn_threshold": None if cnn_index is None else float(THRESHOLDS[cnn_index]),
        "document_macro_iou": summary["document_macro_iou"],
        "global_recall": summary["global_recall"],
        "global_precision": summary["global_precision"],
        "tp": summary["tp"],
        "fp": summary["fp"],
        "tn": summary["tn"],
        "fn": summary["fn"],
        "false_discovery_rate": _ratio(summary["fp"], summary["tp"] + summary["fp"]),
        "false_positive_rate": _ratio(summary["fp"], summary["fp"] + summary["tn"]),
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _selection_report(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    high_recall = [item for item in candidates if (item["global_recall"] or 0) >= 0.98]
    high_precision = [item for item in candidates if (item["global_precision"] or 0) >= 0.95]
    low_fdr = [
        item for item in candidates
        if item["false_discovery_rate"] is not None
        and item["false_discovery_rate"] <= 0.05
    ]
    joint = [
        item for item in high_recall
        if (item["global_precision"] or 0) >= 0.95
        and (item["document_macro_iou"] or 0) >= 0.95
    ]
    all_four_gate = [
        item for item in joint
        if item["false_positive_rate"] is not None
        and item["false_positive_rate"] <= 0.05
    ]
    return {
        "precision_best_at_recall_ge_0_98": _best(high_recall, "global_precision"),
        "macro_iou_best_at_recall_ge_0_98": _best(high_recall, "document_macro_iou"),
        "macro_iou_best_at_precision_ge_0_95": _best(high_precision, "document_macro_iou"),
        "macro_iou_best_at_fdr_le_0_05": _best(low_fdr, "document_macro_iou"),
        "joint_gate_count": len(joint),
        "best_joint_gate": _best(joint, "document_macro_iou"),
        "all_four_gate_count": len(all_four_gate),
        "best_all_four_gate": _best(all_four_gate, "document_macro_iou"),
    }


def _best(candidates: Sequence[Mapping[str, Any]], objective: str) -> dict[str, Any] | None:
    if not candidates:
        return None
    ordered = sorted(
        candidates,
        key=lambda item: (
            float(item[objective] if item[objective] is not None else -1),
            float(item["document_macro_iou"] if item["document_macro_iou"] is not None else -1),
            float(item["global_recall"] if item["global_recall"] is not None else -1),
            float(item["global_precision"] if item["global_precision"] is not None else -1),
        ),
        reverse=True,
    )
    return dict(ordered[0])


def _precision_recall_frontier(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(
        candidates,
        key=lambda item: (
            float(item["global_recall"] or 0),
            float(item["global_precision"] or 0),
        ),
        reverse=True,
    )
    frontier: list[dict[str, Any]] = []
    best_precision = -1.0
    for candidate in ordered:
        precision = float(candidate["global_precision"] or 0)
        if precision > best_precision:
            frontier.append(dict(candidate))
            best_precision = precision
    return frontier


def _serialize_grid(
    rows: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    tp: np.ndarray,
    fp: np.ndarray,
    blank_fp: np.ndarray,
    fused: bool,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    matrices = {
        key: np.zeros(tp.shape[1:], dtype=np.float64)
        for key in ("document_macro_iou", "global_recall", "global_precision")
    }
    if fused:
        pairs: Sequence[tuple[int, int | None]] = [
            (t, c) for t in range(101) for c in range(101)
        ]
    else:
        pairs = [(t, None) for t in range(101)]
    for texture_index, cnn_index in pairs:
        item = _candidate(rows, records, tp, fp, blank_fp, texture_index, cnn_index)
        candidates.append(item)
        coordinates = (texture_index,) if cnn_index is None else (texture_index, cnn_index)
        for key in matrices:
            value = item[key]
            matrices[key][coordinates] = np.nan if value is None else value
    candidate_payloads = [
        {key: item[key] for key in (
            "texture_threshold", "cnn_threshold", "document_macro_iou",
            "global_recall", "global_precision", "tp", "fp", "tn", "fn",
            "false_discovery_rate", "false_positive_rate",
        )}
        for item in candidates
    ]
    return {
        "texture_thresholds": THRESHOLDS.tolist(),
        "cnn_thresholds": THRESHOLDS.tolist() if fused else None,
        "document_macro_iou": _json_matrix(matrices["document_macro_iou"]),
        "global_recall": _json_matrix(matrices["global_recall"]),
        "global_precision": _json_matrix(matrices["global_precision"]),
        "selections": _selection_report(candidates),
        "precision_recall_frontier": (
            _precision_recall_frontier(candidate_payloads) if fused else None
        ),
    }


def _json_matrix(matrix: np.ndarray) -> list[Any]:
    values = [
        [None if not np.isfinite(value) else float(value) for value in row]
        for row in np.atleast_2d(matrix)
    ]
    return values if matrix.ndim > 1 else values[0]


def analyze_model(
    rows: Sequence[Mapping[str, Any]],
    texture_records: Sequence[Mapping[str, Any]],
    cnn_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(rows) != len(texture_records) or len(rows) != len(cnn_records):
        raise EvaluationError("probability cache lengths do not match the development manifest")
    if any(record["id"] != row["id"] for row, record in zip(rows, texture_records)):
        raise EvaluationError("texture cache order does not match the development manifest")
    if any(record["id"] != row["id"] for row, record in zip(rows, cnn_records)):
        raise EvaluationError("CNN cache order does not match the development manifest")

    texture_tp, texture_blank_fp, and_tp, and_fp, and_blank_fp = [], [], [], [], []
    for texture, cnn in zip(texture_records, cnn_records):
        positives, blank = texture["positive"], texture["blank"]
        tp_only, blank_only = _texture_counts(texture["probability"], positives, blank)
        tp_fused, fp_fused, blank_fused = _and_counts(
            texture["probability"], cnn["probability"], positives, blank
        )
        texture_tp.append(tp_only)
        texture_blank_fp.append(blank_only)
        and_tp.append(tp_fused)
        and_fp.append(fp_fused)
        and_blank_fp.append(blank_fused)
    texture_tp_array = np.stack(texture_tp)
    texture_fp_array = np.stack([
        _threshold_counts(record["probability"], ~record["positive"])[0]
        for record in texture_records
    ])
    return {
        "texture_only": _serialize_grid(
            rows, texture_records, texture_tp_array, texture_fp_array,
            np.stack(texture_blank_fp), fused=False,
        ),
        "and": _serialize_grid(
            rows, texture_records, np.stack(and_tp), np.stack(and_fp),
            np.stack(and_blank_fp), fused=True,
        ),
    }


def _load_named_caches(
    named: Sequence[tuple[str, Path]], rows: Sequence[Mapping[str, Any]],
    data_root: Path, probability_name: str,
) -> dict[str, list[dict[str, Any]]]:
    result = {}
    for name, path in named:
        if name in result:
            raise EvaluationError(f"duplicate cache name: {name}")
        result[name] = load_model_cache(rows, data_root, path, probability_name, name)
    return result


def _metric_deltas(
    first: Mapping[str, Any], second: Mapping[str, Any]
) -> dict[str, Any]:
    keys = ("document_macro_iou", "global_recall", "global_precision")
    result: dict[str, Any] = {}
    for mode in ("texture_only", "and"):
        result[mode] = {key: _subtract_grid(first[mode][key], second[mode][key]) for key in keys}
    return result


def _subtract_grid(first: Sequence[Any], second: Sequence[Any]) -> list[Any]:
    if first and isinstance(first[0], list):
        return [
            [None if left is None or right is None else left - right
             for left, right in zip(left_row, right_row)]
            for left_row, right_row in zip(first, second)
        ]
    return [None if left is None or right is None else left - right
            for left, right in zip(first, second)]


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(None if argv is None else list(argv))
    rows = _manifest_rows(args.manifest)
    _validate_manifest(
        args.manifest, args.data_root, rows, args.expected_queries, args.expected_documents
    )
    metadata_names = [name for name, _ in args.training_metadata]
    if len(set(metadata_names)) != len(metadata_names):
        raise EvaluationError("duplicate training metadata name")
    cache_names = {name for name, _ in args.texture_cache} | {"CNN"}
    if not set(metadata_names) <= cache_names:
        raise EvaluationError("training metadata names must match supplied model caches")
    metadata = dict(args.training_metadata)
    _validate_fit_metadata(args.fit_manifest, rows, metadata, args.data_root)
    textures = _load_named_caches(args.texture_cache, rows, args.data_root, "texture_probability")
    cnn = load_model_cache(rows, args.data_root, args.cnn_cache, "cnn_probability", "CNN")
    if len(textures) < 2:
        raise EvaluationError("at least two texture caches are required for the matched comparison")
    model_results = {
        name: analyze_model(rows, records, cnn)
        for name, records in textures.items()
    }
    result = {
        "schema": "hatch-research-pareto/v1",
        "dataset": {
            "manifest": str(args.manifest), "queries": len(rows),
            "documents": len({row["document_id"] for row in rows}),
            "kind": "real", "known_pixels_only": True,
            "coordinate_order": "row-major reviewed coordinates; explicit y/x verified when present",
        },
        "threshold_grid": THRESHOLDS.tolist(),
        "models": model_results,
    }
    if "crop" in model_results and "distribution" in model_results:
        result["matched_deltas_crop_minus_distribution"] = _metric_deltas(
            model_results["crop"], model_results["distribution"]
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
