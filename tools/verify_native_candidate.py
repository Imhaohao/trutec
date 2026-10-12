"""Verify native masks, frozen inputs, official metrics, and measured runtime."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
from PIL import Image

from evaluate import EvaluationError, evaluate, load_manifest


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"Cannot read JSON file: {path}") from exc
    if not isinstance(value, dict):
        raise EvaluationError(f"Expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise EvaluationError(f"Cannot hash file: {path}") from exc
    return digest.hexdigest()


def _check_frozen_files(root: Path, frozen: dict[str, Any], inputs_path: Path) -> None:
    for relative, expected in frozen["source_sha256"].items():
        if _sha256(root / relative) != expected:
            raise EvaluationError(f"Frozen source hash mismatch: {relative}")
    for model in frozen["models"]:
        model_path = root / model["path"]
        if model_path.stat().st_size != model["bytes"] or _sha256(model_path) != model["sha256"]:
            raise EvaluationError(f"Frozen model hash or size mismatch: {model['path']}")
    metadata_path = root / frozen["models"][0]["path"].replace(".joblib", ".json")
    if _sha256(metadata_path) != frozen["model_metadata_sha256"]:
        raise EvaluationError("Frozen texture metadata hash mismatch")
    if _sha256(inputs_path) != frozen["validation_inputs_sha256"]:
        raise EvaluationError("Frozen validation input hash mismatch")


def _input_rows(path: Path) -> list[dict[str, Any]]:
    payload = _read_json(path)
    if payload.get("schema") != "hatch-matching-inputs/v1":
        raise EvaluationError("Unsupported label-free input schema")
    rows = load_manifest(path, require_labels=False)
    allowed = {"id", "image", "width", "height", "query_box", "context_boxes"}
    if any(set(row) != allowed for row in rows):
        raise EvaluationError("Input rows must contain only the frozen inference fields")
    return rows


def _compare_requests(inputs: list[dict[str, Any]], public_rows: list[dict[str, Any]]) -> None:
    input_by_id = {row["id"]: row for row in inputs}
    public_by_id = {row["id"]: row for row in public_rows}
    if len(input_by_id) != len(inputs) or input_by_id.keys() != public_by_id.keys():
        raise EvaluationError("Input and public validation IDs differ")
    fields = ("image", "width", "height", "query_box", "context_boxes")
    for ident, request in input_by_id.items():
        if any(request.get(field) != public_by_id[ident].get(field) for field in fields):
            raise EvaluationError(f"Input request differs from public validation row: {ident}")


def _runtime_ids(report: dict[str, Any], expected: set[str]) -> None:
    examples = report.get("examples")
    if not isinstance(examples, list):
        raise EvaluationError("Runtime report lacks examples")
    identifiers = [item.get("id") for item in examples if isinstance(item, dict)]
    if len(identifiers) != len(examples) or len(set(identifiers)) != len(identifiers):
        raise EvaluationError("Runtime report IDs are invalid or duplicated")
    if set(identifiers) != expected or report.get("processed_query_count") != len(expected):
        raise EvaluationError("Runtime report IDs do not match the request manifest")


def _check_runtime_settings(report: dict[str, Any], frozen: dict[str, Any], inputs: Path) -> None:
    selection = frozen["selection"]
    if report.get("method") != "learned-texture-cnn-fusion-v1" or report.get("operator") != "AND":
        raise EvaluationError("Runtime method does not match frozen fusion pipeline")
    expected_selection = {key: selection[key] for key in ("quantile", "scale", "offset", "cnn_threshold")}
    if report.get("selection") != expected_selection:
        raise EvaluationError("Runtime selection settings differ from freeze")
    if report.get("tile") != {"tile_size": selection["tile_size"], "overlap": selection["overlap"]}:
        raise EvaluationError("Runtime tile settings differ from freeze")
    if report.get("device") != selection["device"] or report.get("inputs") != str(inputs):
        raise EvaluationError("Runtime device or input path differs from freeze")
    for record in report.get("examples", []):
        if any(record.get(key) != selection[key] for key in ("quantile", "scale", "offset", "cnn_threshold")):
            raise EvaluationError(f"Per-query thresholds differ from freeze: {record.get('id')}")
        expected_texture_threshold = min(
            max(selection["offset"], selection["clip"][0]), selection["clip"][1]
        )
        if record.get("texture_threshold") != expected_texture_threshold:
            raise EvaluationError(f"Per-query texture threshold differs from freeze: {record.get('id')}")
    for key, model in (("texture_model", frozen["models"][0]), ("cnn_model", frozen["models"][1])):
        if report.get(key) != model["path"]:
            raise EvaluationError(f"Runtime {key} path differs from freeze")
        if report.get(f"{key}_sha256") != model["sha256"]:
            raise EvaluationError(f"Runtime {key} hash differs from freeze")


def _check_prediction_files(
    rows: list[dict[str, Any]], prediction_dir: Path, report: dict[str, Any]
) -> list[dict[str, Any]]:
    expected_ids = {row["id"] for row in rows}
    observed_ids = {path.stem for path in prediction_dir.glob("*.png")}
    if expected_ids != observed_ids:
        raise EvaluationError("Prediction PNG IDs do not match the request manifest")
    proofs = []
    runtime_counts = {item["id"]: item.get("selected_pixels") for item in report["examples"]}
    for row in rows:
        path = prediction_dir / f"{row['id']}.png"
        with Image.open(path) as image:
            if image.format != "PNG" or image.mode != "L":
                raise EvaluationError(f"Prediction must be an 8-bit grayscale PNG: {path}")
            if image.size != (row["width"], row["height"]):
                raise EvaluationError(f"Prediction does not preserve native geometry: {path}")
            values = np.asarray(image)
            if not set(np.unique(values).tolist()) <= {0, 255}:
                raise EvaluationError(f"Prediction values must use 0/255 encoding: {path}")
            selected = int(np.count_nonzero(values))
        if runtime_counts[row["id"]] != selected:
            raise EvaluationError(f"Runtime selected-pixel count differs from PNG: {row['id']}")
        proofs.append({
            "id": row["id"], "width": row["width"], "height": row["height"],
            "selected_pixels": selected, "prediction_sha256": _sha256(path),
        })
    return proofs


def _runtime_measurements(log_path: Path, report: dict[str, Any]) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    patterns = {
        "external_wall_seconds": r"^\s*([\d.]+)\s+real\b",
        "maximum_resident_set_bytes": r"^\s*(\d+)\s+maximum resident set size\b",
        "peak_memory_footprint_bytes": r"^\s*(\d+)\s+peak memory footprint\b",
        "swaps": r"^\s*(\d+)\s+swaps\b",
    }
    found = {key: re.search(pattern, text, re.MULTILINE) for key, pattern in patterns.items()}
    if any(match is None for match in found.values()):
        raise EvaluationError("Inference log lacks complete /usr/bin/time -l measurements")
    values: dict[str, Any] = {
        key: (float(match.group(1)) if key.endswith("seconds") else int(match.group(1)))
        for key, match in found.items() if match is not None
    }
    values["internal_elapsed_seconds"] = report["elapsed_seconds"]
    values["measurement"] = "/usr/bin/time -l"
    return values


def _rates(summary: dict[str, Any]) -> dict[str, float | None]:
    tp, fp, tn = summary["tp"], summary["fp"], summary["tn"]
    return {
        "false_discovery_rate": fp / (tp + fp) if tp + fp else None,
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
    }


def _check_official_report(path: Path, calculated: dict[str, Any]) -> tuple[dict[str, Any], bytes]:
    try:
        report_bytes = path.read_bytes()
        report = json.loads(report_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"Cannot read official evaluator report: {path}") from exc
    if report != calculated:
        raise EvaluationError("Saved official metrics differ from unchanged evaluator output")
    expected_bytes = (json.dumps(calculated, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    if report_bytes != expected_bytes:
        raise EvaluationError("Official metrics are not the verbatim evaluate.py report")
    return report, report_bytes


def _compare_v2(summary: dict[str, Any], integrity_path: Path) -> dict[str, Any]:
    prior = _read_json(integrity_path)["summary"]
    metrics = ("document_macro_iou", "global_precision", "global_recall", "tp", "fp", "tn", "fn")
    return {
        "v2": {**{key: prior[key] for key in metrics}, **_rates(prior)},
        "v3_minus_v2": {
            key: summary[key] - prior[key]
            for key in metrics
            if summary.get(key) is not None and prior.get(key) is not None
        },
    }


def _cnn_subset_check(
    rows: list[dict[str, Any]], prediction_dir: Path, cache_dir: Path, threshold: float
) -> dict[str, Any]:
    ids = {row["id"] for row in rows}
    if {path.stem for path in cache_dir.glob("*.npz")} != ids:
        raise EvaluationError("CNN cache IDs do not match validation requests")
    checked = mismatches = 0
    for row in rows:
        with np.load(cache_dir / f"{row['id']}.npz", allow_pickle=False) as archive:
            if not {"cnn_probability", "y", "x"}.issubset(archive.files):
                raise EvaluationError(f"CNN cache lacks probability coordinates: {row['id']}")
            probability = np.asarray(archive["cnn_probability"])
            y, x = np.asarray(archive["y"]), np.asarray(archive["x"])
        if (probability.dtype != np.float32 or probability.ndim != 1 or y.shape != probability.shape
                or x.shape != probability.shape or y.dtype.kind not in "iu" or x.dtype.kind not in "iu"):
            raise EvaluationError(f"CNN cache vectors have invalid shape or dtype: {row['id']}")
        if (not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1))
                or len(set(zip(y.tolist(), x.tolist()))) != len(y)):
            raise EvaluationError(f"CNN cache probabilities or coordinates are invalid: {row['id']}")
        if np.any((y < 0) | (y >= row["height"]) | (x < 0) | (x >= row["width"])):
            raise EvaluationError(f"CNN cache coordinates are outside native geometry: {row['id']}")
        with Image.open(prediction_dir / f"{row['id']}.png") as image:
            mask = np.asarray(image) != 0
            selected = mask[y, x]
        cnn_selected = probability >= threshold
        checked += len(y)
        mismatches += int(np.count_nonzero(selected & ~cnn_selected))
    if mismatches:
        raise EvaluationError(f"Native predictions violate CNN threshold gate at {mismatches} cached pixels")
    return {"reviewed_cache_coordinates": checked, "selected_outside_cnn_gate": mismatches}


def verify_candidate(
    root: Path,
    freeze_path: Path,
    inputs_path: Path,
    public_manifest: Path,
    data_root: Path,
    prediction_dir: Path,
    inference_log: Path,
    official_metrics_path: Path,
    prior_integrity: Path,
    output_path: Path,
    cnn_cache: Path | None = None,
) -> dict[str, Any]:
    frozen = _read_json(freeze_path)
    inputs = _input_rows(inputs_path)
    public_rows = load_manifest(public_manifest)
    _compare_requests(inputs, public_rows)
    _check_frozen_files(root, frozen, inputs_path)
    runtime = _read_json(prediction_dir / "runtime.json")
    expected_ids = {row["id"] for row in inputs}
    _runtime_ids(runtime, expected_ids)
    _check_runtime_settings(runtime, frozen, inputs_path)
    prediction_proofs = _check_prediction_files(inputs, prediction_dir, runtime)
    calculated = evaluate(public_manifest, data_root, prediction_dir)
    official, official_bytes = _check_official_report(official_metrics_path, calculated)
    summary = official["summary"]
    gates = {
        "macro_iou_ge_0_95": (summary["document_macro_iou"] or 0) >= 0.95,
        "precision_ge_0_95": (summary["global_precision"] or 0) >= 0.95,
        "recall_ge_0_98": (summary["global_recall"] or 0) >= 0.98,
    }
    proof: dict[str, Any] = {
        "schema": "hatch-matching-native-integrity/v3",
        "validation_informed": True,
        "queries": len(inputs), "documents": len({row["document_id"] for row in public_rows}),
        "all_request_prediction_report_ids_match": True,
        "source_models_metadata_and_inputs_match_frozen": True,
        "runtime_settings_match_frozen": True,
        "native_final_masks_verified": True,
        "official_metrics": summary,
        **_rates(summary),
        "strict_gates": gates,
        "all_strict_gates_met": all(gates.values()),
        "comparison_to_fixed_v2": _compare_v2(summary, prior_integrity),
        "runtime": _runtime_measurements(inference_log, runtime),
        "predictions": prediction_proofs,
        "freeze_sha256": _sha256(freeze_path),
        "official_metrics_sha256": hashlib.sha256(official_bytes).hexdigest(),
        "official_metrics_normalized_sha256": hashlib.sha256(
            json.dumps(official, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "whole_sheet_accuracy_certified": False,
        "private_test_accuracy_known": False,
    }
    if cnn_cache is not None:
        proof["cnn_gate_subset_check"] = _cnn_subset_check(
            inputs, prediction_dir, cnn_cache, frozen["selection"]["cnn_threshold"]
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")
    return proof


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--freeze", type=Path, default=Path("artifacts/frozen-pipeline-v3.json"))
    parser.add_argument("--inputs", type=Path, default=Path("artifacts/validation-inputs.json"))
    parser.add_argument("--public-manifest", type=Path, default=Path("dataset/val.json"))
    parser.add_argument("--data-root", type=Path, default=Path("dataset"))
    parser.add_argument("--prediction-dir", type=Path, default=Path("artifacts/validation-v3/predictions"))
    parser.add_argument("--inference-log", type=Path, default=Path("artifacts/validation-v3/inference.log"))
    parser.add_argument("--official-metrics", type=Path, default=Path("artifacts/validation-v3/official-metrics.json"))
    parser.add_argument("--prior-integrity", type=Path, default=Path("artifacts/validation-v2/integrity.json"))
    parser.add_argument("--cnn-cache", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/validation-v3/integrity.json"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    verify_candidate(
        args.root, args.freeze, args.inputs, args.public_manifest, args.data_root,
        args.prediction_dir, args.inference_log, args.official_metrics, args.prior_integrity,
        args.output, args.cnn_cache,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
