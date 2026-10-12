"""Confirm frozen H5 texture/CNN decisions against completed native masks."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import numpy as np

import evaluate_texture_sweep
import hatch_texture
from evaluate import aggregate, load_binary_png, load_manifest, metrics_from_counts, reviewed_domains
from inference import read_drawing, read_requests
from inference_texture import verify_model_metadata
from texture_learning import load_texture_model

FREEZE_PATH = Path("artifacts/frozen-pipeline-v3.json")
INPUTS_PATH = Path("artifacts/validation-inputs.json")
VALIDATION_PATH = Path("dataset/val.json")
OFFICIAL_METRICS_PATH = Path("artifacts/validation-v3/official-metrics.json")
RUNTIME_PATH = Path("artifacts/validation-v3/predictions/runtime.json")
PREDICTION_DIR = Path("artifacts/validation-v3/predictions")
CNN_CACHE_DIR = Path("artifacts/validation-analysis-v1/probabilities")
OUTPUT_DIR = Path("artifacts/validation-v3/probabilities")
SUMMARY_PATH = Path("artifacts/validation-v3/native-confirmation.json")
TEXTURE_THRESHOLD = 0.15
CNN_THRESHOLD = 0.25


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def validate_freeze(freeze: Mapping[str, Any]) -> None:
    if freeze.get("schema") != "hatch-matching-freeze/v3":
        raise ValueError("Expected the immutable v3 freeze")
    selection = freeze.get("selection")
    expected = {
        "operator": "AND", "quantile": "q10", "scale": 0,
        "offset": 0.15, "clip": [0.01, 0.99], "cnn_threshold": 0.25,
    }
    if not isinstance(selection, dict) or any(selection.get(key) != value for key, value in expected.items()):
        raise ValueError("Frozen selection does not match H5 thresholds")
    if freeze.get("native_confirmation_pending") is not True:
        raise ValueError("Freeze is not marked pending for native confirmation")
    if freeze.get("public_validation_exposed_before_freeze") is not True:
        raise ValueError("Freeze validation disclosure is missing")


def verify_frozen_hashes(root: Path, freeze: Mapping[str, Any]) -> dict[str, str]:
    observed: dict[str, str] = {}
    for model in freeze.get("models", []):
        path = root / str(model["path"])
        actual = sha256_file(path)
        if actual != model.get("sha256"):
            raise ValueError(f"Frozen model hash mismatch: {model['path']}")
        observed[str(model["path"])] = actual

    texture_path = root / str(freeze["models"][0]["path"])
    metadata_path = texture_path.with_suffix(".json")
    metadata_hash = sha256_file(metadata_path)
    if metadata_hash != freeze.get("model_metadata_sha256"):
        raise ValueError("Frozen texture metadata hash mismatch")
    observed[str(metadata_path.relative_to(root))] = metadata_hash

    for relative, expected in freeze.get("source_sha256", {}).items():
        path = root / relative
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"Frozen source hash mismatch: {relative}")
        observed[relative] = actual

    inputs_hash = sha256_file(root / INPUTS_PATH)
    if inputs_hash != freeze.get("validation_inputs_sha256"):
        raise ValueError("Frozen validation-input hash mismatch")
    observed[str(INPUTS_PATH)] = inputs_hash
    return observed


def validate_request_rows(
    inputs: Sequence[Mapping[str, Any]], validation_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    input_by_id = {str(row["id"]): row for row in inputs}
    validation_by_id = {str(row["id"]): row for row in validation_rows}
    if len(input_by_id) != len(inputs) or set(input_by_id) != set(validation_by_id):
        raise ValueError("Validation requests and labeled rows have different IDs")
    keys = ("image", "width", "height", "query_box", "context_boxes")
    for ident, row in validation_by_id.items():
        request = input_by_id[ident]
        if any(request.get(key, [] if key == "context_boxes" else None) != row.get(key, [] if key == "context_boxes" else None) for key in keys):
            raise ValueError(f"Validation request differs from labeled row: {ident}")
    return validation_by_id


def validate_cache_alignment(
    cache: Mapping[str, np.ndarray], row: Mapping[str, Any],
    known_y: np.ndarray, known_x: np.ndarray,
    positive_known: np.ndarray, blank_known: np.ndarray,
) -> np.ndarray:
    ident = str(row["id"])
    if str(np.asarray(cache["id"]).item()) != ident:
        raise ValueError(f"CNN cache ID mismatch for {ident}")
    expected = {
        "y": np.asarray(known_y, dtype=np.int32),
        "x": np.asarray(known_x, dtype=np.int32),
        "positive": np.asarray(positive_known, dtype=bool),
        "blank": np.asarray(blank_known, dtype=bool),
    }
    for key, values in expected.items():
        observed = np.asarray(cache[key])
        if observed.shape != values.shape or not np.array_equal(observed, values):
            raise ValueError(f"CNN cache {key} mismatch for {ident}")
    probability = np.asarray(cache["cnn_probability"])
    if probability.dtype != np.float32 or probability.shape != known_y.shape:
        raise ValueError(f"CNN probability vector has invalid shape or dtype for {ident}")
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError(f"CNN probability vector is invalid for {ident}")
    return probability


def threshold_mask(probability: np.ndarray, threshold: float) -> np.ndarray:
    values = np.asarray(probability)
    if values.dtype != np.float32 or not np.isfinite(values).all():
        raise ValueError("Threshold input must be finite float32 probabilities")
    return values >= np.float32(threshold)


def compare_reviewed_decisions(
    texture_probability: np.ndarray, cnn_probability: np.ndarray,
    native_decision: np.ndarray,
) -> tuple[np.ndarray, int]:
    if texture_probability.shape != cnn_probability.shape or texture_probability.shape != native_decision.shape:
        raise ValueError("Decision vectors must have identical shapes")
    candidate = threshold_mask(texture_probability, TEXTURE_THRESHOLD) & threshold_mask(
        cnn_probability, CNN_THRESHOLD
    )
    return candidate, int(np.count_nonzero(candidate != native_decision))


def _compare_official_counts(
    candidate: Mapping[str, int], official: Mapping[str, Any], ident: str,
) -> None:
    official_counts = official.get("metrics", {})
    keys = ("tp", "fp", "fn", "tn", "reviewed_pixels", "positive_pixels",
            "blank_false_positive_pixels", "reviewed_blank_pixels")
    mismatches = {key: (candidate[key], official_counts.get(key)) for key in keys
                  if candidate[key] != official_counts.get(key)}
    if mismatches:
        raise ValueError(f"Candidate counts differ from official native counts for {ident}: {mismatches}")


def _compare_official_summary(candidate: Mapping[str, Any], official: Mapping[str, Any]) -> None:
    expected = official.get("summary")
    if not isinstance(expected, dict):
        raise ValueError("Official metrics report has no aggregate summary")
    if dict(candidate) != expected:
        differing = sorted(key for key in set(candidate) | set(expected)
                           if candidate.get(key) != expected.get(key))
        raise ValueError(f"Candidate aggregate differs from official summary: {differing}")


def _score_query(
    root: Path, row: Mapping[str, Any], gray: np.ndarray, features: np.ndarray,
    texture_model: Any, metadata: Mapping[str, Any],
    official_row: Mapping[str, Any], cache_path: Path, prediction_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    positive, known, blank, domain_metadata = reviewed_domains(row, root / "dataset")
    known_y, known_x = np.nonzero(known)
    positive_known, blank_known = positive[known], blank[known]
    with np.load(cache_path, allow_pickle=False) as opened:
        cache = {key: opened[key] for key in opened.files}
    cnn_probability = validate_cache_alignment(
        cache, row, known_y, known_x, positive_known, blank_known
    )

    stride = int(metadata.get("stride", 4))
    query_box = tuple(int(value) for value in row["query_box"])
    if len(query_box) != 4:
        raise ValueError(f"Query box must contain four coordinates for {row['id']}")
    query_box = (query_box[0], query_box[1], query_box[2], query_box[3])
    source = hatch_texture.descriptor_source_from_metadata(dict(metadata))
    feature_set = hatch_texture.feature_set_from_metadata(dict(metadata))
    query_descriptor = hatch_texture.descriptor_for_query(
        gray, features, query_box, stride, source, feature_set=feature_set
    )
    texture_probability = evaluate_texture_sweep.score_known_probabilities(
        texture_model, features, query_descriptor, known, gray.shape[0], gray.shape[1], stride
    )
    if texture_probability.dtype != np.float32 or texture_probability.shape != known_y.shape:
        raise ValueError(f"Texture probability vector has invalid shape or dtype for {row['id']}")
    if not np.isfinite(texture_probability).all() or np.any((texture_probability < 0) | (texture_probability > 1)):
        raise ValueError(f"Texture probabilities are invalid for {row['id']}")

    native = load_binary_png(prediction_path, int(row["width"]), int(row["height"]))
    native_decision = native[known_y, known_x]
    candidate_decision, mismatches = compare_reviewed_decisions(
        texture_probability, cnn_probability, native_decision
    )
    tp = int(np.count_nonzero(candidate_decision & positive_known))
    fp = int(np.count_nonzero(candidate_decision & ~positive_known))
    fn = int(np.count_nonzero(~candidate_decision & positive_known))
    tn = int(len(known_y) - tp - fp - fn)
    blank_fp = int(np.count_nonzero(candidate_decision & blank_known))
    counts = metrics_from_counts(tp, fp, fn, tn, blank_fp, int(blank_known.sum()))
    _compare_official_counts(counts, official_row, str(row["id"]))
    if mismatches:
        raise ValueError(f"Native mask decision mismatch for {row['id']}: {mismatches}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path, y=known_y.astype(np.int32), x=known_x.astype(np.int32),
        texture_probability=texture_probability, positive=positive_known.astype(bool),
        blank=blank_known.astype(bool), id=np.asarray(str(row["id"])),
    )
    return {
        "id": str(row["id"]), "document_id": str(row["document_id"]),
        "kind": str(row["kind"]), **domain_metadata, "positive_only": bool(domain_metadata["positive_only"]),
        "metrics": counts, "reviewed_decision_mismatches": mismatches,
        "texture_probability_sha256": sha256_file(output_path),
        "cnn_cache_sha256": sha256_file(cache_path),
    }


def _require_outputs(root: Path, rows: Sequence[Mapping[str, Any]], official: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    official_rows = official.get("examples")
    if not isinstance(official_rows, list):
        raise ValueError("Official metrics report has no examples")
    official_by_id = {str(item["id"]): item for item in official_rows}
    expected_ids = {str(row["id"]) for row in rows}
    if len(official_by_id) != len(official_rows) or set(official_by_id) != expected_ids:
        raise ValueError("Official metrics IDs do not match validation IDs")
    required = [root / OFFICIAL_METRICS_PATH, root / RUNTIME_PATH]
    required.extend(root / PREDICTION_DIR / f"{row['id']}.png" for row in rows)
    required.extend(root / CNN_CACHE_DIR / f"{row['id']}.npz" for row in rows)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Native confirmation prerequisites are missing: {missing[:5]}")
    return official_by_id


def run_confirmation(root: Path) -> dict[str, Any]:
    root = root.resolve()
    freeze_path = root / FREEZE_PATH
    freeze = _read_json(freeze_path)
    validate_freeze(freeze)
    observed_hashes = verify_frozen_hashes(root, freeze)

    input_rows = read_requests(root / INPUTS_PATH)
    validation_rows = load_manifest(root / VALIDATION_PATH)
    rows_by_id = validate_request_rows(input_rows, validation_rows)
    rows = [rows_by_id[str(input_row["id"])] for input_row in input_rows]
    official = _read_json(root / OFFICIAL_METRICS_PATH)
    official_by_id = _require_outputs(root, rows, official)

    output_dir = root / OUTPUT_DIR
    if output_dir.exists() or (root / SUMMARY_PATH).exists():
        raise FileExistsError("Native confirmation output already exists; refusing to overwrite")
    stage_dir = output_dir.with_name(output_dir.name + ".partial")
    if stage_dir.exists():
        raise FileExistsError(f"Incomplete confirmation output exists: {stage_dir}")

    texture_path = root / str(freeze["models"][0]["path"])
    texture_model, metadata = load_texture_model(texture_path)
    verify_model_metadata(metadata)
    if sha256_file(texture_path.with_suffix(".json")) != freeze["model_metadata_sha256"]:
        raise ValueError("Loaded texture metadata differs from frozen metadata")

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["image"])].append(row)
    results: list[dict[str, Any]] = []
    stage_dir.mkdir(parents=True)
    try:
        completed = 0
        for image_rel, image_rows in grouped.items():
            gray = read_drawing(root / "dataset", dict(image_rows[0]))
            stride = int(metadata.get("stride", 4))
            feature_set = hatch_texture.feature_set_from_metadata(metadata)
            features = hatch_texture.texture_features(gray, stride=stride, feature_set=feature_set)
            for row in image_rows:
                results.append(_score_query(
                    root, row, gray, features, texture_model, metadata,
                    official_by_id[str(row["id"])],
                    root / CNN_CACHE_DIR / f"{row['id']}.npz",
                    root / PREDICTION_DIR / f"{row['id']}.png",
                    stage_dir / f"{row['id']}.npz",
                ))
                completed += 1
                print(f"Confirmed {completed}/{len(rows)}: {row['id']}", flush=True)
            del features, gray

        summary = aggregate(results)
        _compare_official_summary(summary, official.get("summary", {}))
        reviewed_pixels = sum(int(row["metrics"]["reviewed_pixels"]) for row in results)
        output = {
            "schema": "hatch-matching-native-confirmation/v1",
            "freeze_sha256": sha256_file(freeze_path),
            "freeze_candidate": freeze.get("selection"),
            "model_source_input_hashes": observed_hashes,
            "texture_scorer_sha256": sha256_file(Path(__file__).resolve().parents[1] / "evaluate_texture_sweep.py"),
            "texture_scorer_source_sha256": sha256_file(Path(evaluate_texture_sweep.__file__).resolve()),
            "official_metrics_sha256": sha256_file(root / OFFICIAL_METRICS_PATH),
            "operator": "AND",
            "texture_threshold": TEXTURE_THRESHOLD,
            "cnn_threshold": CNN_THRESHOLD,
            "native_prediction_count": len(rows),
            "reviewed_pixel_count": reviewed_pixels,
            "reviewed_decision_mismatches": sum(int(row["reviewed_decision_mismatches"]) for row in results),
            "summary": summary,
            "examples": results,
            "whole_sheet_accuracy_certified": False,
            "limitations": ["Scores apply only to reviewed pixels.", "Validation was exposed before freeze; this is confirmation, not independent holdout evidence."],
        }
        if output["reviewed_decision_mismatches"] != 0:
            raise ValueError("Frozen H5 thresholds do not reproduce native mask decisions")
        stage_summary = stage_dir / "native-confirmation.json"
        stage_summary.write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(stage_dir, output_dir)
        os.replace(output_dir / "native-confirmation.json", root / SUMMARY_PATH)
        return output
    except Exception:
        shutil.rmtree(stage_dir, ignore_errors=True)
        raise


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."), help="Challenge repository root")
    args = parser.parse_args(argv)
    report = run_confirmation(args.root)
    print(json.dumps(report["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
