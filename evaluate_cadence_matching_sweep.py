"""Write one exact known-native development probability cache for the cadence model."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

import cadence_matching
import hatch_cadence
import hatch_texture
import texture_learning
from evaluate import EvaluationError, load_manifest, reviewed_domains
from evaluate_dino_matching_sweep import save_known_probabilities
from evaluate_hatchfinder_sweep import validate_split_safety
from evaluate_texture_sweep import find_required_grid_indices, remap_known_coordinates
from inference import read_drawing
from inference_texture import native_grid_coordinates
from tools.dino_patch_support import FIT_MANIFEST_SHA256
from train_cadence_matching import verify_resource_proof

ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / "artifacts/research-v4/cadence-feature-protocol.json"
RESOURCE_PROOF = ROOT / "artifacts/research-v4/cadence-resource-pilot-proof.json"
EVALUATION_PLAN = ROOT / "artifacts/research-v4/cadence-evaluation-plan.json"
SUPPORT_SHA256 = "dd1b43e144349d1fe8a91f83470996fb5f8b6e6201a3812805ce2ba5d1941af3"
DEVELOPMENT_MANIFEST_SHA256 = "2fd028093d0c6cdb4ec063d6c684d642001ced5b7ed154aacc0c7d9a76cd874a"
FIT_ROWS = 357
FIT_DOCUMENTS = 105
DEVELOPMENT_ROWS = 55
DEVELOPMENT_DOCUMENTS = 12
PREDICTION_BATCH_SIZE = 8192
STRIDE = 4


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"Cannot read JSON file: {path}") from exc
    if not isinstance(value, dict):
        raise EvaluationError(f"Expected JSON object: {path}")
    return value


def _load_verified_model(path: Path) -> tuple[Any, dict[str, Any], str]:
    model_sha = _sha256(path)
    digest_sidecar = path.with_suffix(path.suffix + ".sha256")
    try:
        digest_line = digest_sidecar.read_text(encoding="utf-8").strip().split()
    except (OSError, UnicodeError) as exc:
        raise EvaluationError(f"Cannot read model SHA sidecar: {digest_sidecar}") from exc
    if len(digest_line) != 2 or digest_line[0] != model_sha or digest_line[1] != path.name:
        raise EvaluationError("Cadence model SHA sidecar does not match the saved artifact")
    model, metadata = texture_learning.load_texture_model(path)
    sidecar_metadata = _read_json(path.with_suffix(".json"))
    try:
        canonical_metadata = json.loads(json.dumps(metadata, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise EvaluationError("Embedded cadence model metadata is not JSON serializable") from exc
    if sidecar_metadata != canonical_metadata:
        raise EvaluationError("Cadence model metadata sidecar differs from embedded metadata")
    if type(model) is not HistGradientBoostingClassifier or model.n_features_in_ != 204:
        raise EvaluationError("Cadence estimator must be an HGB with 204 input features")
    expected_params = {"learning_rate": 0.08, "max_leaf_nodes": 31, "l2_regularization": 1.0,
                       "max_iter": 200, "early_stopping": False, "random_state": 42}
    model_params = model.get_params(deep=False)
    if any(model_params.get(key) != value for key, value in expected_params.items()):
        raise EvaluationError("Cadence HGB estimator settings differ from the frozen classifier")
    if not np.array_equal(model.classes_, np.asarray([0, 1])):
        raise EvaluationError("Cadence estimator classes must be ordered [0, 1]")
    _validate_model_metadata(metadata)
    return model, metadata, model_sha


def _validate_model_metadata(metadata: dict[str, Any]) -> None:
    expected_base = hatch_texture.texture_feature_names("base") + list(hatch_cadence.FEATURE_NAMES)
    if metadata.get("schema") != "cadence-pair-classifier/v1":
        raise EvaluationError("Unsupported cadence pair-classifier schema")
    if metadata.get("query_descriptor_source") != "crop" or metadata.get("stride") != STRIDE:
        raise EvaluationError("Cadence model must use exact-crop query descriptors at stride 4")
    if metadata.get("texture_feature_set") != "base" or metadata.get("feature_set") != "base+cadence":
        raise EvaluationError("Cadence model must use base texture plus the five cadence channels")
    if metadata.get("base_feature_names") != expected_base or len(expected_base) != 51:
        raise EvaluationError("Cadence model base feature signature differs from the frozen 51-channel order")
    pair_names = metadata.get("pair_feature_names")
    expected_pairs = texture_learning.pair_feature_names(expected_base, "crop")
    if pair_names != expected_pairs or len(pair_names) != 204:
        raise EvaluationError("Cadence model pair feature signature differs from the frozen 204-channel order")
    expected_params = {"learning_rate": 0.08, "max_leaf_nodes": 31, "l2_regularization": 1.0,
                       "max_iter": 200, "early_stopping": False, "random_state": 42}
    if metadata.get("params") != expected_params:
        raise EvaluationError("Cadence model metadata differs from the frozen classifier settings")
    if metadata.get("cadence_protocol_sha256") != _sha256(PROTOCOL):
        raise EvaluationError("Cadence model protocol hash differs from the frozen feature protocol")
    if metadata.get("resource_pilot_sha256") != _sha256(RESOURCE_PROOF):
        raise EvaluationError("Cadence model resource pilot hash differs from the passed proof")
    source_hashes = metadata.get("source_sha256")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise EvaluationError("Cadence model metadata lacks frozen implementation hashes")
    for relative, expected in source_hashes.items():
        if _sha256(ROOT / relative) != expected:
            raise EvaluationError(f"Cadence model source hash mismatch: {relative}")


def _validate_partitions(
    development_path: Path,
    fit_path: Path,
    data_root: Path,
    development_rows: list[dict[str, Any]],
    fit_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> tuple[str, str]:
    fit_sha, development_sha = _sha256(fit_path), _sha256(development_path)
    if fit_sha != FIT_MANIFEST_SHA256 or development_sha != DEVELOPMENT_MANIFEST_SHA256:
        raise EvaluationError("Cadence sweep requires the pinned fit and real-development manifests")
    _check_partition_sizes(fit_rows, development_rows)
    fit_ids = {str(row["id"]) for row in fit_rows}
    fit_docs = {str(row["document_id"]) for row in fit_rows}
    development_ids = {str(row["id"]) for row in development_rows}
    development_docs = {str(row["document_id"]) for row in development_rows}
    if fit_ids & development_ids or fit_docs & development_docs:
        raise EvaluationError("Fit and development manifests share query or document IDs")
    if metadata.get("fit_manifest_sha256") != fit_sha or metadata.get("source_manifest_sha256") != fit_sha:
        raise EvaluationError("Cadence model fit/source manifest hashes differ from the pinned fit split")
    if sorted(metadata.get("training_ids", [])) != sorted(fit_ids):
        raise EvaluationError("Cadence model training IDs differ from the pinned fit split")
    if sorted(metadata.get("training_documents", [])) != sorted(fit_docs):
        raise EvaluationError("Cadence model training documents differ from the pinned fit split")
    validate_split_safety(development_path, data_root, development_rows)
    validate_split_safety(fit_path, data_root, fit_rows)
    return fit_sha, development_sha


def _check_partition_sizes(fit_rows: list[dict[str, Any]], development_rows: list[dict[str, Any]]) -> None:
    fit_docs = {str(row["document_id"]) for row in fit_rows}
    development_docs = {str(row["document_id"]) for row in development_rows}
    if len(fit_rows) != FIT_ROWS or len(fit_docs) != FIT_DOCUMENTS:
        raise EvaluationError("Cadence sweep requires all 357 fit queries across 105 documents")
    if len(development_rows) != DEVELOPMENT_ROWS or len(development_docs) != DEVELOPMENT_DOCUMENTS:
        raise EvaluationError("Cadence sweep requires all 55 development queries across 12 documents")
    if any(row.get("kind") != "real" for row in development_rows):
        raise EvaluationError("Cadence sweep accepts only real development queries")


def _check_resource_pilot(path: Path) -> dict[str, Any]:
    return verify_resource_proof(path, FIT_MANIFEST_SHA256, SUPPORT_SHA256)


def _check_evaluation_plan() -> dict[str, Any]:
    plan = _read_json(EVALUATION_PLAN)
    if plan.get("schema") != "cadence-evaluation-plan/v1":
        raise EvaluationError("Unsupported frozen cadence evaluation plan")
    if plan.get("protocol_sha256") != _sha256(PROTOCOL):
        raise EvaluationError("Cadence evaluation plan refers to another feature protocol")
    manifest_hashes = plan.get("manifest_sha256", {})
    if (manifest_hashes.get("fit") != FIT_MANIFEST_SHA256
            or manifest_hashes.get("development") != DEVELOPMENT_MANIFEST_SHA256):
        raise EvaluationError("Cadence evaluation plan manifest hashes are not the pinned partitions")
    return plan


def _probabilities_for_known(
    model: Any,
    gray: np.ndarray,
    shared_features: np.ndarray,
    query: np.ndarray,
    lag: hatch_cadence.LagState,
    known: np.ndarray,
    batch_size: int = PREDICTION_BATCH_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = gray.shape
    if known.shape != (height, width):
        raise EvaluationError("Known domain shape differs from the drawing")
    native_y, native_x = np.nonzero(known)
    if not len(native_y):
        return np.empty(0, np.float32), native_y.astype(np.int32), native_x.astype(np.int32)
    grid_y, grid_x = native_grid_coordinates(height, width, STRIDE)
    sample_gy, sample_gx = grid_y[native_y], grid_x[native_x]
    grid_h, grid_w = shared_features.shape[:2]
    unique_y, unique_x = find_required_grid_indices(sample_gy, sample_gx, grid_h, grid_w)
    coordinates = np.column_stack((unique_y, unique_x)).astype(np.int64, copy=False)
    cadence = hatch_cadence.cadence_features_tiled(gray, lag, coordinates)
    local_base = shared_features[unique_y, unique_x]
    if cadence.shape != (len(coordinates), len(hatch_cadence.FEATURE_NAMES)):
        raise EvaluationError("Sparse cadence pass returned an invalid feature shape")
    if not np.isfinite(cadence).all() or np.any((cadence < 0) | (cadence > 1)):
        raise EvaluationError("Sparse cadence features are non-finite or outside [0,1]")
    local = np.concatenate((local_base, cadence), axis=1).astype(np.float32)
    probabilities = _predict_batches(model, local, query, batch_size)
    probability_grid = np.zeros((grid_h, grid_w), dtype=np.float32)
    probability_grid[unique_y, unique_x] = probabilities
    known_probabilities = remap_known_coordinates(probability_grid, sample_gx, sample_gy)
    return known_probabilities, native_y.astype(np.int32), native_x.astype(np.int32)


def _predict_batches(model: Any, local: np.ndarray, query: np.ndarray, batch_size: int) -> np.ndarray:
    if batch_size <= 0:
        raise ValueError("Prediction batch size must be positive")
    scores = np.empty(len(local), dtype=np.float32)
    for start in range(0, len(local), batch_size):
        stop = min(start + batch_size, len(local))
        pair = texture_learning.pair_features(local[start:stop], query)
        if pair.shape != (stop - start, 204) or not np.isfinite(pair).all():
            raise EvaluationError("Cadence pair features are invalid")
        result = np.asarray(model.predict_proba(pair))
        if result.shape != (stop - start, 2) or not np.isfinite(result).all():
            raise EvaluationError("Cadence classifier returned invalid probabilities")
        scores[start:stop] = result[:, 1]
    if np.any((scores < 0) | (scores > 1)):
        raise EvaluationError("Cadence classifier probabilities lie outside [0,1]")
    return scores


def _run_queries(
    rows: list[dict[str, Any]], data_root: Path, model: Any, model_metadata: dict[str, Any],
    model_sha: str, output_dir: Path,
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["image"])].append(row)
    probability_dir = output_dir / "probabilities"
    probability_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    with (output_dir / "query-records.jsonl").open("w", encoding="utf-8") as log:
        for group in groups.values():
            _run_image_group(group, data_root, model, model_metadata, model_sha,
                             probability_dir, log, records, output_dir)
    return records


def _run_image_group(
    group: list[dict[str, Any]], data_root: Path, model: Any, model_metadata: dict[str, Any], model_sha: str,
    probability_dir: Path, log: Any, records: list[dict[str, Any]], output_dir: Path,
) -> None:
    gray = read_drawing(data_root, group[0])
    if gray.dtype != np.uint8 or gray.ndim != 2:
        raise EvaluationError("Drawing loader must return a two-dimensional uint8 image")
    image_digest = hashlib.sha256(gray.tobytes()).hexdigest()
    shared = hatch_texture.texture_features(gray, stride=STRIDE, feature_set="base")
    if shared.shape != (int(np.ceil(gray.shape[0] / 4)), int(np.ceil(gray.shape[1] / 4)), 46):
        raise EvaluationError("Shared base texture grid has invalid native dimensions")
    for row in group:
        _run_one_query(row, gray, image_digest, shared, data_root, model, model_metadata, model_sha,
                       probability_dir, log, records, output_dir)


def _run_one_query(
    row: dict[str, Any], gray: np.ndarray, image_digest: str, shared: np.ndarray,
    data_root: Path, model: Any, model_metadata: dict[str, Any], model_sha: str, probability_dir: Path,
    log: Any, records: list[dict[str, Any]], output_dir: Path,
) -> None:
    started = time.monotonic()
    height, width = gray.shape
    if (height, width) != (row["height"], row["width"]):
        raise EvaluationError(f"Drawing dimensions differ from development row {row['id']}")
    query, lag = cadence_matching.query_descriptor(gray, shared, tuple(row["query_box"]))
    positive, known, blank, _ = reviewed_domains(row, data_root)
    probabilities, native_y, native_x = _probabilities_for_known(
        model, gray, shared, query, lag, known
    )
    cache_path = save_known_probabilities(
        probability_dir, row, probabilities, positive[known], blank[known], native_y, native_x
    )
    x0, y0, x1, y1 = row["query_box"]
    record = {
        "id": str(row["id"]), "document_id": str(row["document_id"]),
        "kind": str(row["kind"]), "seconds": time.monotonic() - started,
        "native_shape": [height, width], "patch_grid": [shared.shape[0], shared.shape[1]],
        "known_pixels": len(native_y), "known_probabilities": str(cache_path.relative_to(output_dir)),
        "model_sha256": model_sha,
        "drawing_pixel_sha256": image_digest,
        "query_crop_pixel_sha256": hashlib.sha256(gray[y0:y1, x0:x1].tobytes()).hexdigest(),
        "lag": asdict(lag), "lag_available": lag.available,
        "cache_sha256": _sha256(cache_path),
    }
    records.append(record)
    log.write(json.dumps(record, allow_nan=False) + "\n")
    log.flush()
    print(json.dumps({"id": record["id"], "status": "completed",
                      "known_pixels": record["known_pixels"], "seconds": record["seconds"]},
                     allow_nan=False), flush=True)


def _source_hashes() -> dict[str, str]:
    source_paths = (
        Path("evaluate_cadence_matching_sweep.py"), Path("cadence_matching.py"), Path("hatch_cadence.py"),
        Path("hatch_texture.py"), Path("texture_learning.py"), Path("evaluate.py"),
        Path("evaluate_texture_sweep.py"), Path("evaluate_dino_matching_sweep.py"),
        Path("inference.py"), Path("inference_texture.py"),
        Path("train_cadence_matching.py"),
    )
    return {str(path): _sha256(ROOT / path) for path in source_paths}


def _write_sweep(
    output_dir: Path, records: list[dict[str, Any]], manifest_sha: str, fit_sha: str,
    model_path: Path, model_sha: str, metadata: dict[str, Any],
) -> dict[str, Any]:
    cache_records = sorted(({"id": r["id"], "sha256": r["cache_sha256"]} for r in records), key=lambda r: r["id"])
    cache_map = json.dumps(cache_records, sort_keys=True, separators=(",", ":")).encode()
    report = {
        "schema": "cadence-matching-development-cache/v1",
        "protocol_sha256": _sha256(PROTOCOL), "resource_pilot_sha256": _sha256(RESOURCE_PROOF),
        "evaluation_plan_sha256": _sha256(EVALUATION_PLAN),
        "manifest_sha256": manifest_sha, "development_manifest_sha256": manifest_sha,
        "fit_manifest_sha256": fit_sha, "model_artifact_sha256": model_sha,
        "model_metadata_sha256": _sha256(model_path.with_suffix(".json")),
        "evaluator_sha256": _sha256(ROOT / "evaluate.py"), "source_hashes": _source_hashes(),
        "training_metadata": metadata, "queries": len(records),
        "complete_development_manifest": len(records) == DEVELOPMENT_ROWS,
        "exact_known_only": True, "probability_dir": "probabilities",
        "cache_ids_and_sha256": cache_records,
        "cache_mapping_sha256": hashlib.sha256(cache_map).hexdigest(),
        "records": records,
    }
    (output_dir / "sweep.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def run_development_sweep(
    manifest: Path, fit_manifest: Path, data_root: Path, model_path: Path, output_dir: Path,
) -> dict[str, Any]:
    _check_evaluation_plan()
    _check_resource_pilot(RESOURCE_PROOF)
    model, metadata, model_sha = _load_verified_model(model_path)
    fit_rows = load_manifest(fit_manifest, require_labels=False)
    development_rows = load_manifest(manifest, require_labels=True)
    fit_sha, development_sha = _validate_partitions(
        manifest, fit_manifest, data_root, development_rows, fit_rows, metadata
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    records = _run_queries(development_rows, data_root, model, metadata, model_sha, output_dir)
    report = _write_sweep(output_dir, records, development_sha, fit_sha,
                          model_path, model_sha, metadata)
    report["elapsed_seconds"] = time.monotonic() - started
    (output_dir / "sweep.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fit-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = run_development_sweep(args.manifest, args.fit_manifest, args.data_root, args.model, args.output_dir)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps({key: result[key] for key in (
        "schema", "queries", "complete_development_manifest", "cache_mapping_sha256")}, sort_keys=True))


if __name__ == "__main__":
    main()
