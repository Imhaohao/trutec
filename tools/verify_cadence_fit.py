"""Verify a frozen cadence-HGB fit and its fit-only sampling evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import hatch_cadence
import hatch_texture
import texture_learning
import train_cadence_matching
import train_texture
from evaluate import EvaluationError, load_manifest, reviewed_domains
from inference import read_drawing
from tools.dino_patch_support import FIT_MANIFEST_SHA256
from train_dino_matching import verify_fit_assets

FIT_ASSET_MAPPING_SHA256 = "3ece46714ba0a219c3b8699273fc2fd490dae56a9363a198d185ae7e809b2d66"
MODEL_PARAMS = {
    "learning_rate": 0.08,
    "max_leaf_nodes": 31,
    "l2_regularization": 1.0,
    "max_iter": 200,
    "early_stopping": False,
    "random_state": 42,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise EvaluationError(f"Cadence fit {label} mismatch")


def _json_value(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise EvaluationError("Cadence metadata contains a non-JSON value") from exc


def _metadata(model_path: Path) -> tuple[Any, dict[str, Any]]:
    _verify_artifact_sha_before_load(model_path)
    try:
        model, metadata = texture_learning.load_texture_model(model_path)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"Cannot load cadence model artifact: {exc}") from exc
    if type(model) is not HistGradientBoostingClassifier:
        raise EvaluationError("Estimator is not the locked HistGradientBoostingClassifier")
    metadata_sidecar = model_path.with_suffix(".json")
    try:
        sidecar_metadata = json.loads(metadata_sidecar.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvaluationError("Cadence model metadata sidecar is missing or invalid") from exc
    require_equal(_json_value(sidecar_metadata), _json_value(metadata),
                  "embedded and sidecar metadata")
    return model, metadata


def _verify_artifact_sha_before_load(model_path: Path) -> None:
    if model_path.suffix != ".joblib":
        raise EvaluationError("Expected a trusted .joblib cadence artifact")
    sidecar = model_path.with_suffix(model_path.suffix + ".sha256")
    try:
        line = sidecar.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise EvaluationError("Cadence model SHA-256 sidecar is missing") from exc
    match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
    if match is None or match.group(2) != model_path.name or match.group(1) != sha256(model_path):
        raise EvaluationError("Cadence model SHA-256 sidecar does not match the artifact")


def _verify_model(model: Any, metadata: dict[str, Any], model_path: Path) -> dict[str, Any]:
    require_equal(model.classes_.tolist(), [0, 1], "class ordering")
    require_equal(int(model.n_features_in_), 204, "feature count")
    params = model.get_params(deep=False)
    for key, value in MODEL_PARAMS.items():
        require_equal(params.get(key), value, f"classifier parameter {key}")
    require_equal(metadata.get("params"), MODEL_PARAMS, "metadata parameters")
    require_equal(metadata.get("schema"), "cadence-pair-classifier/v1", "artifact schema")
    require_equal(metadata.get("feature_set"), "base+cadence", "feature set")
    require_equal(metadata.get("query_descriptor_source"), "crop", "query source")
    require_equal(metadata.get("stride"), 4, "stride")
    require_equal(metadata.get("texture_feature_set"), "base", "texture feature set")
    base_names = hatch_texture.texture_feature_names("base") + list(hatch_cadence.FEATURE_NAMES)
    require_equal(metadata.get("base_feature_names"), base_names, "base feature signature")
    require_equal(metadata.get("pair_feature_names"), texture_learning.pair_feature_names(base_names, "crop"), "pair feature signature")
    require_equal(int(model.n_iter_), MODEL_PARAMS["max_iter"], "iteration count")
    nodes = [tree.nodes["value"] for stage in model._predictors for tree in stage]
    if (not nodes or any(not np.isfinite(value).all() for value in nodes)
            or not np.isfinite(model._baseline_prediction).all()):
        raise EvaluationError("Classifier tree values or baseline contain non-finite data")
    return {"model_sha256": sha256(model_path), "features": 204, "iterations": int(model.n_iter_)}


def _serialized_lag(lag: hatch_cadence.LagState) -> dict[str, Any]:
    def estimate(item: hatch_cadence.QuadrantEstimate | None) -> Any:
        if item is None:
            return None
        return {"period": item.period, "angle": item.angle, "power": item.power,
                "quadrant": item.quadrant}

    return {"period": lag.period, "angle": lag.angle,
            "estimates": [estimate(item) for item in lag.estimates]}


def _verify_query_audit(rows: list[dict[str, Any]], data_root: Path,
                        audit: Any) -> dict[str, int]:
    if not isinstance(audit, list) or len(audit) != len(rows):
        raise EvaluationError("Cadence sampling audit does not cover every fit query")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["image"]), []).append(row)
    expected_order = [str(row["id"]) for group in grouped.values() for row in group]
    if [str(record.get("id")) for record in audit if isinstance(record, dict)] != expected_order:
        raise EvaluationError("Cadence sampling audit does not preserve grouped manifest order")
    records = {str(record.get("id")): record for record in audit if isinstance(record, dict)}
    if len(records) != len(audit) or set(records) != {str(row["id"]) for row in rows}:
        raise EvaluationError("Cadence sampling audit IDs are missing, duplicated, or unexpected")
    query_counter = 0
    totals = {"positive": 0, "negative": 0, "stable_lag_queries": 0}
    for group in grouped.values():
        gray = read_drawing(data_root, group[0])
        drawing_hash = hashlib.sha256(gray.tobytes()).hexdigest()
        for row in group:
            record = records[str(row["id"])]
            seed = 42 + query_counter
            query_counter += 1
            _verify_query_record(row, record, gray, drawing_hash, data_root, seed, totals)
    return totals


def _verify_query_record(row: dict[str, Any], record: dict[str, Any], gray: np.ndarray,
                         drawing_hash: str, data_root: Path, seed: int,
                         totals: dict[str, int]) -> None:
    ident = str(row["id"])
    if gray.shape != (row["height"], row["width"]):
        raise EvaluationError(f"Fit drawing geometry differs for {ident}")
    require_equal(record.get("document_id"), row["document_id"], f"document ID for {ident}")
    require_equal(record.get("image"), row["image"], f"image path for {ident}")
    require_equal(record.get("query_seed"), seed, f"seed schedule for {ident}")
    require_equal(record.get("query_box"), row["query_box"], f"query box for {ident}")
    require_equal(record.get("drawing_pixel_sha256"), drawing_hash, f"drawing pixel hash for {ident}")
    x0, y0, x1, y1 = row["query_box"]
    crop = gray[y0:y1, x0:x1]
    require_equal(record.get("crop_shape"), list(crop.shape), f"crop shape for {ident}")
    require_equal(record.get("crop_pixel_sha256"), hashlib.sha256(crop.tobytes()).hexdigest(), f"crop pixel hash for {ident}")
    lag = hatch_cadence.estimate_query_lag(crop)
    require_equal(record.get("lag_available"), lag.available, f"lag availability for {ident}")
    require_equal(_json_value(record.get("lag")), _json_value(_serialized_lag(lag)),
                  f"lag diagnostics for {ident}")
    positive, known, _, _ = reviewed_domains(row, data_root)
    positive_coords, negative_coords = texture_learning.sample_grid_domain_centers(
        *gray.shape, positive, known & ~positive, stride=4, samples_per_class=512, seed=seed
    )
    _verify_coordinates(record, "positive", positive_coords, row)
    _verify_coordinates(record, "negative", negative_coords, row)
    labels = np.concatenate((np.ones(len(positive_coords), dtype=np.int32),
                             np.zeros(len(negative_coords), dtype=np.int32)))
    positive_weight = (0.5 if len(negative_coords) else 1.0) / len(positive_coords) if len(positive_coords) else 0.0
    negative_weight = ((0.5 if len(positive_coords) else 1.0) / len(negative_coords)
                       if len(negative_coords) else 0.0)
    weights = np.concatenate((np.full(len(positive_coords), positive_weight, dtype=np.float32),
                              np.full(len(negative_coords), negative_weight, dtype=np.float32)))
    if record.get("labels") != labels.tolist() or record.get("sample_weights") != weights.tolist():
        raise EvaluationError(f"Cadence labels or weights differ for {ident}")
    require_equal(record.get("pair_features_shape"), [len(labels), 204], f"pair shape for {ident}")
    totals["positive"] += len(positive_coords)
    totals["negative"] += len(negative_coords)
    totals["stable_lag_queries"] += int(lag.available)


def _verify_coordinates(record: dict[str, Any], cls: str, expected: np.ndarray,
                        row: dict[str, Any]) -> None:
    raw = record.get(cls + "_grid_coordinates")
    actual = np.empty((0, 2), dtype=np.int32) if isinstance(raw, list) and not raw else np.asarray(raw)
    if actual.shape != expected.shape or actual.dtype.kind not in "iu" or not np.array_equal(actual, expected):
        raise EvaluationError(f"Recomputed {cls} sample coordinates differ for {row['id']}")


def verify_fit(model_path: Path, manifest_path: Path, data_root: Path,
               support_path: Path, resource_path: Path, asset_path: Path) -> dict[str, Any]:
    manifest_sha = sha256(manifest_path)
    require_equal(manifest_sha, FIT_MANIFEST_SHA256, "manifest SHA-256")
    rows = load_manifest(manifest_path, require_labels=True)
    train_texture.validate_split_safety(manifest_path, data_root, rows)
    model, metadata = _metadata(model_path)
    model_result = _verify_model(model, metadata, model_path)
    require_equal(metadata.get("fit_manifest_sha256"), FIT_MANIFEST_SHA256, "fit manifest pin")
    require_equal(metadata.get("source_manifest_sha256"), manifest_sha, "source manifest hash")
    require_equal(metadata.get("seed"), 42, "base seed")
    require_equal(metadata.get("samples_per_class"), 512, "sample cap")
    require_equal(metadata.get("cadence_algorithm_version"), "hatch-cadence-feature-protocol/v1", "cadence algorithm version")
    require_equal(metadata.get("training_ids"), [row["id"] for row in rows], "training row order")
    documents = list(dict.fromkeys(row["document_id"] for row in rows))
    require_equal(metadata.get("training_documents"), documents, "training document order")
    asset_mapping_sha = verify_fit_assets(asset_path, manifest_sha, rows, data_root)
    require_equal(asset_mapping_sha, FIT_ASSET_MAPPING_SHA256, "pinned fit asset mapping")
    require_equal(metadata.get("fit_asset_mapping_sha256"), asset_mapping_sha, "fit asset mapping")
    asset_proof = json.loads(asset_path.read_text(encoding="utf-8"))
    require_equal(metadata.get("fit_asset_mapping"), asset_proof.get("assets"), "recorded fit asset mapping")
    require_equal(metadata.get("support_evidence_sha256"), sha256(support_path), "support evidence hash")
    require_equal(metadata.get("resource_pilot_sha256"), sha256(resource_path), "resource proof hash")
    require_equal(metadata.get("fit_asset_proof_sha256"), sha256(asset_path), "asset proof hash")
    support = train_cadence_matching.verify_support_evidence(support_path, manifest_sha)
    train_cadence_matching.verify_resource_proof(resource_path, manifest_sha,
                                                train_cadence_matching.SUPPORT_SHA256)
    totals = _verify_query_audit(rows, data_root, metadata.get("query_audit"))
    coverage = metadata.get("training_coverage", {})
    require_equal(coverage.get("queries"), len(rows), "training query coverage")
    require_equal(coverage.get("documents"), len(documents), "training document coverage")
    require_equal(coverage.get("positive_samples"), totals["positive"], "positive sample total")
    require_equal(coverage.get("negative_samples"), totals["negative"], "negative sample total")
    require_equal(coverage.get("total_samples"), totals["positive"] + totals["negative"], "total sample count")
    require_equal(coverage.get("stable_lag_queries"), totals["stable_lag_queries"], "stable lag count")
    _verify_sources(metadata)
    return {**model_result, **totals, "queries": len(rows),
            "documents": len(documents), "fit_asset_mapping_sha256": asset_mapping_sha,
            "support_gate_passed": support["support_gate"]["passed"]}


def _verify_sources(metadata: dict[str, Any]) -> None:
    protocol_path = Path("artifacts/research-v4/cadence-feature-protocol.json")
    require_equal(metadata.get("cadence_protocol_sha256"), sha256(ROOT / protocol_path), "protocol SHA-256")
    paths = ("hatch_texture.py", "texture_learning.py", "train_texture.py",
             "hatch_cadence.py", "cadence_matching.py", "train_cadence_matching.py",
             str(protocol_path))
    expected = {relative: sha256(ROOT / relative) for relative in paths}
    require_equal(metadata.get("source_sha256"), expected, "source hashes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/splits/fit.json"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--support", type=Path, default=train_cadence_matching.DEFAULT_SUPPORT)
    parser.add_argument("--resource-proof", type=Path, default=train_cadence_matching.DEFAULT_RESOURCE_PROOF)
    parser.add_argument("--asset-proof", type=Path, default=train_cadence_matching.DEFAULT_ASSET_PROOF)
    args = parser.parse_args()
    print(json.dumps(verify_fit(args.model, args.manifest, args.data_root,
                                args.support, args.resource_proof, args.asset_proof),
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
