"""Fit the frozen query-conditioned cadence pair classifier on the fit split."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Callable

import cv2
import numpy as np
import sklearn

import cadence_matching
import hatch_cadence
import hatch_texture
import train_texture
import texture_learning
from evaluate import load_manifest, reviewed_domains
from inference import read_drawing
from tools.dino_patch_support import FIT_MANIFEST_SHA256

STRIDE = 4
SAMPLES_PER_CLASS = 512
BASE_SEED = 42
SUPPORT_SHA256 = "dd1b43e144349d1fe8a91f83470996fb5f8b6e6201a3812805ce2ba5d1941af3"
DEFAULT_SUPPORT = Path("artifacts/research-v4/cadence-fit-support.json")
DEFAULT_RESOURCE_PROOF = Path("artifacts/research-v4/cadence-resource-pilot-proof.json")
DEFAULT_ASSET_PROOF = Path("artifacts/research-v4/fit-input-assets.json")
PROTOCOL_PATH = Path("artifacts/research-v4/cadence-feature-protocol.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--support", type=Path, default=DEFAULT_SUPPORT)
    parser.add_argument("--resource-proof", type=Path, default=DEFAULT_RESOURCE_PROOF)
    parser.add_argument("--asset-proof", type=Path, default=DEFAULT_ASSET_PROOF)
    return parser


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_support_evidence(path: Path, manifest_sha256: str) -> dict[str, Any]:
    if file_sha256(path) != SUPPORT_SHA256:
        raise ValueError("cadence fit-support evidence hash mismatch")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != "cadence-fit-support/v1":
        raise ValueError("unsupported cadence fit-support schema")
    if report.get("manifest_sha256") != manifest_sha256:
        raise ValueError("fit-support evidence refers to a different fit manifest")
    if report.get("protocol_sha256") != file_sha256(PROTOCOL_PATH):
        raise ValueError("fit-support evidence refers to a different frozen protocol")
    expected_sources = {
        "hatch_cadence.py": Path("hatch_cadence.py"),
        "tools/cadence_fit_support.py": Path("tools/cadence_fit_support.py"),
    }
    source_hashes = report.get("source_sha256", {})
    for relative, source_path in expected_sources.items():
        if source_hashes.get(relative) != file_sha256(source_path):
            raise ValueError(f"fit-support source hash mismatch: {relative}")
    gate = report.get("support_gate", {})
    if (gate.get("passed") is not True or int(gate.get("stable_dual_queries", 0)) < 30
            or int(gate.get("stable_dual_documents", 0)) < 10):
        raise ValueError("cadence fit-support gate did not pass its frozen minimum")
    return report


def verify_resource_proof(
    path: Path, manifest_sha256: str, support_sha256: str
) -> dict[str, Any]:
    proof = json.loads(path.read_text(encoding="utf-8"))
    if proof.get("schema") != "cadence-resource-pilot-proof/v1" or proof.get("status") != "passed":
        raise ValueError("cadence resource pilot proof is not passed")
    if (proof.get("manifest_sha256") != manifest_sha256
            or proof.get("protocol_sha256") != file_sha256(PROTOCOL_PATH)
            or proof.get("support_sha256") != support_sha256):
        raise ValueError("resource proof does not match the fit manifest, protocol, and support evidence")
    source_hashes = proof.get("source_sha256", {})
    if source_hashes.get("hatch_cadence.py") != file_sha256(Path("hatch_cadence.py")):
        raise ValueError("resource proof cadence helper hash mismatch")
    if (proof.get("process_resource_gate_passed") is not True
            or int(proof.get("process_swap_count", -1)) != 0
            or int(proof.get("peak_process_footprint_bytes", 1 << 63)) >= 16 * 1024**3
            or int(proof.get("system_swapout_delta_pages", -1)) != 0
            or proof.get("feature_bounds_finite_fallback_verified") is not True
            or proof.get("input_pixels_independently_reconciled") is not True
            or proof.get("crop_lag_hash_matches_support") is not True
            or float(proof.get("direct_max_abs_diff", math.inf)) > 1e-6
            or float(proof.get("sparse_dense_max_abs_diff", math.inf)) > 1e-6):
        raise ValueError("cadence resource proof does not pass the frozen process and numerical gates")
    pilot_path = path.with_name("cadence-resource-pilot.json")
    log_path = path.with_name("cadence-resource-pilot.log")
    if (file_sha256(pilot_path) != proof.get("pilot_sha256")
            or file_sha256(log_path) != proof.get("log_sha256")):
        raise ValueError("resource proof pilot or log hash does not match its referenced artifact")
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    if (pilot.get("manifest_sha256") != manifest_sha256
            or pilot.get("protocol_sha256") != file_sha256(PROTOCOL_PATH)
            or pilot.get("support_sha256") != support_sha256
            or pilot.get("all_features_finite") is not True
            or pilot.get("unavailable_fallback_verified") is not True
            or float(pilot.get("direct_max_abs_diff", math.inf)) > 1e-6
            or float(pilot.get("sparse_dense_max_abs_diff", math.inf)) > 1e-6):
        raise ValueError("referenced cadence pilot does not match the passed resource proof")
    pilot_sources = pilot.get("source_sha256", {})
    if pilot_sources.get("hatch_cadence.py") != file_sha256(Path("hatch_cadence.py")):
        raise ValueError("referenced cadence pilot helper hash mismatch")
    minima = np.asarray(pilot.get("feature_min", []), dtype=np.float64)
    maxima = np.asarray(pilot.get("feature_max", []), dtype=np.float64)
    if (minima.shape != (5,) or maxima.shape != (5,) or not np.isfinite(minima).all()
            or not np.isfinite(maxima).all() or np.any(minima < 0) or np.any(maxima > 1)):
        raise ValueError("referenced cadence pilot has invalid feature bounds")
    return proof


def extract_query_training_pairs(
    row: dict[str, Any],
    gray: np.ndarray,
    shared_features: np.ndarray,
    data_root: Path,
    query_seed: int,
    domain_reader: Callable[[dict[str, Any], Path], tuple[np.ndarray, np.ndarray, Any, Any]] = reviewed_domains,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    positive_domain, known, _, _ = domain_reader(row, data_root)
    if positive_domain.shape != gray.shape or known.shape != gray.shape:
        raise ValueError(f"reviewed domains do not match drawing geometry for {row['id']}")
    negative_domain = known & ~positive_domain
    positive_coordinates, negative_coordinates = texture_learning.sample_grid_domain_centers(
        gray.shape[0], gray.shape[1], positive_domain, negative_domain, stride=STRIDE,
        samples_per_class=SAMPLES_PER_CLASS, seed=query_seed,
    )
    query, lag = cadence_matching.query_descriptor(gray, shared_features, tuple(row["query_box"]))
    coordinates = np.concatenate((positive_coordinates, negative_coordinates), axis=0)
    pair_array = cadence_matching.pair_features_for_cells(
        gray, shared_features, coordinates, query, lag
    )
    positive_count, negative_count = len(positive_coordinates), len(negative_coordinates)
    labels = np.concatenate((np.ones(positive_count, dtype=np.int32),
                             np.zeros(negative_count, dtype=np.int32)))
    weights = np.concatenate((_class_weights(positive_count, negative_count),
                              _class_weights(negative_count, positive_count)))
    audit = _query_audit(row, gray, query, lag, query_seed, positive_coordinates,
                         negative_coordinates, labels, weights, pair_array)
    return pair_array, labels, weights, audit


def collect_training_dataset(
    rows: list[dict[str, Any]],
    data_root: Path,
    image_reader: Callable[[Path, dict[str, Any]], np.ndarray] = read_drawing,
    domain_reader: Callable[[dict[str, Any], Path], tuple[np.ndarray, np.ndarray, Any, Any]] = reviewed_domains,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["image"])].append(row)
    pair_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    weight_chunks: list[np.ndarray] = []
    audit: list[dict[str, Any]] = []
    query_counter = 0
    started = time.monotonic()
    for image_rows in grouped.values():
        gray = image_reader(data_root, image_rows[0])
        if gray.dtype != np.uint8 or gray.ndim != 2:
            raise ValueError(f"drawing must be native grayscale uint8: {image_rows[0]['image']}")
        features = hatch_texture.texture_features(gray, stride=STRIDE, feature_set="base")
        drawing_digest = hashlib.sha256(gray.tobytes()).hexdigest()
        for row in image_rows:
            if gray.shape != (row["height"], row["width"]):
                raise ValueError(f"image dimension mismatch for row {row['id']}")
            pairs, labels, weights, query_audit = extract_query_training_pairs(
                row, gray, features, data_root, BASE_SEED + query_counter, domain_reader
            )
            query_counter += 1
            query_audit["drawing_pixel_sha256"] = drawing_digest
            audit.append(query_audit)
            if len(pairs):
                pair_chunks.append(pairs)
                label_chunks.append(labels)
                weight_chunks.append(weights)
        del gray, features
        print(json.dumps({"drawing": image_rows[0]["image"],
                          "queries_in_drawing": len(image_rows),
                          "processed_queries": query_counter,
                          "elapsed_seconds": time.monotonic() - started}), flush=True)
    if not pair_chunks:
        raise ValueError("No cadence training samples were extracted from the fit manifest")
    return (np.concatenate(pair_chunks), np.concatenate(label_chunks),
            np.concatenate(weight_chunks), audit)


def _class_weights(class_count: int, other_count: int) -> np.ndarray:
    if class_count == 0:
        return np.empty(0, dtype=np.float32)
    total_weight = 0.5 if other_count > 0 else 1.0
    return np.full(class_count, total_weight / class_count, dtype=np.float32)


def _query_audit(
    row: dict[str, Any], gray: np.ndarray, query: np.ndarray, lag: hatch_cadence.LagState,
    seed: int, positive: np.ndarray, negative: np.ndarray, labels: np.ndarray,
    weights: np.ndarray, pairs: np.ndarray,
) -> dict[str, Any]:
    x0, y0, x1, y1 = row["query_box"]
    crop = gray[y0:y1, x0:x1]
    return {
        "id": row["id"], "document_id": row["document_id"], "image": row["image"],
        "query_seed": seed, "query_box": [x0, y0, x1, y1],
        "crop_shape": list(crop.shape),
        "crop_pixel_sha256": hashlib.sha256(crop.tobytes()).hexdigest(),
        "lag": asdict(lag), "lag_available": lag.available,
        "query_descriptor_sha256": _array_sha256(query),
        "positive_grid_coordinates": positive.tolist(),
        "negative_grid_coordinates": negative.tolist(),
        "labels": labels.tolist(), "sample_weights": weights.tolist(),
        "pair_features_shape": list(pairs.shape), "pair_features_sha256": _array_sha256(pairs),
    }


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def train_classifier(x: np.ndarray, y: np.ndarray, weights: np.ndarray):
    return train_texture.train_classifier(x, y, weights, max_iter=200, seed=BASE_SEED)


def build_metadata(
    manifest_sha256: str,
    rows: list[dict[str, Any]],
    labels: np.ndarray,
    audit: list[dict[str, Any]],
    support_path: Path,
    resource_path: Path,
    asset_path: Path,
    asset_mapping_sha256: str,
    asset_mapping: dict[str, Any],
    wall_time_seconds: float,
) -> dict[str, Any]:
    base_names = hatch_texture.texture_feature_names("base") + list(hatch_cadence.FEATURE_NAMES)
    resource = json.loads(resource_path.read_text(encoding="utf-8"))
    return {
        "schema": "cadence-pair-classifier/v1",
        "stride": STRIDE,
        "query_descriptor_source": "crop",
        "texture_feature_set": "base",
        "feature_set": "base+cadence",
        "base_feature_names": base_names,
        "pair_feature_names": texture_learning.pair_feature_names(base_names, "crop"),
        "source_manifest_sha256": manifest_sha256,
        "fit_manifest_sha256": manifest_sha256,
        "training_ids": [row["id"] for row in rows],
        "training_documents": list(dict.fromkeys(row["document_id"] for row in rows)),
        "training_coverage": {
            "queries": len(rows), "documents": len({row["document_id"] for row in rows}),
            "total_samples": int(len(labels)), "positive_samples": int(np.sum(labels == 1)),
            "negative_samples": int(np.sum(labels == 0)),
            "stable_lag_queries": sum(bool(record["lag_available"]) for record in audit),
        },
        "params": {"learning_rate": 0.08, "max_leaf_nodes": 31,
                   "l2_regularization": 1.0, "max_iter": 200,
                   "early_stopping": False, "random_state": BASE_SEED},
        "seed": BASE_SEED, "samples_per_class": SAMPLES_PER_CLASS,
        "cadence_protocol_sha256": file_sha256(PROTOCOL_PATH),
        "cadence_algorithm_version": "hatch-cadence-feature-protocol/v1",
        "source_sha256": _implementation_hashes(),
        "support_evidence_sha256": file_sha256(support_path),
        "resource_pilot_sha256": file_sha256(resource_path),
        "resource_pilot_artifact_sha256": resource.get("pilot_sha256"),
        "resource_pilot_log_sha256": resource.get("log_sha256"),
        "fit_asset_proof_sha256": file_sha256(asset_path),
        "fit_asset_mapping_sha256": asset_mapping_sha256,
        "fit_asset_mapping": asset_mapping,
        "input_asset_proof": str(asset_path),
        "query_audit": audit,
        "wall_time_seconds": wall_time_seconds,
        "sklearn_version": sklearn.__version__, "numpy_version": np.__version__,
        "opencv_version": cv2.__version__,
    }


def _implementation_hashes() -> dict[str, str]:
    paths = ("hatch_texture.py", "texture_learning.py", "train_texture.py",
             "hatch_cadence.py", "cadence_matching.py", "train_cadence_matching.py",
             str(PROTOCOL_PATH))
    return {path: file_sha256(Path(path)) for path in paths}


def save_model(model: Any, metadata: dict[str, Any], output: Path) -> str:
    texture_learning.save_texture_model(model, metadata, output)
    digest = file_sha256(output)
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{digest}  {output.name}\n", encoding="utf-8"
    )
    return digest


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    manifest_sha256 = file_sha256(args.manifest)
    if manifest_sha256 != FIT_MANIFEST_SHA256:
        raise ValueError("cadence training requires the frozen fit manifest")
    support = verify_support_evidence(args.support, manifest_sha256)
    resource = verify_resource_proof(args.resource_proof, manifest_sha256, SUPPORT_SHA256)
    rows = load_manifest(args.manifest, require_labels=True)
    train_texture.validate_split_safety(args.manifest, args.data_root, rows)
    if len(rows) != int(support.get("queries", -1)):
        raise ValueError("fit-support query count differs from the pinned manifest")
    from train_dino_matching import verify_fit_assets

    asset_mapping_sha256 = verify_fit_assets(
        args.asset_proof, manifest_sha256, rows, args.data_root
    )
    asset_mapping = json.loads(args.asset_proof.read_text(encoding="utf-8")).get("assets", {})
    started = time.monotonic()
    x, y, weights, audit = collect_training_dataset(rows, args.data_root)
    print(json.dumps({"classifier": "HistGradientBoostingClassifier",
                      "samples": len(y), "features": x.shape[1], "fit_started": True}), flush=True)
    model = train_classifier(x, y, weights)
    print(json.dumps({"classifier": "HistGradientBoostingClassifier",
                      "fit_completed": True}), flush=True)
    metadata = build_metadata(
        manifest_sha256, rows, y, audit, args.support, args.resource_proof,
        args.asset_proof, asset_mapping_sha256, asset_mapping, time.monotonic() - started,
    )
    metadata["resource_pilot_status"] = resource["status"]
    metadata["support_evidence_gate"] = support["support_gate"]
    model_sha256 = save_model(model, metadata, args.output)
    print(json.dumps({"queries": len(rows), "samples": len(y),
                      "model_sha256": model_sha256, "output": str(args.output)}))


if __name__ == "__main__":
    main()
