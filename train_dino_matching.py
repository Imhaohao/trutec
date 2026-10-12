"""Fit the pinned frozen DINO patch-pair classifier on reviewed fit labels."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier

from evaluate import data_path, load_manifest, reviewed_domains
from inference import read_drawing
from tools.dino_patch_support import FIT_MANIFEST_SHA256
from dino_matching import file_sha256
from train_hatchfinder import check_split_safety
from dino_matching import (
    FEATURE_SIGNATURE, MAX_TRAINING_SAMPLES, MODEL_PARAMS, SAMPLES_PER_CLASS,
    extract_tiled_features, grayscale_to_rgb, load_verified_probe, pair_features, query_embedding,
    sample_pure_patch_pixels, sample_weights, save_dino_model,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True, help="Pinned clean local DINOv2 checkout")
    parser.add_argument("--weights", type=Path, required=True, help="Pinned local ViT-S/14 checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="Destination .joblib model")
    parser.add_argument("--support", type=Path, default=Path("artifacts/research-v4/dino-patch-support-fit.json"))
    parser.add_argument("--asset-proof", type=Path, default=Path("artifacts/research-v4/fit-input-assets.json"))
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    return parser


def _manifest_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _query_crop(rgb: np.ndarray, row: dict[str, Any]) -> np.ndarray:
    x0, y0, x1, y1 = row["query_box"]
    crop = rgb[y0:y1, x0:x1]
    if crop.shape[:2] != (y1 - y0, x1 - x0):
        raise ValueError(f"query box crop is invalid for {row['id']}")
    return crop


def collect_fit_pairs(rows: list[dict[str, Any]], data_root: Path, extractor: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ordered_rows = sorted(rows, key=lambda row: (str(row["document_id"]), str(row["id"])))
    sample_seeds = {str(row["id"]): 45 + index for index, row in enumerate(ordered_rows)}
    for row in ordered_rows:
        grouped[str(row["image"])].append(row)
    pair_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    weight_chunks: list[np.ndarray] = []
    sampled_queries = 0
    sampled_documents: set[str] = set()
    sampled_ids: list[str] = []
    skipped_ids: list[str] = []
    dual_ids: list[str] = []
    positive_only_ids: list[str] = []
    audit: list[dict[str, Any]] = []

    for image_rows in grouped.values():
        gray = read_drawing(data_root, image_rows[0])
        rgb = grayscale_to_rgb(gray)
        features = extract_tiled_features(rgb, extractor)
        for row in image_rows:
            if rgb.shape[:2] != (row["height"], row["width"]):
                raise ValueError(f"drawing dimensions do not match fit query {row['id']}")
            positive, known, _, _ = reviewed_domains(row, data_root)
            pos_pixels, neg_pixels = sample_pure_patch_pixels(
                positive, known, sample_seeds[str(row["id"])], SAMPLES_PER_CLASS
            )
            positive_cells = pos_pixels // 14
            negative_cells = neg_pixels // 14
            vectors: list[np.ndarray] = []
            labels: list[np.ndarray] = []
            if len(positive_cells):
                vectors.append(features[positive_cells[:, 0], positive_cells[:, 1]])
                labels.append(np.ones(len(positive_cells), dtype=np.int8))
            if len(negative_cells):
                vectors.append(features[negative_cells[:, 0], negative_cells[:, 1]])
                labels.append(np.zeros(len(negative_cells), dtype=np.int8))
            if not len(positive_cells):
                skipped_ids.append(str(row["id"]))
                continue
            if len(negative_cells):
                dual_ids.append(str(row["id"]))
            else:
                positive_only_ids.append(str(row["id"]))
            query = query_embedding(_query_crop(rgb, row), extractor)
            query_labels = np.concatenate(labels)
            pairs = pair_features(np.concatenate(vectors), query)
            weights = sample_weights(query_labels)
            pair_chunks.append(pairs)
            label_chunks.append(query_labels)
            weight_chunks.append(weights)
            sampled_queries += 1
            sampled_documents.add(str(row["document_id"]))
            sampled_ids.append(str(row["id"]))
            audit.append({
                "query_id": str(row["id"]), "document_id": str(row["document_id"]),
                "seed": sample_seeds[str(row["id"])],
                "positive_native_pixels": pos_pixels.tolist(),
                "negative_native_pixels": neg_pixels.tolist(),
                "positive_patch_indices": positive_cells.tolist(),
                "negative_patch_indices": negative_cells.tolist(),
                "class_counts": {"positive": len(pos_pixels), "negative": len(neg_pixels)},
                "class_weights": {
                    "positive": 0.5 / len(pos_pixels) if len(neg_pixels) else 1.0 / len(pos_pixels),
                    "negative": 0.5 / len(neg_pixels) if len(neg_pixels) else 0.0,
                },
                "positive_patch_centers_yx": _patch_centers(positive_cells, row),
                "negative_patch_centers_yx": _patch_centers(negative_cells, row),
            })
        del rgb, features

    if not pair_chunks:
        raise ValueError("fit manifest yielded no pure labeled patch pairs")
    dual_id_set = set(dual_ids)
    dual_documents = {str(row["document_id"]) for row in ordered_rows if str(row["id"]) in dual_id_set}
    if len(dual_ids) < 30 or len(dual_documents) < 10:
        raise ValueError("sampled fit rows fail the minimum 30 dual-class queries across 10 documents")
    x, y, weights = np.concatenate(pair_chunks), np.concatenate(label_chunks), np.concatenate(weight_chunks)
    if len(y) > MAX_TRAINING_SAMPLES:
        raise ValueError(f"training sample count {len(y)} exceeds the pinned cap {MAX_TRAINING_SAMPLES}")
    if len(np.unique(y)) != 2:
        raise ValueError("DINO classifier fit requires both positive and negative samples")
    return x, y, weights, {
        "queries": sampled_queries, "documents": len(sampled_documents),
        "sampling_audit": sorted(audit, key=lambda record: (record["document_id"], record["query_id"])),
        "sampled_query_ids": sorted(sampled_ids),
        "dual_class_query_ids": sorted(dual_ids),
        "positive_only_query_ids": sorted(positive_only_ids),
        "skipped_query_ids": sorted(skipped_ids),
    }


def fit_classifier(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(**MODEL_PARAMS)
    model.fit(x, y, sample_weight=weights)
    return model


def build_metadata(
    rows: list[dict[str, Any]], manifest_sha256: str, pins: dict[str, Any],
    coverage: dict[str, Any], y: np.ndarray, elapsed: float,
) -> dict[str, Any]:
    fit_ids = sorted(str(row["id"]) for row in rows)
    fit_documents = sorted({str(row["document_id"]) for row in rows})
    return {
        "schema": "dino-patch-pair-classifier/v1",
        "feature_signature": FEATURE_SIGNATURE,
        "patch_size": 14,
        "input_tile": 896,
        "retained_core": 672,
        "context_margin_each_side": 112,
        "query_limit_pixels_per_side": 896,
        "query_signature": "mean-L2-normalized-patch-vector-from-exact-query-crop",
        "feature_protocol": "DINOv2 ViT-S/14; native grayscale replicated to RGB; white-pad bottom/right",
        "classifier_protocol_sha256": file_sha256(Path("artifacts/research-v4/dino-classifier-protocol.json")),
        "implementation_sha256": {
            "dino_matching.py": file_sha256(Path("dino_matching.py")),
            "train_dino_matching.py": file_sha256(Path(__file__)),
            "tools/dino_feature_probe.py": file_sha256(Path("tools/dino_feature_probe.py")),
        },
        "fit_pair_signature": "[l2_local_384,l2_local_minus_l2_query_384,dot_product_1]",
        "fit_manifest_sha256": manifest_sha256,
        "training_ids": fit_ids,
        "training_documents": fit_documents,
        "sampled_training_coverage": {
            **{key: value for key, value in coverage.items() if key != "sampling_audit"},
            "samples": len(y),
            "positive_samples": int((y == 1).sum()),
            "negative_samples": int((y == 0).sum()),
            "max_per_class_per_query": SAMPLES_PER_CLASS,
            "positive_only_queries_retained": True,
        },
        "pair_weighting": "per-query balanced 0.5 per class; positive-only queries weight 1.0",
        "sampling_audit": coverage["sampling_audit"],
        "classifier_params": MODEL_PARAMS,
        "seed": 45,
        **pins,
        "wall_time_seconds": elapsed,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
    }


def _patch_centers(indices: np.ndarray, row: dict[str, Any]) -> list[list[int]]:
    return [[min(int(y) * 14 + 7, int(row["height"]) - 1), min(int(x) * 14 + 7, int(row["width"]) - 1)] for y, x in indices]


def verify_support_evidence(path: Path, manifest_sha256: str) -> dict[str, Any]:
    if file_sha256(path) != "8973e05d4599809f7f80206131bc71cd213cf7c7baaae648f4585aafdeed9201":
        raise ValueError("fit patch-support evidence hash mismatch")
    report = json.loads(path.read_text(encoding="utf-8"))
    gate = report.get("support_gate", {})
    if report.get("manifest_sha256") != manifest_sha256 or not gate.get("passed"):
        raise ValueError("fit patch-support evidence does not match the pinned manifest or pass its gate")
    if int(report.get("dual_pure_queries", 0)) < 30 or int(report.get("dual_pure_documents", 0)) < 10:
        raise ValueError("fit patch-support evidence is below the locked dual-class support minimum")
    return report



def verify_fit_assets(path: Path, manifest_sha256: str, rows: list[dict[str, Any]], data_root: Path) -> str:
    expected_proof_sha = "a38903333b8fefad9869f9b573880533163d3dc7b0859ba044ca9685ae2ab960"
    expected_mapping_sha = "3ece46714ba0a219c3b8699273fc2fd490dae56a9363a198d185ae7e809b2d66"
    if file_sha256(path) != expected_proof_sha:
        raise ValueError("fit input asset proof file hash mismatch")
    proof = json.loads(path.read_text(encoding="utf-8"))
    if proof.get("manifest_sha256") != manifest_sha256:
        raise ValueError("fit input asset proof refers to a different manifest")
    references = set()
    for row in rows:
        references.add(str(row["image"]))
        for key, relative in row.get("labels", {}).items():
            if key.endswith("_mask"):
                references.add(str(relative))
    actual_assets = {}
    for relative in sorted(references):
        asset_path = data_path(data_root, relative)
        actual_assets[relative] = {"sha256": file_sha256(asset_path), "bytes": asset_path.stat().st_size}
    canonical = json.dumps(actual_assets, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()
    if digest != expected_mapping_sha or digest != proof.get("asset_digest_sha256"):
        raise ValueError("fit image/label asset mapping does not match the pinned evidence")
    if actual_assets != proof.get("assets"):
        raise ValueError("fit image/label assets differ from the pinned byte mapping")
    return digest


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    manifest_sha = _manifest_hash(args.manifest)
    if manifest_sha != FIT_MANIFEST_SHA256:
        raise SystemExit(f"fit manifest SHA-256 mismatch: expected {FIT_MANIFEST_SHA256}, got {manifest_sha}")
    rows = load_manifest(args.manifest, require_labels=True)
    check_split_safety(rows, args.data_root)
    support = verify_support_evidence(args.support, manifest_sha)
    asset_mapping_sha256 = verify_fit_assets(args.asset_proof, manifest_sha, rows, args.data_root)
    extractor, pins = load_verified_probe(args.source, args.weights, args.device)
    if extractor.model.training or any(parameter.requires_grad for parameter in extractor.model.parameters()):
        raise ValueError("DINOv2 must be in eval mode with every parameter frozen")
    started = time.monotonic()
    x, y, weights, coverage = collect_fit_pairs(rows, args.data_root, extractor)
    model = fit_classifier(x, y, weights)
    metadata = build_metadata(rows, manifest_sha, pins, coverage, y, time.monotonic() - started)
    metadata["support_evidence_sha256"] = file_sha256(args.support)
    metadata["support_evidence_path"] = str(args.support)
    metadata["fit_asset_mapping_sha256"] = asset_mapping_sha256
    metadata["fit_asset_proof_sha256"] = file_sha256(args.asset_proof)
    metadata["fit_asset_proof"] = str(args.asset_proof)
    metadata["support_evidence_dual_queries"] = int(support["dual_pure_queries"])
    metadata["support_evidence_dual_documents"] = int(support["dual_pure_documents"])
    metadata["torch_version"] = str(extractor.torch.__version__)
    metadata["dino_eval_mode"] = not bool(extractor.model.training)
    metadata["dino_all_parameters_frozen"] = all(
        not parameter.requires_grad for parameter in extractor.model.parameters()
    )
    save_dino_model(model, metadata, args.output)
    print(json.dumps({"queries": len(rows), "samples": len(y), "output": str(args.output)}))


if __name__ == "__main__":
    main()
