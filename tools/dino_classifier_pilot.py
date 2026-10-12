"""Run one discarded fit-only DINO/HGB resource and persistence pilot."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import tempfile
import time
from typing import Any, Sequence

import numpy as np
import sklearn

from dino_matching import (
    CORE_SIZE, INPUT_TILE, MAX_TRAINING_SAMPLES, MODEL_PARAMS, PATCH_SIZE,
    SAMPLES_PER_CLASS, CONTEXT_MARGIN, extract_tiled_features, file_sha256,
    grayscale_to_rgb, load_dino_model, load_verified_probe, pair_features,
    query_embedding, sample_pure_patch_pixels, sample_weights, save_dino_model,
    tile_core_slices,
)
from evaluate import load_manifest, reviewed_domains
from inference import read_drawing
from train_dino_matching import (
    FIT_MANIFEST_SHA256, _manifest_hash, _patch_centers, _query_crop,
    fit_classifier, verify_fit_assets, verify_support_evidence,
)
from train_hatchfinder import check_split_safety


class CountingExtractor:
    def __init__(self, extractor: Any) -> None:
        self.extractor = extractor
        self.tile_shapes: list[list[int]] = []

    @property
    def model(self) -> Any:
        return self.extractor.model

    def extract(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.tile_shapes.append([int(rgb.shape[0]), int(rgb.shape[1])])
        return self.extractor.extract(rgb)

    def query_embedding(self, rgb: np.ndarray) -> np.ndarray:
        return self.extractor.query_embedding(rgb)


def select_pilot_drawing(
    rows: list[dict[str, Any]], support: dict[str, Any]
) -> list[dict[str, Any]]:
    eligible = {
        str(item["query_id"])
        for item in support["per_query"]
        if int(item["pure_positive_patch_cells"]) > 0
        and int(item["pure_negative_patch_cells"]) > 0
    }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row["id"]) in eligible:
            grouped[str(row["image"])].append(row)
    if not grouped:
        raise ValueError("fit support evidence has no dual-pure drawing for the pilot")
    chosen_image = min(
        grouped,
        key=lambda image: (
            -int(grouped[image][0]["height"]) * int(grouped[image][0]["width"]), image
        ),
    )
    return sorted(grouped[chosen_image], key=lambda row: (str(row["document_id"]), str(row["id"])))


def _is_frozen(extractor: Any) -> bool:
    return not extractor.model.training and all(
        not parameter.requires_grad for parameter in extractor.model.parameters()
    )


def _state_sha256(extractor: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(extractor.model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def collect_drawing_pairs(
    rows: list[dict[str, Any]], data_root: Path, extractor: CountingExtractor
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]], list[list[int]]]:
    gray = read_drawing(data_root, rows[0])
    rgb = grayscale_to_rgb(gray)
    feature_grid = extract_tiled_features(rgb, extractor)
    pair_rows: list[np.ndarray] = []
    label_rows: list[np.ndarray] = []
    weight_rows: list[np.ndarray] = []
    audit: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if rgb.shape[:2] != (int(row["height"]), int(row["width"])):
            raise ValueError(f"fit drawing dimensions do not match query {row['id']}")
        positive, known, _, _ = reviewed_domains(row, data_root)
        pos, neg = sample_pure_patch_pixels(positive, known, 45 + index, SAMPLES_PER_CLASS)
        if not len(pos) or not len(neg):
            raise ValueError(f"support evidence and reviewed domains disagree for {row['id']}")
        pos_cells, neg_cells = pos // PATCH_SIZE, neg // PATCH_SIZE
        query = query_embedding(_query_crop(rgb, row), extractor)
        vectors = np.concatenate((
            feature_grid[pos_cells[:, 0], pos_cells[:, 1]],
            feature_grid[neg_cells[:, 0], neg_cells[:, 1]],
        ))
        labels = np.concatenate((np.ones(len(pos), dtype=np.int8), np.zeros(len(neg), dtype=np.int8)))
        pair_rows.append(pair_features(vectors, query))
        label_rows.append(labels)
        weight_rows.append(sample_weights(labels))
        audit.append({
            "query_id": str(row["id"]), "document_id": str(row["document_id"]),
            "seed": 45 + index,
            "positive_pixels_yx": pos.tolist(), "negative_pixels_yx": neg.tolist(),
            "positive_patch_indices_yx": pos_cells.tolist(), "negative_patch_indices_yx": neg_cells.tolist(),
            "positive_patch_centers_yx": _patch_centers(pos_cells, row),
            "negative_patch_centers_yx": _patch_centers(neg_cells, row),
            "positive_sample_count": len(pos), "negative_sample_count": len(neg),
            "per_class_weight": {"positive": 0.5 / len(pos), "negative": 0.5 / len(neg)},
        })
    x, y, weights = np.concatenate(pair_rows), np.concatenate(label_rows), np.concatenate(weight_rows)
    if len(y) > MAX_TRAINING_SAMPLES or not np.isfinite(x).all() or not np.isfinite(weights).all():
        raise ValueError("pilot pair matrix exceeded its cap or contains non-finite values")
    del feature_grid, rgb, gray
    return x, y, weights, audit, extractor.tile_shapes.copy()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/splits/fit.json"))
    parser.add_argument("--data-root", type=Path, default=Path("dataset"))
    parser.add_argument("--support", type=Path, default=Path("artifacts/research-v4/dino-patch-support-fit.json"))
    parser.add_argument("--asset-proof", type=Path, default=Path("artifacts/research-v4/fit-input-assets.json"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def run_pilot(args: argparse.Namespace) -> dict[str, Any]:
    manifest_sha = _manifest_hash(args.manifest)
    if manifest_sha != FIT_MANIFEST_SHA256:
        raise ValueError("pilot requires the pinned fit manifest")
    rows = load_manifest(args.manifest, require_labels=True)
    check_split_safety(rows, args.data_root)
    support = verify_support_evidence(args.support, manifest_sha)
    asset_mapping_sha = verify_fit_assets(args.asset_proof, manifest_sha, rows, args.data_root)
    selected = select_pilot_drawing(rows, support)

    extractor, pins = load_verified_probe(args.source, args.weights, args.device)
    frozen_before = _is_frozen(extractor)
    if not frozen_before:
        raise ValueError("DINO parameters must be frozen before pilot extraction")
    state_before = _state_sha256(extractor)
    counting = CountingExtractor(extractor)
    started = time.monotonic()
    x, y, weights, audit, tile_shapes = collect_drawing_pairs(selected, args.data_root, counting)
    trained = fit_classifier(x, y, weights)
    state_after = _state_sha256(extractor)
    if not _is_frozen(extractor) or state_after != state_before:
        raise ValueError("DINO parameters changed during the classifier pilot")
    pair_probabilities = trained.predict_proba(x)
    if not np.isfinite(pair_probabilities).all():
        raise ValueError("pilot classifier produced non-finite training-pair probabilities")

    pilot_metadata = {
        "feature_signature": ["l2_local_384", "l2_local_minus_l2_query_384", "dot_product_1"],
        "fit_manifest_sha256": manifest_sha,
        "pilot_only": True,
    }
    with tempfile.TemporaryDirectory(prefix="dino-classifier-pilot-") as temp_dir:
        artifact_path = Path(temp_dir) / "discarded.joblib"
        save_dino_model(trained, pilot_metadata, artifact_path)
        loaded, _ = load_dino_model(artifact_path)
        loaded_probabilities = loaded.predict_proba(x)
    if not np.array_equal(loaded_probabilities, pair_probabilities):
        raise ValueError("discarded HGB persistence changed pilot probabilities")

    height, width = int(selected[0]["height"]), int(selected[0]["width"])
    largest_fit_drawing = max(
        rows, key=lambda row: (
            int(row["height"]) * int(row["width"]),
            str(row["image"]),
        )
    )
    tile_count = len(tile_core_slices((height + 13) // 14, (width + 13) // 14))
    return {
        "schema": "dino-fit-classifier-pilot/v1",
        "status": "discarded_single_fit_drawing_pilot",
        "no_development_or_public_rows": True,
        "manifest_sha256": manifest_sha,
        "support_sha256": file_sha256(args.support),
        "support_gate": support["support_gate"],
        "fit_asset_proof_sha256": file_sha256(args.asset_proof),
        "fit_asset_mapping_sha256": asset_mapping_sha,
        "selected_image": str(selected[0]["image"]),
        "selected_query_ids": [str(row["id"]) for row in selected],
        "selected_document_ids": sorted({str(row["document_id"]) for row in selected}),
        "drawing_shape": [height, width],
        "largest_fit_drawing_id": str(largest_fit_drawing["image"]),
        "fit_manifest_max_drawing_shape": [
            int(largest_fit_drawing["height"]), int(largest_fit_drawing["width"])
        ],
        "planned_tile_count": tile_count,
        "actual_tile_count": len(tile_shapes),
        "max_tile_shape": [max(shape[0] for shape in tile_shapes), max(shape[1] for shape in tile_shapes)],
        "tile_shapes": tile_shapes,
        "tile_contract": {"input": INPUT_TILE, "core": CORE_SIZE, "context_margin": CONTEXT_MARGIN},
        "dino_frozen_before_and_after": _is_frozen(extractor),
        "dino_state_sha256_before": state_before,
        "dino_state_sha256_after": state_after,
        "feature_grid_width": x.shape[1],
        "pair_matrix_shape": list(x.shape),
        "sample_count": len(y),
        "positive_samples": int((y == 1).sum()),
        "negative_samples": int((y == 0).sum()),
        "all_sampled_pixels_are_known_fit_labels": True,
        "sampling_audit": audit,
        "classifier_parameters": MODEL_PARAMS,
        "classifier_probability_finite": True,
        "classifier_roundtrip_exact": True,
        "temporary_model_saved_outside_project": True,
        "feature_and_pair_values_finite": bool(np.isfinite(x).all()),
        "elapsed_seconds": time.monotonic() - started,
        "sklearn_version": sklearn.__version__,
        **pins,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = run_pilot(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
