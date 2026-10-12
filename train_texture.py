"""Train a query-conditioned hatch texture classifier on reviewed public training pixels."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time
from typing import Any, cast

import cv2
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
import sklearn

from evaluate import EvaluationError, data_path, load_manifest
import hatch_texture
from texture_learning import (
    pair_feature_names,
    pair_features,
    sample_grid_domain_centers,
    save_texture_model,
)

DEFAULT_STRIDE: int = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path, help="Path to training manifest")
    parser.add_argument("--data-root", required=True, type=Path, help="Path to dataset root")
    parser.add_argument("--output", required=True, type=Path, help="Destination .joblib model path")
    parser.add_argument("--samples-per-class", type=int, default=512, help="Max grid samples per class per query")
    parser.add_argument("--max-iter", type=int, default=200, help="Max iterations for HistGradientBoosting")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    parser.add_argument(
        "--query-source",
        choices=("drawing", "crop"),
        default="drawing",
        help="Query descriptor source. drawing is the legacy full-image descriptor.",
    )
    parser.add_argument(
        "--features",
        choices=("base", "variance8"),
        default="base",
        help="Texture feature set (base or variance8). base is the legacy 46-channel descriptor.",
    )
    return parser.parse_args()


def get_validation_document_ids(data_root: Path) -> set[str]:
    """Extract document IDs from val.json strictly for split verification (never reading labels)."""
    candidates = [data_root / "val.json", Path("dataset/val.json")]
    for candidate in candidates:
        if candidate.is_file():
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            examples = payload.get("examples", payload) if isinstance(payload, dict) else payload
            if isinstance(examples, list):
                return {
                    row["document_id"]
                    for row in examples
                    if isinstance(row, dict) and "document_id" in row
                }
    return set()


def validate_split_safety(manifest_path: Path, data_root: Path, rows: list[dict[str, Any]]) -> None:
    """Enforce strict isolation between training examples and validation documents."""
    resolved = manifest_path.resolve()
    val_candidates = [(data_root / "val.json").resolve(), Path("dataset/val.json").resolve()]
    if resolved.name == "val.json" or any(resolved == c for c in val_candidates if c.exists()):
        raise EvaluationError(f"Training must not use validation manifest: {manifest_path}")

    val_doc_ids = get_validation_document_ids(data_root)
    if not val_doc_ids:
        return

    for row in rows:
        doc_id = row.get("document_id")
        if doc_id in val_doc_ids:
            raise EvaluationError(
                f"Training example {row.get('id')} shares validation document ID: {doc_id}"
            )


def extract_query_samples(
    row: dict[str, Any],
    features: np.ndarray,
    gray_shape: tuple[int, int],
    data_root: Path,
    stride: int,
    samples_per_class: int,
    query_seed: int,
    *,
    gray: np.ndarray,
    query_source: str = "drawing",
    feature_set: str = "base",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample reviewed positive and negative pairs with per-query balanced weights."""
    from evaluate import reviewed_domains

    pos_domain, known, _, _ = reviewed_domains(row, data_root)
    neg_domain = known & ~pos_domain

    pos_coords, neg_coords = sample_grid_domain_centers(
        gray_shape[0],
        gray_shape[1],
        pos_domain,
        neg_domain,
        stride=stride,
        samples_per_class=samples_per_class,
        seed=query_seed,
    )

    if feature_set == "base":
        q_desc = hatch_texture.descriptor_for_query(
            gray, features, _query_box(row), stride, query_source
        )
    else:
        q_desc = hatch_texture.descriptor_for_query(
            gray, features, _query_box(row), stride, query_source, feature_set=feature_set
        )

    pair_list: list[np.ndarray] = []
    label_list: list[np.ndarray] = []
    weight_list: list[np.ndarray] = []

    n_pos = len(pos_coords)
    n_neg = len(neg_coords)

    if n_pos > 0:
        pos_vecs = features[pos_coords[:, 0], pos_coords[:, 1]]
        pair_list.append(pair_features(pos_vecs, q_desc))
        label_list.append(np.ones(n_pos, dtype=np.int32))
        pos_weight = 0.5 / n_pos if n_neg > 0 else 1.0 / n_pos
        weight_list.append(np.full(n_pos, pos_weight, dtype=np.float32))

    if n_neg > 0:
        neg_vecs = features[neg_coords[:, 0], neg_coords[:, 1]]
        pair_list.append(pair_features(neg_vecs, q_desc))
        label_list.append(np.zeros(n_neg, dtype=np.int32))
        neg_weight = 0.5 / n_neg if n_pos > 0 else 1.0 / n_neg
        weight_list.append(np.full(n_neg, neg_weight, dtype=np.float32))

    if not pair_list:
        return np.empty((0, 4 * len(q_desc)), dtype=np.float32), np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)

    return np.concatenate(pair_list, axis=0), np.concatenate(label_list, axis=0), np.concatenate(weight_list, axis=0)


def collect_training_dataset(
    rows: list[dict[str, Any]],
    data_root: Path,
    stride: int,
    samples_per_class: int,
    base_seed: int,
    query_source: str = "drawing",
    feature_set: str = "base",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute features once per drawing and collect balanced pairs across all training queries."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["image"]].append(row)

    all_pairs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_weights: list[np.ndarray] = []

    query_counter = 0
    for image_rel, query_group in grouped.items():
        image_start = time.monotonic()
        image_path = data_path(data_root, image_rel)
        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise EvaluationError(f"Cannot read training drawing: {image_rel}")

        if feature_set == "base":
            features = hatch_texture.texture_features(gray, stride=stride)
        else:
            features = hatch_texture.texture_features(gray, stride=stride, feature_set=feature_set)

        for row in query_group:
            if gray.shape != (row["height"], row["width"]):
                raise EvaluationError(f"Image dimension mismatch for row {row['id']}")
            pairs, labels, weights = extract_query_samples(
                row,
                features,
                gray.shape,
                data_root,
                stride=stride,
                samples_per_class=samples_per_class,
                query_seed=base_seed + query_counter,
                gray=gray,
                query_source=query_source,
                feature_set=feature_set,
            )
            query_counter += 1
            if len(pairs) > 0:
                all_pairs.append(pairs)
                all_labels.append(labels)
                all_weights.append(weights)

        del gray, features
        print(json.dumps({'drawing':image_rel, 'queries':len(query_group),
                          'processed_queries':query_counter,
                          'seconds':time.monotonic()-image_start}), flush=True)

    if not all_pairs:
        raise ValueError("No training samples extracted across all manifest queries")

    return np.concatenate(all_pairs, axis=0), np.concatenate(all_labels, axis=0), np.concatenate(all_weights, axis=0)


def train_classifier(
    x: np.ndarray, y: np.ndarray, weights: np.ndarray, max_iter: int, seed: int
) -> HistGradientBoostingClassifier:
    """Train HistGradientBoostingClassifier with fixed regularized parameters."""
    clf = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        max_iter=max_iter,
        early_stopping=False,
        random_state=seed,
    )
    clf.fit(x, y, sample_weight=weights)
    return clf


def build_metadata(
    args: argparse.Namespace,
    manifest_sha256: str,
    base_feature_names: list[str],
    rows: list[dict[str, Any]],
    y: np.ndarray,
    wall_time: float,
) -> dict[str, Any]:
    """Assemble exact model metadata and training coverage records."""
    training_ids = sorted(row["id"] for row in rows)
    training_docs = sorted({row["document_id"] for row in rows if "document_id" in row})
    return {
        "stride": DEFAULT_STRIDE,
        "query_descriptor_source": _recorded_query_source(args),
        "texture_feature_set": _recorded_feature_set(args),
        "base_feature_names": base_feature_names,
        "pair_feature_names": pair_feature_names(base_feature_names),
        "source_manifest_sha256": manifest_sha256,
        "training_ids": training_ids,
        "training_documents": training_docs,
        "training_coverage": {
            "queries": len(training_ids),
            "documents": len(training_docs),
            "total_samples": len(y),
            "positive_samples": int(np.sum(y == 1)),
            "negative_samples": int(np.sum(y == 0)),
        },
        "params": {
            "learning_rate": 0.08,
            "max_leaf_nodes": 31,
            "l2_regularization": 1.0,
            "max_iter": args.max_iter,
            "early_stopping": False,
        },
        "seed": args.seed,
        "samples_per_class": args.samples_per_class,
        "wall_time_seconds": wall_time,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
    }


def _recorded_query_source(args: argparse.Namespace) -> str:
    """Validate the CLI choice with the same rule inference uses for metadata."""
    return hatch_texture.descriptor_source_from_metadata(
        {"query_descriptor_source": args.query_source}
    )


def _recorded_feature_set(args: argparse.Namespace) -> str:
    """Validate the CLI choice with the same rule inference uses for metadata."""
    feature_set = getattr(args, "features", "base")
    return hatch_texture.feature_set_from_metadata(
        {"texture_feature_set": feature_set}
    )


def _query_box(row: dict[str, Any]) -> tuple[int, int, int, int]:
    return cast(tuple[int, int, int, int], tuple(row["query_box"]))


def main() -> None:
    args = parse_args()
    start_time = time.monotonic()

    manifest_bytes = args.manifest.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()

    rows = load_manifest(args.manifest, require_labels=True)
    validate_split_safety(args.manifest, args.data_root, rows)

    feature_set = _recorded_feature_set(args)
    base_feature_names = hatch_texture.texture_feature_names(feature_set)
    query_source = _recorded_query_source(args)
    x, y, weights = collect_training_dataset(
        rows,
        args.data_root,
        stride=DEFAULT_STRIDE,
        samples_per_class=args.samples_per_class,
        base_seed=args.seed,
        query_source=query_source,
        feature_set=feature_set,
    )

    clf = train_classifier(x, y, weights, max_iter=args.max_iter, seed=args.seed)
    wall_time = time.monotonic() - start_time

    metadata = build_metadata(args, manifest_sha256, base_feature_names, rows, y, wall_time)
    save_texture_model(clf, metadata, args.output)

    print(
        f"Fitted model on {len(rows)} queries ({metadata['training_coverage']['documents']} docs, "
        f"{len(y)} samples) in {wall_time:.2f}s -> {args.output}"
    )


if __name__ == "__main__":
    main()
