"""Fast and exact training-development threshold sweep for learned hatch texture matcher.

Development Scoring Optimization vs Production Full Inference:
Dataset known domains are sparse (~100k reviewed pixels in 34M-pixel drawings).
Because the classifier is pointwise on local texture descriptors and bilinear
remapping depends strictly on the four neighboring grid predictions surrounding
each native pixel coordinate, predicting the entire grid of billions of pixels
during threshold exploration is unnecessary.

This module scores every known native pixel EXACTLY as full inference would by
predicting only the required neighboring grid cells for reviewed pixels and
interpolating via cv2.remap. Unreviewed grid cells are initialized to zero, but
because every evaluated pixel is completely enveloped by its 4 computed grid
neighbors, the zeros never affect the score.

Production inference (inference_texture.py) predicts the entire grid and writes
full native binary PNGs. This sweep is strictly for development threshold
selection on training/development partitions and must never evaluate on public val.
No annotation values enter the query descriptor or classifier, only spatial
coordinates of known domains for evaluation efficiency.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Sequence

import cv2
import numpy as np

from evaluate import (
    EvaluationError,
    aggregate,
    load_manifest,
    metrics_from_counts,
    reviewed_domains,
)
import hatch_texture
from inference import read_drawing
from inference_texture import native_grid_coordinates, verify_model_metadata
from texture_learning import load_texture_model, pair_features

DEFAULT_THRESHOLDS: tuple[float, ...] = (
    0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95
)
BATCH_SIZE: int = 65536
REMAP_CHUNK_SIZE: int = 16384


def parse_args() -> argparse.Namespace:
    """Parse CLI options for threshold sweep."""
    parser = argparse.ArgumentParser(
        description="Fast exact threshold sweep on reviewed training/development domains."
    )
    parser.add_argument(
        "--manifest", required=True, type=Path, help="Annotated training-development JSON manifest"
    )
    parser.add_argument(
        "--data-root", required=True, type=Path, help="Dataset root directory"
    )
    parser.add_argument(
        "--model", required=True, type=Path, help="Trained .joblib model artifact"
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="Directory to save sweep results"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Optional maximum number of queries to evaluate"
    )
    return parser.parse_args()


def get_validation_document_ids(data_root: Path) -> set[str]:
    """Extract document IDs from public val.json strictly to prevent data leakage."""
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
    """Ensure that the sweep manifest does not touch public validation data."""
    resolved = manifest_path.resolve()
    val_candidates = [(data_root / "val.json").resolve(), Path("dataset/val.json").resolve()]
    if resolved.name == "val.json" or any(resolved == c for c in val_candidates if c.exists()):
        raise EvaluationError(
            f"Development threshold sweep must not evaluate on validation manifest: {manifest_path}"
        )

    val_doc_ids = get_validation_document_ids(data_root)
    if not val_doc_ids:
        return

    for row in rows:
        doc_id = row.get("document_id")
        if doc_id in val_doc_ids:
            raise EvaluationError(
                f"Development sweep example {row.get('id')} belongs to validation document: {doc_id}"
            )


def find_required_grid_indices(
    sample_gy: np.ndarray,
    sample_gx: np.ndarray,
    grid_h: int,
    grid_w: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect unique grid indices for the four bounding neighbor cells of every coordinate."""
    yf = np.clip(np.floor(sample_gy).astype(np.int32), 0, grid_h - 1)
    yc = np.clip(np.ceil(sample_gy).astype(np.int32), 0, grid_h - 1)
    xf = np.clip(np.floor(sample_gx).astype(np.int32), 0, grid_w - 1)
    xc = np.clip(np.ceil(sample_gx).astype(np.int32), 0, grid_w - 1)

    corners_y = np.concatenate([yf, yf, yc, yc])
    corners_x = np.concatenate([xf, xc, xf, xc])
    linear_indices = corners_y.astype(np.int64) * grid_w + corners_x.astype(np.int64)
    unique_linear = np.unique(linear_indices)

    uy = (unique_linear // grid_w).astype(np.int32)
    ux = (unique_linear % grid_w).astype(np.int32)
    return uy, ux


def predict_unique_grid_probabilities(
    model: Any,
    features: np.ndarray,
    query_desc: np.ndarray,
    uy: np.ndarray,
    ux: np.ndarray,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """Predict match probabilities for required grid cells in memory-bounded batches."""
    num_unique = len(uy)
    unique_probs = np.empty(num_unique, dtype=np.float32)

    for start_idx in range(0, num_unique, batch_size):
        end_idx = min(start_idx + batch_size, num_unique)
        batch_vecs = features[uy[start_idx:end_idx], ux[start_idx:end_idx]]
        batch_pairs = pair_features(batch_vecs, query_desc)
        raw_probs = model.predict_proba(batch_pairs)
        if raw_probs.ndim == 2 and raw_probs.shape[1] > 1:
            probs = raw_probs[:, 1]
        else:
            probs = raw_probs.ravel()
        unique_probs[start_idx:end_idx] = probs.astype(np.float32)

    if not bool(np.all(np.isfinite(unique_probs))):
        raise EvaluationError("Model returned non-finite probabilities")
    if bool(np.any((unique_probs < 0.0) | (unique_probs > 1.0))):
        raise EvaluationError("Model returned probabilities outside [0, 1]")

    return unique_probs


def remap_known_coordinates(
    prob_grid: np.ndarray,
    sample_gx: np.ndarray,
    sample_gy: np.ndarray,
    chunk_size: int = REMAP_CHUNK_SIZE,
) -> np.ndarray:
    """Interpolate grid probabilities at native coordinates using OpenCV remap chunks."""
    total_known = len(sample_gx)
    known_probs = np.empty(total_known, dtype=np.float32)

    for c_start in range(0, total_known, chunk_size):
        c_end = min(c_start + chunk_size, total_known)
        chunk_x = sample_gx[c_start:c_end].reshape(-1, 1).astype(np.float32)
        chunk_y = sample_gy[c_start:c_end].reshape(-1, 1).astype(np.float32)
        chunk_res = cv2.remap(
            prob_grid,
            chunk_x,
            chunk_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        known_probs[c_start:c_end] = chunk_res.ravel()

    if not bool(np.all(np.isfinite(known_probs))):
        raise EvaluationError("Remapped probabilities contain non-finite values")

    np.clip(known_probs, 0.0, 1.0, out=known_probs)
    return known_probs


def score_known_probabilities(
    model: Any,
    features: np.ndarray,
    query_desc: np.ndarray,
    known: np.ndarray,
    height: int,
    width: int,
    stride: int,
) -> np.ndarray:
    """Score every known native pixel matching np.nonzero(known) row-major order."""
    if known.shape != (height, width):
        raise ValueError(f"known domain shape {known.shape} does not match {(height, width)}")

    known_y, known_x = np.nonzero(known)
    if len(known_y) == 0:
        return np.empty(0, dtype=np.float32)

    gy_native, gx_native = native_grid_coordinates(height, width, stride)
    sample_gy = gy_native[known_y]
    sample_gx = gx_native[known_x]

    grid_h, grid_w, _ = features.shape
    uy, ux = find_required_grid_indices(sample_gy, sample_gx, grid_h, grid_w)
    unique_probs = predict_unique_grid_probabilities(model, features, query_desc, uy, ux)

    prob_grid = np.zeros((grid_h, grid_w), dtype=np.float32)
    prob_grid[uy, ux] = unique_probs

    return remap_known_coordinates(prob_grid, sample_gx, sample_gy)


def score_query_thresholds(
    known_probs: np.ndarray,
    positive: np.ndarray,
    known: np.ndarray,
    blank: np.ndarray,
    row: dict[str, Any],
    metadata: dict[str, Any],
    thresholds: Sequence[float],
) -> dict[str, dict[str, Any]]:
    """Compute exact confusion counts and metrics across candidate thresholds."""
    known_y, known_x = np.nonzero(known)
    pos_vector = positive[known_y, known_x]
    blank_vector = blank[known_y, known_x]

    positive_count = int(pos_vector.sum())
    negative_count = len(known_y) - positive_count
    blank_count = int(blank_vector.sum())

    scores: dict[str, dict[str, Any]] = {}
    for threshold in thresholds:
        selected = known_probs >= threshold
        tp = int(np.count_nonzero(selected & pos_vector))
        fp = int(np.count_nonzero(selected & ~pos_vector))
        fn = positive_count - tp
        tn = negative_count - fp
        blank_fp = int(np.count_nonzero(selected & blank_vector))

        metrics = metrics_from_counts(tp, fp, fn, tn, blank_fp, blank_count)
        scores[str(threshold)] = {
            "id": row["id"],
            "document_id": row["document_id"],
            "kind": row["kind"],
            **metadata,
            "metrics": metrics,
        }

    return scores


def process_sweep_query(
    row: dict[str, Any],
    features: np.ndarray,
    gray_shape: tuple[int, int],
    model: Any,
    data_root: Path,
    stride: int,
    thresholds: Sequence[float],
    probability_dir: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Compute query descriptor, score known probabilities, and compute threshold metrics."""
    if gray_shape != (row["height"], row["width"]):
        raise EvaluationError(f"Inconsistent dimensions for query {row['id']}")

    positive, known, blank, metadata = reviewed_domains(row, data_root)
    query_box = tuple(row["query_box"])
    query_desc = hatch_texture.query_descriptor(features, query_box, gray_shape, stride=stride)

    known_probs = score_known_probabilities(
        model, features, query_desc, known, gray_shape[0], gray_shape[1], stride
    )
    if probability_dir is not None:
        probability_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            probability_dir / f"{row['id']}.npz", probability=known_probs,
            positive=positive[known], blank=blank[known],
        )
    return score_query_thresholds(known_probs, positive, known, blank, row, metadata, thresholds)


def run_sweep(
    rows: list[dict[str, Any]],
    data_root: Path,
    model: Any,
    metadata: dict[str, Any],
    output_dir: Path,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> list[dict[str, Any]]:
    """Execute threshold sweep across queries grouped by image, logging incrementally."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["image"]].append(row)

    stride = int(metadata.get("stride", 4))
    records: list[dict[str, Any]] = []
    jsonl_path = output_dir / "query-records.jsonl"

    with jsonl_path.open("w", encoding="utf-8") as log_file:
        for image_path, query_group in grouped.items():
            gray = read_drawing(data_root, query_group[0])
            features = hatch_texture.texture_features(gray, stride=stride)

            for row in query_group:
                query_start = time.monotonic()
                scores = process_sweep_query(
                    row, features, gray.shape, model, data_root, stride, thresholds,
                    output_dir / "probabilities",
                )
                query_seconds = time.monotonic() - query_start

                record = {
                    "id": row["id"],
                    "document_id": row["document_id"],
                    "kind": row["kind"],
                    "seconds": query_seconds,
                    "scores": scores,
                }
                records.append(record)
                log_file.write(json.dumps(record) + "\n")
                log_file.flush()

                sample_key = str(0.5) if str(0.5) in scores else str(thresholds[0])
                progress_info = {
                    "id": row["id"],
                    "seconds": round(query_seconds, 4),
                    f"iou_at_{sample_key}": scores[sample_key]["metrics"]["iou"],
                }
                print(json.dumps(progress_info), flush=True)

            del features
            del gray

    return records


def build_sweep_report(
    rows: list[dict[str, Any]],
    manifest_path: Path,
    model_path: Path,
    summaries: dict[str, dict[str, Any]],
    elapsed_seconds: float,
    thresholds: Sequence[float],
) -> dict[str, Any]:
    """Assemble structured sweep report with integrity hashes, summaries, and disclaimers."""
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
    impl_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    doc_count = len({r["document_id"] for r in rows})

    first_key = str(thresholds[0])
    first_summary = summaries.get(first_key, {})

    return {
        "method": "learned-texture-sweep-v1",
        "partition": "public-training development only",
        "manifest_sha256": manifest_sha256,
        "source_manifest_sha256": manifest_sha256,
        "model_sha256": model_sha256,
        "implementation_sha256": impl_sha256,
        "queries": len(rows),
        "documents": doc_count,
        "real_document_count": doc_count,
        "timing": {
            "elapsed_seconds": elapsed_seconds,
            "timing_coverage": (
                "image loading, feature computation once per image, exact sparse "
                "known scoring, and aggregation; imports excluded"
            ),
        },
        "elapsed_seconds": elapsed_seconds,
        "summaries": summaries,
        "all_summaries": summaries,
        "thresholds": summaries,
        "eligible_counts": {
            "queries": len(rows),
            "documents": doc_count,
            "nonempty_target_examples": first_summary.get("nonempty_target_examples", 0),
            "empty_target_examples": first_summary.get("empty_target_examples", 0),
            "positive_only_examples": first_summary.get("positive_only_examples", 0),
        },
        "exact_known_scoring_only": True,
        "whole_sheet_accuracy_certified": False,
        "target_guarantees": None,
        "production_inference_distinction": (
            "Development sweep only scores known reviewed pixels by evaluating "
            "necessary neighboring grid cells; production inference predicts all grid "
            "cells and remaps the whole native image."
        ),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model, metadata = load_texture_model(args.model)
    verify_model_metadata(metadata)

    rows = load_manifest(args.manifest, require_labels=True)
    validate_split_safety(args.manifest, args.data_root, rows)

    if args.limit is not None:
        rows = rows[: args.limit]

    start_time = time.monotonic()
    records = run_sweep(rows, args.data_root, model, metadata, args.output_dir, DEFAULT_THRESHOLDS)
    elapsed_seconds = time.monotonic() - start_time

    summaries = {
        str(t): aggregate([record["scores"][str(t)] for record in records])
        for t in DEFAULT_THRESHOLDS
    }

    report = build_sweep_report(
        rows, args.manifest, args.model, summaries, elapsed_seconds, DEFAULT_THRESHOLDS
    )
    sweep_path = args.output_dir / "sweep.json"
    sweep_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    primary_metrics = {
        key: {
            name: value[name]
            for name in ("document_macro_iou", "global_precision", "global_recall")
        }
        for key, value in summaries.items()
    }
    print(json.dumps(primary_metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
