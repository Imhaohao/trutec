"""Label-free native-resolution inference with trained texture classifier."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np

from evaluate import EvaluationError
import hatch_texture
from inference import read_drawing, read_requests, save_prediction
from texture_learning import load_texture_model, pair_features

BATCH_SIZE: int = 65536
STRIPE_HEIGHT: int = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path, help="Label-free input requests JSON")
    parser.add_argument("--data-root", required=True, type=Path, help="Dataset root directory")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory to save predicted masks")
    parser.add_argument("--model", required=True, type=Path, help="Trained .joblib model artifact")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold in [0, 1]")
    args = parser.parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("threshold must lie in [0, 1]")
    return args


def verify_model_metadata(metadata: dict[str, Any]) -> None:
    """Verify that model feature signatures match the current runtime texture descriptor."""
    current_features = hatch_texture.texture_feature_names()
    expected_features = metadata.get("base_feature_names")
    if expected_features != current_features:
        raise EvaluationError(
            f"Model feature mismatch: expected {expected_features}, current runtime has {current_features}"
        )


def predict_probability_grid(
    model: Any,
    features: np.ndarray,
    query_desc: np.ndarray,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """Predict match probabilities across feature grid in memory-bounded batches."""
    grid_h, grid_w, channels = features.shape
    flat_features = features.reshape(-1, channels)
    total_vectors = flat_features.shape[0]
    flat_probs = np.empty(total_vectors, dtype=np.float32)

    for start_idx in range(0, total_vectors, batch_size):
        end_idx = min(start_idx + batch_size, total_vectors)
        pairs = pair_features(flat_features[start_idx:end_idx], query_desc)
        probs = model.predict_proba(pairs)[:, 1]
        flat_probs[start_idx:end_idx] = probs

    return flat_probs.reshape(grid_h, grid_w)


def interpolate_and_threshold_mask(
    prob_grid: np.ndarray,
    height: int,
    width: int,
    stride: int,
    threshold: float,
    stripe_height: int = STRIPE_HEIGHT,
) -> np.ndarray:
    """Interpolate grid probabilities to native resolution via row stripes and threshold."""
    grid_h, grid_w = prob_grid.shape
    mask = np.zeros((height, width), dtype=bool)

    xs = np.arange(width, dtype=np.float32)
    gx = np.clip((xs - stride // 2) / float(stride), 0.0, float(grid_w - 1))

    for y0 in range(0, height, stripe_height):
        y1 = min(y0 + stripe_height, height)
        ys = np.arange(y0, y1, dtype=np.float32)
        gy = np.clip((ys - stride // 2) / float(stride), 0.0, float(grid_h - 1))

        map_x, map_y = np.meshgrid(gx, gy)
        stripe_probs = cv2.remap(
            prob_grid,
            map_x.astype(np.float32),
            map_y.astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        mask[y0:y1, :] = stripe_probs >= threshold

    return mask


def process_query_group(
    model: Any,
    gray: np.ndarray,
    query_group: list[dict[str, Any]],
    output_dir: Path,
    stride: int,
    threshold: float,
) -> list[dict[str, Any]]:
    """Compute texture features once and run prediction for all queries sharing an image."""
    features = hatch_texture.texture_features(gray, stride=stride)
    records: list[dict[str, Any]] = []

    for row in query_group:
        if gray.shape != (row["height"], row["width"]):
            raise EvaluationError(f"Inconsistent image dimensions for query {row['id']}")

        query_start = time.monotonic()
        q_desc = hatch_texture.query_descriptor(
            features, tuple(row["query_box"]), gray.shape, stride=stride
        )
        prob_grid = predict_probability_grid(model, features, q_desc)
        mask = interpolate_and_threshold_mask(
            prob_grid, row["height"], row["width"], stride, threshold
        )
        save_prediction(output_dir / f"{row['id']}.png", mask)

        record = {
            "id": row["id"],
            "seconds": time.monotonic() - query_start,
            "selected_pixels": int(mask.sum()),
        }
        records.append(record)
        print(json.dumps(record), flush=True)

    del features
    return records


def run_texture_inference(
    rows: list[dict[str, Any]],
    data_root: Path,
    output_dir: Path,
    model: Any,
    metadata: dict[str, Any],
    threshold: float,
) -> list[dict[str, Any]]:
    """Group requests by image and generate native-resolution predictions."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["image"]].append(row)

    stride = int(metadata.get("stride", 4))
    all_records: list[dict[str, Any]] = []

    for query_group in grouped.values():
        gray = read_drawing(data_root, query_group[0])
        group_records = process_query_group(
            model, gray, query_group, output_dir, stride, threshold
        )
        all_records.extend(group_records)
        del gray

    return all_records


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model, metadata = load_texture_model(args.model)
    verify_model_metadata(metadata)

    rows = read_requests(args.inputs)
    total_start = time.monotonic()

    records = run_texture_inference(
        rows, args.data_root, args.output_dir, model, metadata, args.threshold
    )

    elapsed_total = time.monotonic() - total_start
    report = {
        "method": "learned-texture-v1",
        "model": str(args.model),
        "threshold": args.threshold,
        "processed_query_count": len(records),
        "elapsed_seconds": elapsed_total,
        "timing_coverage": "image loading, feature computation, inference, and writing; imports excluded",
        "examples": records,
    }
    (args.output_dir / "runtime.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
