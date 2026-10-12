"""Label-free native-resolution inference with trained texture classifier."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, cast

import cv2
import numpy as np

from evaluate import EvaluationError
import hatch_texture
from inference import read_drawing, read_requests, save_prediction
from texture_learning import grid_centers, load_texture_model, pair_features

BATCH_SIZE: int = 65536
STRIPE_HEIGHT: int = 256
QUERY_SELF_QUANTILES: tuple[str, ...] = ("median", "q10", "q25", "q75")
CALIBRATION_CLIP_LOW: float = 0.01
CALIBRATION_CLIP_HIGH: float = 0.99


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
    try:
        feature_set = hatch_texture.feature_set_from_metadata(metadata)
    except (TypeError, ValueError) as exc:
        raise EvaluationError(f"Invalid texture feature set metadata: {exc}") from exc
    if feature_set == "base":
        current_features = hatch_texture.texture_feature_names()
    else:
        current_features = hatch_texture.texture_feature_names(feature_set)
    expected_features = metadata.get("base_feature_names")
    if expected_features != current_features:
        raise EvaluationError(
            f"Model feature mismatch for {feature_set}: expected {expected_features}, "
            f"current runtime has {current_features}"
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
    flat_probs = _predict_probabilities(model, flat_features, query_desc, batch_size)
    return flat_probs.reshape(grid_h, grid_w)


def query_self_quantiles(
    model: Any,
    features: np.ndarray,
    query_desc: np.ndarray,
    query_box: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    stride: int,
) -> dict[str, float]:
    """Predict self-match probability quantiles at feature cells in the query."""
    centers_y, centers_x = grid_centers(image_shape[0], image_shape[1], stride)
    y0, x0, y1, x1 = query_box[1], query_box[0], query_box[3], query_box[2]
    y_indices = np.flatnonzero((centers_y >= y0) & (centers_y < y1))
    x_indices = np.flatnonzero((centers_x >= x0) & (centers_x < x1))
    if len(y_indices) == 0 or len(x_indices) == 0:
        y_indices, x_indices = _nearest_query_cell(centers_y, centers_x, query_box)
    vectors = features[np.ix_(y_indices, x_indices)].reshape(-1, features.shape[-1])
    probabilities = _predict_probabilities(model, vectors, query_desc)
    return {
        "median": float(np.quantile(probabilities, 0.50)),
        "q10": float(np.quantile(probabilities, 0.10)),
        "q25": float(np.quantile(probabilities, 0.25)),
        "q75": float(np.quantile(probabilities, 0.75)),
        "count": float(len(probabilities)),
    }


def threshold_from_query_self(
    query_self: Mapping[str, Any], quantile: str, scale: float, offset: float
) -> float:
    """Convert a query-self statistic into a clipped probability threshold."""
    if not isinstance(query_self, Mapping):
        raise ValueError("query_self must be a mapping")
    if quantile not in QUERY_SELF_QUANTILES:
        raise ValueError(f"Unsupported query-self quantile: {quantile}")
    value = query_self.get(quantile)
    _validate_finite_number(value, f"query_self[{quantile}]")
    _validate_finite_number(scale, "scale")
    _validate_finite_number(offset, "offset")
    threshold = float(np.clip(
        float(scale) * float(cast(float, value)) + float(offset),
        CALIBRATION_CLIP_LOW,
        CALIBRATION_CLIP_HIGH,
    ))
    if not math.isfinite(threshold):
        raise ValueError("calibrated threshold must be finite")
    return threshold


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

    gy_native, gx = native_grid_coordinates(height, width, stride)

    for y0 in range(0, height, stripe_height):
        y1 = min(y0 + stripe_height, height)
        gy = gy_native[y0:y1]

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


def native_grid_coordinates(height: int, width: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    centers_y, centers_x = grid_centers(height, width, stride)
    gy = np.interp(np.arange(height), centers_y, np.arange(len(centers_y)))
    gx = np.interp(np.arange(width), centers_x, np.arange(len(centers_x)))
    return gy.astype(np.float32), gx.astype(np.float32)


def _predict_probabilities(
    model: Any,
    vectors: np.ndarray,
    query_desc: np.ndarray,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    result = np.empty(len(vectors), dtype=np.float32)
    for start in range(0, len(vectors), batch_size):
        end = min(start + batch_size, len(vectors))
        raw = model.predict_proba(pair_features(vectors[start:end], query_desc))
        probs = _validated_model_probabilities(raw, end - start)
        result[start:end] = probs
    if not bool(np.all(np.isfinite(result))) or bool(np.any((result < 0) | (result > 1))):
        raise EvaluationError("Query-self model probabilities are invalid")
    return result


def _validated_model_probabilities(raw: Any, expected_count: int) -> np.ndarray:
    try:
        values = np.asarray(raw)
    except (TypeError, ValueError) as exc:
        raise EvaluationError("Model probabilities have an invalid output shape") from exc
    if values.ndim == 1 and values.shape == (expected_count,):
        probability_column = values
    elif values.ndim == 2 and values.shape[1] in (1, 2):
        if values.shape[0] != expected_count:
            raise EvaluationError("Model probabilities have an invalid output length")
        probability_column = values[:, -1]
    else:
        raise EvaluationError("Model probabilities have an invalid output shape")
    try:
        values = np.asarray(values, dtype=np.float32)
        probability_column = np.asarray(probability_column, dtype=np.float32)
        valid = np.isfinite(values).all() and np.all((values >= 0) & (values <= 1))
    except (TypeError, ValueError, OverflowError) as exc:
        raise EvaluationError("Model probabilities are invalid") from exc
    if not bool(valid):
        raise EvaluationError("Model probabilities are invalid")
    return probability_column


def _nearest_query_cell(
    centers_y: np.ndarray, centers_x: np.ndarray, query_box: tuple[int, int, int, int]
) -> tuple[np.ndarray, np.ndarray]:
    cx = (query_box[0] + query_box[2] - 1) / 2.0
    cy = (query_box[1] + query_box[3] - 1) / 2.0
    return np.array([int(np.argmin(np.abs(centers_y - cy)))]), np.array(
        [int(np.argmin(np.abs(centers_x - cx)))]
    )


def _validate_finite_number(value: Any, name: str) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite number")


def process_query_group(
    model: Any,
    gray: np.ndarray,
    query_group: list[dict[str, Any]],
    output_dir: Path,
    stride: int,
    threshold: float,
    source: str = "drawing",
    feature_set: str = "base",
) -> list[dict[str, Any]]:
    """Compute texture features once and run prediction for all queries sharing an image."""
    if feature_set == "base":
        features = hatch_texture.texture_features(gray, stride=stride)
    else:
        features = hatch_texture.texture_features(gray, stride=stride, feature_set=feature_set)
    records: list[dict[str, Any]] = []

    for row in query_group:
        if gray.shape != (row["height"], row["width"]):
            raise EvaluationError(f"Inconsistent image dimensions for query {row['id']}")

        query_start = time.monotonic()
        if feature_set == "base":
            q_desc = hatch_texture.descriptor_for_query(
                gray, features, _query_box(row), stride, source
            )
        else:
            q_desc = hatch_texture.descriptor_for_query(
                gray, features, _query_box(row), stride, source, feature_set=feature_set
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
    source = hatch_texture.descriptor_source_from_metadata(metadata)
    feature_set = hatch_texture.feature_set_from_metadata(metadata)
    all_records: list[dict[str, Any]] = []

    for query_group in grouped.values():
        gray = read_drawing(data_root, query_group[0])
        group_records = process_query_group(
            model, gray, query_group, output_dir, stride, threshold, source, feature_set
        )
        all_records.extend(group_records)
        del gray

    return all_records


def _query_box(row: dict[str, Any]) -> tuple[int, int, int, int]:
    return cast(tuple[int, int, int, int], tuple(row["query_box"]))


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
