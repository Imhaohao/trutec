"""Label-free native-resolution fusion of texture and CNN hatch predictions."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import time
from typing import Any, Sequence, cast

import numpy as np

import hatch_texture
from evaluate import EvaluationError
from inference import read_drawing, read_requests, save_prediction
from inference_hatchfinder import (
    load_hatchfinder,
    read_rgb_drawing,
    resolve_device,
    sha256_file,
    tiled_probability,
)
from inference_texture import (
    QUERY_SELF_QUANTILES,
    interpolate_and_threshold_mask,
    predict_probability_grid,
    query_self_quantiles,
    threshold_from_query_self,
    verify_model_metadata,
)
from texture_learning import load_texture_model

DEFAULT_QUANTILE = "q10"
DEFAULT_SCALE = 0.7
DEFAULT_OFFSET = -0.1
DEFAULT_CNN_THRESHOLD = 0.4
DEFAULT_TILE_SIZE = 896
DEFAULT_OVERLAP = 128


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--texture-model", required=True, type=Path)
    parser.add_argument("--cnn-model", required=True, type=Path)
    parser.add_argument("--quantile", choices=QUERY_SELF_QUANTILES, default=DEFAULT_QUANTILE)
    parser.add_argument("--scale", type=float, default=DEFAULT_SCALE)
    parser.add_argument("--offset", type=float, default=DEFAULT_OFFSET)
    parser.add_argument("--cnn-threshold", type=float, default=DEFAULT_CNN_THRESHOLD)
    parser.add_argument("--device", choices=("mps", "cpu"), default=None)
    parser.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(None if argv is None else list(argv))
    try:
        validate_fusion_settings(
            args.quantile,
            args.scale,
            args.offset,
            args.cnn_threshold,
            args.tile_size,
            args.overlap,
        )
    except ValueError as exc:
        parser.error(str(exc))
    return args


def validate_fusion_settings(
    quantile: str,
    scale: float,
    offset: float,
    cnn_threshold: float,
    tile_size: int,
    overlap: int,
) -> None:
    if quantile not in QUERY_SELF_QUANTILES:
        raise ValueError(f"unsupported query-self quantile: {quantile}")
    threshold_from_query_self({quantile: 0.5}, quantile, scale, offset)
    _validate_probability_threshold(cnn_threshold, "cnn threshold")
    if tile_size <= 0 or tile_size % 64 != 0:
        raise ValueError("tile size must be a positive multiple of 64")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("overlap must be >= 0 and smaller than tile size")


def run_fusion_inference(
    rows: list[dict[str, Any]],
    data_root: Path,
    output_dir: Path,
    texture_model: Any,
    texture_metadata: dict[str, Any],
    cnn_model: Any,
    device: str,
    quantile: str = DEFAULT_QUANTILE,
    scale: float = DEFAULT_SCALE,
    offset: float = DEFAULT_OFFSET,
    cnn_threshold: float = DEFAULT_CNN_THRESHOLD,
    tile_size: int = DEFAULT_TILE_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[dict[str, Any]]:
    """Generate native AND masks while retaining only one drawing group."""
    validate_fusion_settings(quantile, scale, offset, cnn_threshold, tile_size, overlap)
    verify_model_metadata(texture_metadata)
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["image"])].append(row)

    stride = int(texture_metadata.get("stride", 4))
    source = hatch_texture.descriptor_source_from_metadata(texture_metadata)
    records: list[dict[str, Any]] = []
    for query_group in grouped.values():
        gray = read_drawing(data_root, query_group[0])
        rgb = read_rgb_drawing(data_root, query_group[0])
        _check_shared_shape(gray, rgb, query_group[0])
        features = hatch_texture.texture_features(gray, stride=stride)
        for row in query_group:
            records.append(
                _process_query(
                    row,
                    gray,
                    rgb,
                    features,
                    texture_model,
                    cnn_model,
                    output_dir,
                    device,
                    quantile,
                    scale,
                    offset,
                    cnn_threshold,
                    tile_size,
                    overlap,
                    stride,
                    source,
                )
            )
        del features, rgb, gray
    return records


def _process_query(
    row: dict[str, Any],
    gray: np.ndarray,
    rgb: np.ndarray,
    features: np.ndarray,
    texture_model: Any,
    cnn_model: Any,
    output_dir: Path,
    device: str,
    quantile: str,
    scale: float,
    offset: float,
    cnn_threshold: float,
    tile_size: int,
    overlap: int,
    stride: int,
    source: str,
) -> dict[str, Any]:
    _check_shared_shape(gray, rgb, row)
    started = time.monotonic()
    query_box = _query_box(row)
    query_desc = hatch_texture.descriptor_for_query(gray, features, query_box, stride, source)
    query_self = query_self_quantiles(
        texture_model, features, query_desc, query_box, gray.shape, stride
    )
    texture_threshold = threshold_from_query_self(query_self, quantile, scale, offset)
    texture_grid = predict_probability_grid(texture_model, features, query_desc)
    texture_mask = interpolate_and_threshold_mask(
        texture_grid, gray.shape[0], gray.shape[1], stride, texture_threshold
    )
    del texture_grid

    cnn_probability = tiled_probability(
        cnn_model, rgb, query_box, tile_size, overlap, device
    )
    _validate_native_probability(cnn_probability, rgb.shape[:2], "CNN")
    cnn_mask = cnn_probability >= cnn_threshold
    selected = np.logical_and(texture_mask, cnn_mask)
    save_prediction(output_dir / f"{row['id']}.png", selected)
    record = {
        "id": row["id"],
        "seconds": time.monotonic() - started,
        "selected_pixels": int(np.count_nonzero(selected)),
        "texture_threshold": texture_threshold,
        "cnn_threshold": cnn_threshold,
        "quantile": quantile,
        "scale": scale,
        "offset": offset,
        "query_self": {key: query_self[key] for key in QUERY_SELF_QUANTILES},
        "query_self_cell_count": int(query_self["count"]),
    }
    print(json.dumps(record), flush=True)
    del cnn_probability, cnn_mask, texture_mask, selected
    return record


def build_runtime_report(
    elapsed_seconds: float,
    records: list[dict[str, Any]],
    args: argparse.Namespace,
    device: str,
) -> dict[str, Any]:
    texture_hash = sha256_file(args.texture_model)
    cnn_hash = sha256_file(args.cnn_model)
    return {
        "method": "learned-texture-cnn-fusion-v1",
        "operator": "AND",
        "elapsed_seconds": elapsed_seconds,
        "timing_coverage": "model loading, image loading, feature computation, inference, and PNG writing; model hashing and report writing excluded",
        "device": device,
        "inputs": str(args.inputs),
        "data_root": str(args.data_root),
        "texture_model": str(args.texture_model),
        "texture_model_sha256": texture_hash,
        "cnn_model": str(args.cnn_model),
        "cnn_model_sha256": cnn_hash,
        "selection": {
            "quantile": args.quantile,
            "scale": args.scale,
            "offset": args.offset,
            "cnn_threshold": args.cnn_threshold,
        },
        "tile": {"tile_size": args.tile_size, "overlap": args.overlap},
        "processed_query_count": len(records),
        "notes": "Defaults are diagnostic current-best settings and are not a frozen final pipeline.",
        "examples": records,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    rows = read_requests(args.inputs)
    texture_model, texture_metadata = load_texture_model(args.texture_model)
    verify_model_metadata(texture_metadata)
    device = resolve_device(args.device)
    cnn_model = load_hatchfinder(args.cnn_model, device)
    records = run_fusion_inference(
        rows,
        args.data_root,
        args.output_dir,
        texture_model,
        texture_metadata,
        cnn_model,
        device,
        args.quantile,
        args.scale,
        args.offset,
        args.cnn_threshold,
        args.tile_size,
        args.overlap,
    )
    report = build_runtime_report(time.monotonic() - started, records, args, device)
    (args.output_dir / "runtime.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


def _check_shared_shape(gray: np.ndarray, rgb: np.ndarray, row: dict[str, Any]) -> None:
    if gray.shape != (int(row["height"]), int(row["width"])):
        raise EvaluationError(f"inconsistent grayscale dimensions for {row['id']}")
    if rgb.shape != (int(row["height"]), int(row["width"]), 3):
        raise EvaluationError(f"inconsistent RGB dimensions for {row['id']}")


def _query_box(row: dict[str, Any]) -> tuple[int, int, int, int]:
    return cast(tuple[int, int, int, int], tuple(row["query_box"]))


def _validate_native_probability(probability: np.ndarray, shape: tuple[int, int], name: str) -> None:
    if probability.shape != shape or probability.dtype != np.float32:
        raise EvaluationError(f"{name} probability map does not match native dimensions")
    if not bool(np.all(np.isfinite(probability))) or bool(np.any((probability < 0) | (probability > 1))):
        raise EvaluationError(f"{name} probability map is invalid")


def _validate_probability_threshold(value: Any, name: str) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be a finite number in [0, 1]")


if __name__ == "__main__":
    main()
