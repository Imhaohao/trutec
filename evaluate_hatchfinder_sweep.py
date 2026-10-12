"""Exact development threshold sweep for the pretrained HatchFinder adapter.

The sweep evaluates only reviewed pixels, but it preserves the production
adapter's native tile layout.  A tile is skipped only when it contains no
reviewed pixel; every tile that overlaps a reviewed pixel contributes to that
pixel's numerator and denominator with the same padding and positive Hann
weights used by :func:`inference_hatchfinder.tiled_probability`.

This module is a development scoring tool.  It writes sparse reviewed
probabilities for later calibration and never presents them as a full-sheet
prediction.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np

from evaluate import EvaluationError, aggregate, load_manifest, reviewed_domains
import inference_hatchfinder as adapter


DEFAULT_THRESHOLDS: tuple[float, ...] = (
    0.01,
    0.025,
    0.05,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    0.95,
    0.99,
)
PROBABILITY_DIRNAME = "probabilities"


def validate_split_safety(
    manifest_path: Path, data_root: Path, rows: list[dict[str, Any]]
) -> None:
    """Use the shared development/public-validation partition guard."""
    from evaluate_texture_sweep import validate_split_safety as shared_guard

    shared_guard(manifest_path, data_root, rows)


def score_query_thresholds(
    known_probs: np.ndarray,
    positive: np.ndarray,
    known: np.ndarray,
    blank: np.ndarray,
    row: dict[str, Any],
    metadata: dict[str, Any],
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> dict[str, dict[str, Any]]:
    """Use the shared official count and metric implementation."""
    from evaluate_texture_sweep import score_query_thresholds as shared_scorer

    return shared_scorer(known_probs, positive, known, blank, row, metadata, thresholds)


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for the development sweep."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--tile-size", type=int, default=896)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI options, accepting an argv sequence for tests."""
    return build_parser().parse_args(None if argv is None else list(argv))


def score_known_probabilities(
    model: Any,
    rgb: np.ndarray,
    query_box: tuple[int, int, int, int],
    known: np.ndarray,
    tile_size: int,
    overlap: int,
    device: str,
) -> np.ndarray:
    """Score known pixels exactly as full native tiled inference would.

    The returned vector follows ``np.nonzero(known)`` row-major order.  Only
    tiles containing at least one known pixel are forwarded, while all tiles
    that overlap any known pixel are retained in the stitch accumulators.
    """
    _validate_score_inputs(rgb, known, tile_size, overlap)
    known_y, known_x = np.nonzero(known)
    if len(known_y) == 0:
        return np.empty(0, dtype=np.float32)

    hatch = adapter.rgb_to_tensor(adapter.extract_query(rgb, query_box), device)
    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    known_linear = known_y.astype(np.int64) * width + known_x.astype(np.int64)
    numerator = np.zeros(len(known_y), dtype=np.float32)
    denominator = np.zeros(len(known_y), dtype=np.float32)

    for y0 in adapter.axis_starts(height, tile_size, overlap):
        valid_h = min(tile_size, height - y0)
        for x0 in adapter.axis_starts(width, tile_size, overlap):
            valid_w = min(tile_size, width - x0)
            local_known = known[y0 : y0 + valid_h, x0 : x0 + valid_w]
            if not bool(local_known.any()):
                continue
            _accumulate_known_tile(
                model,
                rgb,
                hatch,
                local_known,
                known_linear,
                numerator,
                denominator,
                y0,
                x0,
                valid_h,
                valid_w,
                width,
                height,
                tile_size,
                overlap,
                device,
            )

    del hatch
    if not bool(np.all(denominator > 0.0)):
        raise EvaluationError("known pixels were left uncovered by sparse stitching")
    probabilities = numerator / denominator
    if not bool(np.all(np.isfinite(probabilities))):
        raise EvaluationError("sparse stitched probabilities are not finite")
    np.clip(probabilities, 0.0, 1.0, out=probabilities)
    return probabilities.astype(np.float32, copy=False)


def process_sweep_query(
    row: dict[str, Any],
    rgb: np.ndarray,
    model: Any,
    data_root: Path,
    tile_size: int,
    overlap: int,
    device: str,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> tuple[dict[str, dict[str, Any]], np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Load reviewed domains, score one query, and return calibration vectors."""
    expected = (int(row["height"]), int(row["width"]), 3)
    if rgb.shape != expected:
        raise EvaluationError(f"Inconsistent dimensions for query {row['id']}")

    positive, known, blank, metadata = reviewed_domains(row, data_root)
    query_box = _query_tuple(row["query_box"])
    known_probs = score_known_probabilities(
        model, rgb, query_box, known, tile_size, overlap, device
    )
    scores = score_query_thresholds(
        known_probs, positive, known, blank, row, metadata, thresholds
    )
    known_y, known_x = np.nonzero(known)
    return scores, known_probs, np.asarray(positive[known], dtype=bool), np.asarray(
        blank[known], dtype=bool
    ), np.stack((known_y, known_x), axis=1).astype(np.int32), metadata


def run_sweep(
    rows: list[dict[str, Any]],
    data_root: Path,
    model: Any,
    output_dir: Path,
    tile_size: int = 896,
    overlap: int = 128,
    device: str = "mps",
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> list[dict[str, Any]]:
    """Run the sparse exact sweep and append each query record incrementally."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["image"])].append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    probability_dir = output_dir / PROBABILITY_DIRNAME
    probability_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    jsonl_path = output_dir / "query-records.jsonl"

    with jsonl_path.open("w", encoding="utf-8") as log_file:
        for query_group in grouped.values():
            rgb = adapter.read_rgb_drawing(data_root, query_group[0])
            for row in query_group:
                query_started = time.monotonic()
                scores, probs, positive, blank, coordinates, metadata = process_sweep_query(
                    row,
                    rgb,
                    model,
                    data_root,
                    tile_size,
                    overlap,
                    device,
                    thresholds,
                )
                adapter._sync(device)
                probability_path = save_known_probabilities(
                    probability_dir, row, probs, positive, blank, coordinates, metadata
                )
                query_seconds = time.monotonic() - query_started
                record = {
                    "id": row["id"],
                    "document_id": row["document_id"],
                    "kind": row["kind"],
                    "seconds": query_seconds,
                    "known_probabilities": str(probability_path.relative_to(output_dir)),
                    "scores": scores,
                }
                records.append(record)
                log_file.write(json.dumps(record) + "\n")
                log_file.flush()
                _print_progress(row, scores, thresholds, query_seconds)
                del probs, positive, blank, coordinates
            del rgb
    adapter._sync(device)
    return records


def save_known_probabilities(
    probability_dir: Path,
    row: dict[str, Any],
    probabilities: np.ndarray,
    positive: np.ndarray,
    blank: np.ndarray,
    coordinates: np.ndarray,
    metadata: dict[str, Any],
) -> Path:
    """Persist one query's reviewed probabilities and label vectors atomically."""
    destination = probability_dir / (str(row["id"]) + ".npz")
    temporary = destination.with_suffix(".partial.npz")
    np.savez_compressed(
        temporary,
        probability=np.asarray(probabilities, dtype=np.float32),
        positive=np.asarray(positive, dtype=bool),
        blank=np.asarray(blank, dtype=bool),
        y=np.asarray(coordinates[:, 0], dtype=np.int32),
        x=np.asarray(coordinates[:, 1], dtype=np.int32),
        id=np.asarray(str(row["id"])),
        document_id=np.asarray(str(row["document_id"])),
        kind=np.asarray(str(row["kind"])),
        height=np.asarray(int(row["height"]), dtype=np.int32),
        width=np.asarray(int(row["width"]), dtype=np.int32),
        original_known_pixels=np.asarray(int(metadata["original_known_pixels"]), dtype=np.int64),
        excluded_known_pixels=np.asarray(int(metadata["excluded_known_pixels"]), dtype=np.int64),
    )
    temporary.replace(destination)
    return destination


def build_sweep_report(
    rows: list[dict[str, Any]],
    manifest_path: Path,
    model_path: Path,
    summaries: dict[str, dict[str, Any]],
    elapsed_seconds: float,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    *,
    device: str = "mps",
    tile_size: int = 896,
    overlap: int = 128,
) -> dict[str, Any]:
    """Build the reproducibility report for a sparse development sweep."""
    manifest_sha256 = _sha256_file(manifest_path)
    model_sha256 = _sha256_file(model_path)
    implementation_sha256 = _sha256_file(Path(__file__))
    config = {
        "device": device,
        "tile_size": int(tile_size),
        "overlap": int(overlap),
        "threshold_values": [float(value) for value in thresholds],
        "thresholds": summaries,
        "probability_dir": PROBABILITY_DIRNAME,
    }
    config_sha256 = _sha256_bytes(json.dumps(config, sort_keys=True).encode("utf-8"))
    first_summary = summaries.get(str(thresholds[0]), {}) if thresholds else {}
    document_count = len({str(row["document_id"]) for row in rows})
    return {
        "schema": "hatch-matching-challenge-hatchfinder-sweep/v1",
        "method": "hatchfinder-native-tiled-development-sweep-v1",
        "partition": "training-development only",
        "manifest_sha256": manifest_sha256,
        "source_manifest_sha256": manifest_sha256,
        "model_sha256": model_sha256,
        "implementation_sha256": implementation_sha256,
        "config_sha256": config_sha256,
        "config": config,
        "queries": len(rows),
        "documents": document_count,
        "real_document_count": document_count,
        "timing": {
            "elapsed_seconds": elapsed_seconds,
            "timing_coverage": (
                "manifest and model loading, image loading, query crop, native tile "
                "forward, sparse stitch, threshold scoring, and NPZ/JSONL writing"
            ),
        },
        "elapsed_seconds": elapsed_seconds,
        "thresholds": [float(value) for value in thresholds],
        "summaries": summaries,
        "all_summaries": summaries,
        "eligible_counts": {
            "queries": len(rows),
            "documents": document_count,
            "nonempty_target_examples": first_summary.get("nonempty_target_examples", 0),
            "empty_target_examples": first_summary.get("empty_target_examples", 0),
            "positive_only_examples": first_summary.get("positive_only_examples", 0),
        },
        "known_probability_dir": PROBABILITY_DIRNAME,
        "exact_known_scoring_only": True,
        "whole_sheet_accuracy_certified": False,
        "target_guarantees": None,
        "production_inference_distinction": (
            "This development artifact stores probabilities only at reviewed pixels. "
            "Production inference must run every native tile and produce a complete "
            "native prediction before any binary mask is submitted."
        ),
    }


def main(argv: Sequence[str] | None = None) -> None:
    """Run the development-only pretrained sweep."""
    parser = build_parser()
    args = parser.parse_args(None if argv is None else list(argv))
    try:
        _validate_cli(args)
        rows = load_manifest(args.manifest, require_labels=True)
        validate_split_safety(args.manifest, args.data_root, rows)
        if args.limit is not None:
            rows = rows[: args.limit]
            if not rows:
                raise EvaluationError("--limit selected no queries")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        model = adapter.load_hatchfinder(args.model, args.device)
        records = run_sweep(
            rows,
            args.data_root,
            model,
            args.output_dir,
            int(args.tile_size),
            int(args.overlap),
            args.device,
            DEFAULT_THRESHOLDS,
        )
        summaries = {
            str(threshold): aggregate(
                [record["scores"][str(threshold)] for record in records]
            )
            for threshold in DEFAULT_THRESHOLDS
        }
        elapsed_seconds = time.monotonic() - started
        report = build_sweep_report(
            rows,
            args.manifest,
            args.model,
            summaries,
            elapsed_seconds,
            DEFAULT_THRESHOLDS,
            device=args.device,
            tile_size=int(args.tile_size),
            overlap=int(args.overlap),
        )
        (args.output_dir / "sweep.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _print_summary(summaries)
    except (EvaluationError, OSError, ValueError) as exc:
        parser.exit(2, "Sweep failed: " + str(exc) + "\n")


def _accumulate_known_tile(
    model: Any,
    rgb: np.ndarray,
    hatch: Any,
    local_known: np.ndarray,
    known_linear: np.ndarray,
    numerator: np.ndarray,
    denominator: np.ndarray,
    y0: int,
    x0: int,
    valid_h: int,
    valid_w: int,
    width: int,
    height: int,
    tile_size: int,
    overlap: int,
    device: str,
) -> None:
    """Forward and blend one tile's known positions into sparse arrays."""
    local_y, local_x = np.nonzero(local_known)
    global_linear = (local_y.astype(np.int64) + y0) * width + local_x + x0
    indices = np.searchsorted(known_linear, global_linear)
    canvas = adapter._tile_canvas(rgb[y0 : y0 + valid_h, x0 : x0 + valid_w], tile_size)
    probabilities = adapter._infer_tile(model, canvas, hatch, valid_h, valid_w, device)
    window = adapter._window(y0, x0, valid_h, valid_w, height, width, overlap)
    weights = window[local_y, local_x]
    numerator[indices] += probabilities[local_y, local_x] * weights
    denominator[indices] += weights
    del canvas, probabilities, window


def _validate_score_inputs(
    rgb: np.ndarray, known: np.ndarray, tile_size: int, overlap: int
) -> None:
    adapter._check_rgb(rgb)
    adapter._validate_tile_params(int(tile_size), int(overlap))
    if known.ndim != 2 or known.shape != rgb.shape[:2]:
        raise ValueError(f"known domain shape {known.shape} does not match {rgb.shape[:2]}")


def _validate_cli(args: argparse.Namespace) -> None:
    adapter._validate_tile_params(int(args.tile_size), int(args.overlap))
    adapter._require_weights(args.model)
    if args.limit is not None and int(args.limit) <= 0:
        raise ValueError("--limit must be positive")
    adapter.resolve_device(args.device)


def _query_tuple(box: Sequence[int]) -> tuple[int, int, int, int]:
    if len(box) != 4 or any(type(value) is not int for value in box):
        raise EvaluationError("query box must contain four integers")
    return (int(box[0]), int(box[1]), int(box[2]), int(box[3]))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _print_progress(
    row: dict[str, Any],
    scores: dict[str, dict[str, Any]],
    thresholds: Sequence[float],
    seconds: float,
) -> None:
    sample_key = str(0.5) if str(0.5) in scores else str(thresholds[0])
    print(
        json.dumps(
            {
                "id": row["id"],
                "seconds": round(seconds, 4),
                f"iou_at_{sample_key}": scores[sample_key]["metrics"]["iou"],
            }
        ),
        flush=True,
    )


def _print_summary(summaries: dict[str, dict[str, Any]]) -> None:
    print(
        json.dumps(
            {
                key: {
                    name: value[name]
                    for name in ("document_macro_iou", "global_precision", "global_recall")
                }
                for key, value in summaries.items()
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
