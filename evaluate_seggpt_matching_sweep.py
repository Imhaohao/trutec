"""Score frozen SegGPT on the reviewed native development coordinates."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable, Sequence

import numpy as np

import seggpt_matching
from evaluate import data_path, load_manifest, reviewed_domains
from inference import read_drawing
from dino_matching import file_sha256, grayscale_to_rgb

PROTOCOL_PATH = Path("artifacts/research-v4/seggpt-development-protocol.json")
PROTOCOL_SHA256 = "9950eac8fa9ca277e89e31626adfd3135e591a558c7382a45699f3ed96883dcb"
EXPECTED_QUERIES = 55
EXPECTED_DOCUMENTS = 12
EXPECTED_KNOWN_PIXELS = 3_465_164
TILE_SIDE = 448
THRESHOLDS = np.linspace(0.0, 1.0, 101, dtype=np.float64)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def score_reviewed_coordinates(
    image: np.ndarray,
    query_crop: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    predictor: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> np.ndarray:
    """Score row-major native points using each occupied global 448 tile once."""
    gray = np.asarray(image)
    if gray.ndim != 2 or gray.dtype != np.uint8 or min(gray.shape) <= 0:
        raise ValueError("drawing must be a nonempty native uint8 grayscale image")
    yy, xx = _validate_coordinates(y, x, gray.shape)
    height, width = gray.shape
    tile_columns = (width + TILE_SIDE - 1) // TILE_SIDE
    tile_ids = (yy // TILE_SIDE) * tile_columns + (xx // TILE_SIDE)
    unique_tiles = np.unique(tile_ids)
    gathered = np.empty(len(yy), dtype=np.float32)
    for tile_id in unique_tiles:
        selected = tile_ids == tile_id
        tile_y, tile_x = divmod(int(tile_id), tile_columns)
        y0, x0 = tile_y * TILE_SIDE, tile_x * TILE_SIDE
        y1, x1 = min(y0 + TILE_SIDE, height), min(x0 + TILE_SIDE, width)
        native_tile = grayscale_to_rgb(gray[y0:y1, x0:x1])
        tile_scores = np.asarray(predictor(native_tile, query_crop), dtype=np.float32)
        if tile_scores.shape != native_tile.shape[:2]:
            raise ValueError("SegGPT predictor must return scores for the exact native tile area")
        if not np.isfinite(tile_scores).all() or np.any((tile_scores < 0) | (tile_scores > 1)):
            raise ValueError("SegGPT tile scores must be finite float values in [0, 1]")
        local_y = yy[selected] - tile_y * TILE_SIDE
        local_x = xx[selected] - tile_x * TILE_SIDE
        gathered[selected] = tile_scores[local_y, local_x]
    return gathered


def _validate_coordinates(y: np.ndarray, x: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.asarray(y), np.asarray(x)
    if yy.ndim != 1 or xx.ndim != 1 or yy.shape != xx.shape:
        raise ValueError("reviewed coordinates must be aligned one-dimensional arrays")
    if yy.dtype.kind not in "iu" or xx.dtype.kind not in "iu":
        raise ValueError("reviewed coordinates must use integer dtypes")
    if np.any((yy < 0) | (yy >= shape[0]) | (xx < 0) | (xx >= shape[1])):
        raise ValueError("reviewed coordinates fall outside the native drawing")
    linear = yy.astype(np.int64) * shape[1] + xx.astype(np.int64)
    if len(linear) and np.any(np.diff(linear) <= 0):
        raise ValueError("reviewed coordinates must be unique and row-major")
    return yy.astype(np.int32, copy=False), xx.astype(np.int32, copy=False)


def save_known_scores(
    directory: Path,
    row: dict[str, Any],
    score: np.ndarray,
    positive: np.ndarray,
    blank: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    metadata: dict[str, str],
) -> Path:
    """Persist one native row-major score vector and its audit metadata."""
    target = directory / f"{row['id']}.npz"
    temporary = target.with_suffix(".partial.npz")
    arrays = {
        "probability": np.asarray(score, dtype=np.float32),
        "seggpt_score": np.asarray(score, dtype=np.float32),
        "score_semantics": np.asarray("palette_distance_score: normalized black/white distance margin; not calibrated probability"),
        "positive": np.asarray(positive, dtype=np.bool_), "blank": np.asarray(blank, dtype=np.bool_),
        "y": np.asarray(y, dtype=np.int32), "x": np.asarray(x, dtype=np.int32),
        "id": np.asarray(str(row["id"])), "document_id": np.asarray(str(row["document_id"])),
        "kind": np.asarray(str(row.get("kind", "real"))),
        "width": np.asarray(int(row["width"]), dtype=np.int32),
        "height": np.asarray(int(row["height"]), dtype=np.int32),
        "official_reviewed_pixels": np.asarray(len(y), dtype=np.int64),
        **{key: np.asarray(value) for key, value in metadata.items()},
    }
    np.savez_compressed(temporary, **arrays)  # type: ignore[arg-type]
    temporary.replace(target)
    return target


def run_development_sweep(
    rows: list[dict[str, Any]],
    data_root: Path,
    model: Any,
    processor: Any,
    output_dir: Path,
    torch_module: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run sequential query and occupied-tile forwards and save sparse caches."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["image"])].append(row)
    output_dir.mkdir(parents=True, exist_ok=True)
    score_dir = output_dir / "probabilities"
    score_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.jsonl"
    records: list[dict[str, Any]] = []
    memory_snapshots: list[dict[str, int | None]] = []
    with progress_path.open("w", encoding="utf-8") as progress:
        for group in groups.values():
            gray = read_drawing(data_root, group[0])
            for row in group:
                started = time.monotonic()
                if gray.shape != (int(row["height"]), int(row["width"])):
                    raise ValueError(f"drawing dimensions do not match development query {row['id']}")
                positive, known, blank, _ = reviewed_domains(row, data_root)
                y, x = np.nonzero(known)
                x0, y0, x1, y1 = row["query_box"]
                query_crop = grayscale_to_rgb(gray[y0:y1, x0:x1])

                def predict(tile: np.ndarray, prompt: np.ndarray) -> np.ndarray:
                    memory_snapshots.append(_mps_snapshot(torch_module))
                    result = seggpt_matching.predict_tile(model, processor, tile, prompt, "mps", torch_module)
                    torch_module.mps.synchronize()
                    memory_snapshots.append(_mps_snapshot(torch_module))
                    return result

                scores = score_reviewed_coordinates(gray, query_crop, y, x, predict)
                positive_known, blank_known = positive[known], blank[known]
                cache_metadata = _cache_metadata(row, query_crop, data_root)
                cache_path = save_known_scores(
                    score_dir, row, scores, positive_known, blank_known, y, x, cache_metadata
                )
                record = {
                    "id": str(row["id"]), "document_id": str(row["document_id"]),
                    "seconds": time.monotonic() - started,
                    "known_scores": str(cache_path.relative_to(output_dir)),
                    "known_pixels": len(y), "occupied_tiles": int(len(np.unique(
                        (y // TILE_SIDE) * ((int(row["width"]) + TILE_SIDE - 1) // TILE_SIDE) + x // TILE_SIDE
                    ))),
                    "native_shape": [int(row["height"]), int(row["width"])],
                    "score_range": [float(scores.min()), float(scores.max())],
                    "cache_sha256": file_sha256(cache_path),
                    "original_image_sha256": cache_metadata["original_image_sha256"],
                    "query_crop_sha256": cache_metadata["query_crop_sha256"],
                }
                records.append(record)
                progress.write(json.dumps({
                    "completed_queries": len(records), "total_queries": len(rows),
                    "completed_occupied_tiles": sum(int(item["occupied_tiles"]) for item in records),
                }) + "\n")
                progress.flush()
                print(json.dumps({"completed_queries": len(records), "total_queries": len(rows)}), flush=True)
    return records, memory_snapshots


def _cache_metadata(row: dict[str, Any], query: np.ndarray, data_root: Path) -> dict[str, str]:
    return {
        "original_image_sha256": file_sha256(data_path(data_root, str(row["image"]))),
        "query_crop_sha256": hashlib.sha256(np.ascontiguousarray(query).tobytes()).hexdigest(),
        "manifest_sha256": file_sha256(Path("artifacts/splits/development-real.json")),
        "protocol_sha256": PROTOCOL_SHA256,
        "model_weights_sha256": seggpt_matching.WEIGHTS_SHA256,
        "model_config_sha256": seggpt_matching.CONFIG_SHA256,
        "processor_config_sha256": seggpt_matching.PREPROCESSOR_CONFIG_SHA256,
        "model_source_commit": seggpt_matching.MODEL_SOURCE_COMMIT,
        "model_source_url": seggpt_matching.MODEL_SOURCE_URL,
        "processor_source_url": seggpt_matching.PROCESSOR_SOURCE_URL,
        "adapter_source_sha256": file_sha256(Path("seggpt_matching.py")),
        "sweep_source_sha256": file_sha256(Path(__file__)),
        "row_id": str(row["id"]),
    }


def _mps_snapshot(torch_module: Any) -> dict[str, int | None]:
    mps = torch_module.mps
    return {
        "current_allocated_bytes": _counter(mps, "current_allocated_memory"),
        "driver_allocated_bytes": _counter(mps, "driver_allocated_memory"),
        "recommended_max_bytes": _counter(mps, "recommended_max_memory"),
    }


def _counter(mps: Any, name: str) -> int | None:
    counter = getattr(mps, name, None)
    return int(counter()) if callable(counter) else None


def _memory_drift_summary(snapshots: list[dict[str, int | None]]) -> dict[str, Any]:
    after_forward = snapshots[1::2]
    allocated = [item["current_allocated_bytes"] for item in after_forward]
    measured = [value for value in allocated if value is not None]
    if not measured:
        return {"sampled_forward_count": len(after_forward), "current_allocated_delta_bytes": None}
    return {
        "sampled_forward_count": len(after_forward),
        "first_after_forward_allocated_bytes": measured[0],
        "last_after_forward_allocated_bytes": measured[-1],
        "current_allocated_delta_bytes": measured[-1] - measured[0],
        "maximum_sampled_after_forward_allocated_bytes": max(measured),
        "interpretation": "sampled live counters across forwards; not a high-water mark",
    }


def _asset_digests(rows: list[dict[str, Any]], data_root: Path) -> dict[str, dict[str, Any]]:
    paths = {str(row["image"]) for row in rows}
    for row in rows:
        paths.update(str(value) for key, value in row["labels"].items() if key.endswith("_mask"))
    result = {}
    for relative in sorted(paths):
        path = data_path(data_root, relative)
        result[relative] = {"sha256": file_sha256(path), "bytes": path.stat().st_size}
    return result


def validate_protocol_inputs(
    manifest_path: Path, rows: list[dict[str, Any]], data_root: Path
) -> dict[str, Any]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if file_sha256(PROTOCOL_PATH) != PROTOCOL_SHA256:
        raise ValueError("frozen SegGPT development protocol hash changed")
    if manifest_path.resolve() != Path("artifacts/splits/development-real.json").resolve():
        raise ValueError("manifest path differs from the frozen development manifest")
    if len(rows) != EXPECTED_QUERIES or len({str(row["document_id"]) for row in rows}) != EXPECTED_DOCUMENTS:
        raise ValueError("SegGPT sweep requires the frozen 55-query, 12-document manifest")
    if any(row.get("kind") != "real" for row in rows):
        raise ValueError("SegGPT sweep accepts only the frozen real-query manifest")
    expected_pins = protocol["pre_inference_sha256"]
    for relative, expected in expected_pins.items():
        path = Path(relative)
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"frozen protocol input pin changed: {relative}")
    model_pins = {
        "models/texture-hgb-crop-v1.joblib": protocol["controls"]["texture_sha256"],
        "models/hatchfinder-adapt-v1.pt": protocol["controls"]["cnn_sha256"],
    }
    for relative, expected in model_pins.items():
        if file_sha256(Path(relative)) != expected:
            raise ValueError(f"frozen comparison model pin changed: {relative}")
    known_count = 0
    occupied_tile_count = 0
    for row in rows:
        _, known, _, _ = reviewed_domains(row, data_root)
        known_count += int(known.sum())
        y, x = np.nonzero(known)
        tile_columns = (int(row["width"]) + TILE_SIDE - 1) // TILE_SIDE
        occupied_tile_count += int(len(np.unique(
            (y // TILE_SIDE) * tile_columns + x // TILE_SIDE
        )))
    if known_count != EXPECTED_KNOWN_PIXELS:
        raise ValueError(f"known coordinate count differs from protocol: {known_count}")
    if occupied_tile_count != int(protocol["geometry"]["required_forwards"]):
        raise ValueError(f"occupied tile forward count differs from protocol: {occupied_tile_count}")
    _control_cache_signatures(rows, protocol)
    return protocol


def _load_frozen_model(checkpoint_dir: Path) -> tuple[Any, Any, Any, dict[str, Any]]:
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for SegGPT inference") from exc
    model, processor, pins = seggpt_matching.load_local_model(checkpoint_dir, torch)
    return model, processor, torch, pins


def cache_directory_sha256(directory: Path) -> str:
    """Hash sorted UTF-8 filenames followed by file bytes, without separators."""
    digest = hashlib.sha256()
    paths = sorted(directory.glob("*.npz"), key=lambda item: item.name.encode("utf-8"))
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _verify_control_cache(
    directory: Path, expected_ids: set[str], expected_digest: str, name: str
) -> list[dict[str, str]]:
    paths = {path.stem: path for path in directory.glob("*.npz")}
    if set(paths) != expected_ids:
        raise ValueError(f"{name} frozen cache IDs do not match the development manifest")
    if cache_directory_sha256(directory) != expected_digest:
        raise ValueError(f"{name} frozen cache directory hash does not match the paired evaluation plan")
    return [
        {"id": identifier, "sha256": file_sha256(paths[identifier])}
        for identifier in sorted(expected_ids)
    ]


def _control_cache_signatures(
    rows: list[dict[str, Any]], protocol: dict[str, Any]
) -> dict[str, list[dict[str, str]]]:
    expected = {str(row["id"]) for row in rows}
    provenance = protocol["controls"]["cache_provenance"]
    plan_path = Path(provenance["path"])
    if file_sha256(plan_path) != provenance["sha256"]:
        raise ValueError("paired evaluation plan hash does not match the frozen SegGPT protocol")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    original = plan["models_and_existing_inputs"]
    controls = {
        "original_hgb": (
            original["original_hgb"]["cache_dir"],
            provenance["original_hgb_cache_dir_sha256"],
            protocol["controls"]["texture_cache"],
        ),
        "original_cnn": (
            original["original_cnn"]["cache_dir"],
            provenance["original_cnn_cache_dir_sha256"],
            protocol["controls"]["cnn_cache"],
        ),
    }
    result = {}
    for name, (directory, expected_digest, protocol_directory) in controls.items():
        if Path(directory) != Path(protocol_directory):
            raise ValueError(f"{name} cache directory differs from the frozen protocol")
        if expected_digest != original[name]["cache_dir_sha256"]:
            raise ValueError(f"{name} cache digest differs from the pinned paired evaluation plan")
        result[name] = _verify_control_cache(Path(directory), expected, expected_digest, name)
    return result


def _write_reports(
    output_dir: Path,
    rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
    memory_snapshots: list[dict[str, int | None]],
    protocol: dict[str, Any],
    pins: dict[str, Any],
    data_root: Path,
    elapsed: float,
) -> None:
    assets = _asset_digests(rows, data_root)
    cache_ids = sorted(
        [{"id": record["id"], "sha256": record["cache_sha256"]} for record in records],
        key=lambda item: item["id"],
    )
    source_paths = (
        Path(__file__), Path("seggpt_matching.py"), Path("evaluate.py"), Path("inference.py"),
        Path("dino_matching.py"), Path("tools/dino_feature_probe.py"),
    )
    source_hashes = {str(path): file_sha256(path) for path in source_paths}
    controls = _control_cache_signatures(rows, protocol)
    occupied_tile_count = sum(int(record["occupied_tiles"]) for record in records)
    if occupied_tile_count != int(protocol["geometry"]["required_forwards"]):
        raise ValueError(f"occupied tile forwards differ from protocol: {occupied_tile_count}")
    cache_mapping = json.dumps(cache_ids, sort_keys=True, separators=(",", ":")).encode("utf-8")
    asset_mapping = json.dumps(assets, sort_keys=True, separators=(",", ":")).encode("utf-8")
    report = {
        "schema": "seggpt-matching-development-cache/v1",
        "protocol_sha256": PROTOCOL_SHA256,
        "manifest_sha256": file_sha256(Path("artifacts/splits/development-real.json")),
        "model": pins,
        "comparison_models": {
            "original_texture_sha256": protocol["controls"]["texture_sha256"],
            "original_cnn_sha256": protocol["controls"]["cnn_sha256"],
        },
        "source_hashes": source_hashes,
        "protocol_input_pins": protocol["pre_inference_sha256"],
        "input_asset_mapping_sha256": hashlib.sha256(json.dumps(assets, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "input_assets": assets,
        "known_pixels": sum(int(record["known_pixels"]) for record in records),
        "expected_known_pixels": EXPECTED_KNOWN_PIXELS,
        "occupied_tile_forwards": occupied_tile_count,
        "expected_occupied_tile_forwards": int(protocol["geometry"]["required_forwards"]),
        "queries": len(records), "documents": len({record["document_id"] for record in records}),
        "complete_development_manifest": len(records) == EXPECTED_QUERIES,
        "score_semantics": "normalized black/white RGB palette-distance margin; bounded float32; not a calibrated probability",
        "cache_ids_and_sha256": cache_ids,
        "cache_mapping_sha256": hashlib.sha256(cache_mapping).hexdigest(),
        "frozen_control_cache_ids_and_sha256": controls,
        "frozen_control_cache_mapping_sha256": {
            name: hashlib.sha256(json.dumps(items, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            for name, items in controls.items()
        },
        "asset_mapping_sha256": hashlib.sha256(asset_mapping).hexdigest(),
        "records": records,
        "threshold_grid": THRESHOLDS.tolist(),
        "standalone_initial_sweep": {
            "threshold_count": len(THRESHOLDS),
            "threshold_comparison": "float32 native scores >= threshold; analyzer casts locked thresholds to score dtype",
            "analyzer": "tools/analyze_research_pareto.py",
            "legacy_cache_key": "probability",
            "invocation": [
                "python", "tools/analyze_research_pareto.py",
                "--manifest", "artifacts/splits/development-real.json",
                "--data-root", str(data_root),
                "--cnn-cache", "artifacts/hatchfinder-adapt-development-v1/probabilities",
                "--texture-cache", "original_hgb=artifacts/texture-crop-development-v1/probabilities",
                "--texture-cache", f"seggpt={output_dir / 'probabilities'}",
                "--output", str(output_dir / "seggpt-vs-original-cnn.json"),
            ],
        },
        "runtime": {
            "sequential": True, "batch_size": 1, "dtype": "float32", "device": "mps",
            "cpu_fallback": False, "elapsed_seconds": elapsed,
            "internal_mps_memory_snapshots": memory_snapshots,
            "internal_mps_drift": _memory_drift_summary(memory_snapshots),
            "memory_limitation": "MPS snapshots show live counters only and are not high-water marks.",
        },
        "protocol_selection": protocol["selection"],
        "protocol_final_gates": protocol["final_gates"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sweep.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (output_dir / "query-records.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(None if argv is None else list(argv))
    started = time.monotonic()
    rows = load_manifest(args.manifest, require_labels=True)
    protocol = validate_protocol_inputs(args.manifest, rows, args.data_root)
    model, processor, torch, pins = _load_frozen_model(args.checkpoint_dir)
    records, memory = run_development_sweep(rows, args.data_root, model, processor, args.output_dir, torch)
    _write_reports(args.output_dir, rows, records, memory, protocol, pins, args.data_root, time.monotonic() - started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
