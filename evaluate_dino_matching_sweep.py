"""Score a DINO pair model on reviewed development pixels and save sparse probabilities."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np

from dino_matching import (
    extract_tiled_features, file_sha256, grayscale_to_rgb, load_dino_model, load_verified_probe, verify_pinned_sources,
    query_embedding, score_known_sparse,
)
from evaluate import data_path, load_manifest, reviewed_domains
from evaluate_hatchfinder_sweep import validate_split_safety
from inference import read_drawing
from train_dino_matching import _query_crop, _manifest_hash


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Development manifest")
    parser.add_argument("--fit-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--limit", type=int)
    return parser


def validate_partitions(
    development_path: Path, fit_path: Path, data_root: Path,
    development_rows: list[dict[str, Any]], fit_rows: list[dict[str, Any]], metadata: dict[str, Any],
) -> None:
    validate_split_safety(development_path, data_root, development_rows)
    if len(development_rows) != 55 or len({str(row["document_id"]) for row in development_rows}) != 12:
        raise ValueError("DINO development sweep requires all 55 queries across 12 documents")
    if any(row.get("kind") != "real" for row in development_rows):
        raise ValueError("DINO development sweep accepts only real queries")
    validate_split_safety(fit_path, data_root, fit_rows)
    dev_ids = {str(row["id"]) for row in development_rows}
    dev_docs = {str(row["document_id"]) for row in development_rows}
    fit_ids = {str(row["id"]) for row in fit_rows}
    fit_docs = {str(row["document_id"]) for row in fit_rows}
    if dev_ids & fit_ids or dev_docs & fit_docs:
        raise ValueError("fit and development manifests share query or document IDs")
    if metadata.get("fit_manifest_sha256") != _manifest_hash(fit_path):
        raise ValueError("model fit manifest hash does not match --fit-manifest")
    if set(metadata.get("training_ids", [])) != fit_ids or set(metadata.get("training_documents", [])) != fit_docs:
        raise ValueError("model training IDs do not match the pinned fit manifest")


def save_known_probabilities(
    directory: Path, row: dict[str, Any], probability: np.ndarray,
    positive: np.ndarray, blank: np.ndarray, y: np.ndarray, x: np.ndarray,
) -> Path:
    target = directory / f"{row['id']}.npz"
    temporary = target.with_suffix(".partial.npz")
    np.savez_compressed(
        temporary, probability=np.asarray(probability, dtype=np.float32),
        positive=np.asarray(positive, dtype=bool), blank=np.asarray(blank, dtype=bool),
        y=np.asarray(y, dtype=np.int32), x=np.asarray(x, dtype=np.int32),
        id=np.asarray(str(row["id"])), document_id=np.asarray(str(row["document_id"])),
        kind=np.asarray(str(row.get("kind", "real"))),
        width=np.asarray(int(row["width"]), dtype=np.int32),
        height=np.asarray(int(row["height"]), dtype=np.int32),
        official_reviewed_pixels=np.asarray(len(y), dtype=np.int64),
    )
    temporary.replace(target)
    return target


def run_development_sweep(
    rows: list[dict[str, Any]], fit_rows: list[dict[str, Any]], data_root: Path,
    model: Any, metadata: dict[str, Any], extractor: Any, output_dir: Path,
) -> list[dict[str, Any]]:
    del fit_rows
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["image"])].append(row)
    output_dir.mkdir(parents=True, exist_ok=True)
    probability_dir = output_dir / "probabilities"
    probability_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    with (output_dir / "query-records.jsonl").open("w", encoding="utf-8") as log:
        for group in groups.values():
            gray = read_drawing(data_root, group[0])
            rgb = grayscale_to_rgb(gray)
            grid_features = extract_tiled_features(rgb, extractor)
            for row in group:
                started = time.monotonic()
                if rgb.shape[:2] != (row["height"], row["width"]):
                    raise ValueError(f"drawing dimensions do not match development query {row['id']}")
                query = query_embedding(_query_crop(rgb, row), extractor)
                positive, known, blank, _ = reviewed_domains(row, data_root)
                y, x = np.nonzero(known)
                sparse = score_known_sparse(
                    model, grid_features, query, y, x, int(row["height"]), int(row["width"])
                )
                positive_known = positive[known]
                blank_known = blank[known]
                cache_path = save_known_probabilities(
                    probability_dir, row, sparse, positive_known, blank_known, y, x
                )
                record = {
                    "id": str(row["id"]), "document_id": str(row["document_id"]),
                    "kind": str(row.get("kind", "real")), "seconds": time.monotonic() - started,
                    "known_probabilities": str(cache_path.relative_to(output_dir)),
                    "known_pixels": len(y), "model_signature": metadata["model_signature"],
                    "native_shape": [int(row["height"]), int(row["width"])],
                    "patch_grid": [int(grid_features.shape[0]), int(grid_features.shape[1])],
                    "cache_sha256": file_sha256(cache_path),
                }
                records.append(record)
                log.write(json.dumps(record) + "\n")
                log.flush()
            del grid_features, rgb, gray
    return records



def input_asset_digests(rows: list[dict[str, Any]], data_root: Path) -> dict[str, dict[str, Any]]:
    paths = set()
    for row in rows:
        paths.add(str(row["image"]))
        for key, relative in row.get("labels", {}).items():
            if key.endswith("_mask"):
                paths.add(str(relative))
    return {
        relative: {"sha256": file_sha256(data_path(data_root, relative)), "bytes": data_path(data_root, relative).stat().st_size}
        for relative in sorted(paths)
    }


def write_cache_report(
    output_dir: Path, rows: list[dict[str, Any]], fit_path: Path, dev_path: Path,
    model_path: Path, metadata: dict[str, Any], records: list[dict[str, Any]],
    data_root: Path, elapsed: float,
) -> None:
    implementation_paths = (
        Path(__file__), Path("dino_matching.py"), Path("tools/dino_feature_probe.py"),
        Path("evaluate.py"), Path("evaluate_texture_sweep.py"), Path("inference.py"),
        Path("train_dino_matching.py"), Path("tools/dino_patch_support.py"),
        Path("tools/analyze_research_pareto.py"),
    )
    source_hashes = {str(path): file_sha256(path) for path in implementation_paths}
    assets = input_asset_digests(rows, data_root)
    cache_records = sorted(({"id": record["id"], "sha256": record["cache_sha256"]} for record in records), key=lambda record: record["id"])
    cache_manifest = json.dumps(cache_records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    protocol_path = Path("artifacts/research-v4/dino-classifier-protocol.json")
    protocol_sha256 = file_sha256(protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    report = {
        "schema": "dino-matching-development-cache/v1",
        "protocol_sha256": protocol_sha256,
        "predeclared_incumbents": protocol["exact_incumbents"],
        "fit_manifest_sha256": _manifest_hash(fit_path),
        "development_manifest_sha256": _manifest_hash(dev_path),
        "model_artifact_sha256": metadata["artifact_sha256"],
        "model_signature": metadata["model_signature"],
        "support_evidence_sha256": metadata.get("support_evidence_sha256"),
        "fit_asset_mapping_sha256": metadata.get("fit_asset_mapping_sha256"),
        "feature_implementation_sha256": source_hashes[str(Path("dino_matching.py"))],
        "source_hashes": source_hashes,
        "input_asset_mapping_sha256": hashlib.sha256(json.dumps(assets, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(),
        "input_assets": assets,
        "cache_ids_and_sha256": cache_records,
        "cache_mapping_sha256": hashlib.sha256(cache_manifest).hexdigest(),
        "native_scoring": "exact known pixels; predict unique bilinear-neighbor cells only; cv2.INTER_LINEAR remap",
        "patch_size": 14,
        "tile": {"input": 896, "core": 672, "context_margin_each_side": 112},
        "queries": len(records),
        "complete_development_manifest": len(records) == len(rows) == 55,
        "exact_known_only": True,
        "elapsed_seconds": elapsed,
        "probability_dir": "probabilities",
        "records": records,
    }
    (output_dir / "sweep.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    sweep_started = time.monotonic()
    development_rows = load_manifest(args.manifest, require_labels=True)
    fit_rows = load_manifest(args.fit_manifest, require_labels=True)
    metadata_path = args.model.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    validate_partitions(args.manifest, args.fit_manifest, args.data_root, development_rows, fit_rows, metadata)
    pins = verify_pinned_sources(args.source, args.weights)
    if any(pins.get(key) != metadata.get(key) for key in ("source_commit", "checkpoint_sha256")):
        raise SystemExit("runtime DINO source/checkpoint pins differ from the trained model")
    model, metadata = load_dino_model(args.model)
    if args.limit is not None:
        if args.limit <= 0:
            raise SystemExit("--limit must be positive")
        development_rows = development_rows[:args.limit]
    try:
        extractor, _ = load_verified_probe(
            args.source, args.weights, args.device, metadata, verified_pins=pins
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    records = run_development_sweep(
        development_rows, fit_rows, args.data_root, model, metadata, extractor, args.output_dir
    )
    write_cache_report(
        args.output_dir, development_rows, args.fit_manifest, args.manifest,
        args.model, metadata, records, args.data_root, time.monotonic() - sweep_started,
    )


if __name__ == "__main__":
    main()
