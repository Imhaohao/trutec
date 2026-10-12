"""Fit the locked H5 faint-feature model after matching its H2 samples exactly."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

import hatch_texture
from evaluate import EvaluationError, load_manifest
import texture_learning
from texture_learning import pair_feature_names, save_texture_model
import train_texture

STRIDE = 4
SAMPLES_PER_CLASS = 512
SEED = 42
MAX_ITER = 200
EXPECTED_PARAMS = {
    "learning_rate": 0.08,
    "max_leaf_nodes": 31,
    "l2_regularization": 1.0,
    "max_iter": MAX_ITER,
    "early_stopping": False,
}
BASE_CHANNELS = 46


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-manifest", required=True, type=Path)
    parser.add_argument("--development-manifest", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--h2-cache", required=True, type=Path, help="Compressed NPZ with x, y, and weights arrays.")
    parser.add_argument("--h2-metadata", required=True, type=Path, help="JSON sidecar for the H2 distribution model.")
    parser.add_argument("--output-model", required=True, type=Path)
    parser.add_argument("--output-report", required=True, type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_h2_cache(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"x", "y", "weight"}
        if not required.issubset(archive.files):
            raise ValueError(f"H2 cache must contain arrays {sorted(required)}")
        x, y, weights = archive["x"], archive["y"], archive["weight"]
    if x.ndim != 2 or y.shape != (len(x),) or weights.shape != (len(x),):
        raise ValueError("H2 cache arrays have incompatible sample dimensions")
    if x.dtype != np.float32 or y.dtype != np.int32 or weights.dtype != np.float32:
        raise ValueError("H2 cache must use float32 x/weights and int32 y arrays")
    return x, y, weights


def validate_h2_metadata(metadata: dict[str, Any], fit_manifest: Path) -> list[str]:
    expected_names = pair_feature_names(hatch_texture.texture_feature_names("base"), "distribution")
    if metadata.get("query_descriptor_source") != "distribution":
        raise ValueError("H2 metadata must describe distribution query features")
    if metadata.get("texture_feature_set") != "base":
        raise ValueError("H2 control metadata must describe the base feature set")
    if metadata.get("stride") != STRIDE or metadata.get("seed") != SEED:
        raise ValueError("H2 metadata does not match the locked stride and seed")
    if metadata.get("samples_per_class") != SAMPLES_PER_CLASS or metadata.get("params") != EXPECTED_PARAMS:
        raise ValueError("H2 metadata does not match the locked sample cap or classifier parameters")
    if metadata.get("source_manifest_sha256") != sha256_file(fit_manifest):
        raise ValueError("Fit manifest does not match the H2 source manifest")
    if metadata.get("pair_feature_names") != expected_names:
        raise ValueError("H2 metadata pair feature names do not match the base distribution signature")
    return expected_names


def validate_fit_development_split(
    fit_rows: list[dict[str, Any]],
    development_rows: list[dict[str, Any]],
    fit_manifest: Path,
    development_manifest: Path,
    data_root: Path,
) -> dict[str, list[str]]:
    fit_documents = _document_ids(fit_rows, "fit")
    development_documents = _document_ids(development_rows, "development")
    if fit_documents & development_documents:
        raise EvaluationError("Fit and development manifests share document IDs")
    validation_documents = train_texture.get_validation_document_ids(data_root)
    if fit_documents & validation_documents or development_documents & validation_documents:
        raise EvaluationError("Fit or development manifest overlaps public validation documents")
    if fit_manifest.name == "val.json" or development_manifest.name == "val.json":
        raise EvaluationError("Public validation manifest cannot be used for fitting or development")
    return {
        "fit": sorted(fit_documents),
        "development": sorted(development_documents),
    }


def _document_ids(rows: list[dict[str, Any]], partition: str) -> set[str]:
    documents = {row.get("document_id") for row in rows}
    if not rows or None in documents or "" in documents:
        raise EvaluationError(f"{partition} manifest must contain document IDs for every row")
    return {str(document_id) for document_id in documents}


def _validate_h2_coverage(metadata: dict[str, Any], fit_rows: list[dict[str, Any]]) -> None:
    expected_ids = sorted(row["id"] for row in fit_rows)
    expected_documents = sorted(_document_ids(fit_rows, "fit"))
    if metadata.get("training_ids") != expected_ids:
        raise ValueError("Fit manifest query IDs do not match H2 metadata")
    if metadata.get("training_documents") != expected_documents:
        raise ValueError("Fit manifest document IDs do not match H2 metadata")


def _validate_cache_coverage(
    metadata: dict[str, Any], x: np.ndarray, y: np.ndarray
) -> None:
    coverage = metadata.get("training_coverage")
    if not isinstance(coverage, dict):
        raise ValueError("H2 metadata must include training coverage")
    expected = {
        "total_samples": len(y),
        "positive_samples": int(np.count_nonzero(y == 1)),
        "negative_samples": int(np.count_nonzero(y == 0)),
    }
    if x.shape[1] != len(metadata["pair_feature_names"]):
        raise ValueError("H2 cache feature width differs from its metadata signature")
    if any(coverage.get(key) != value for key, value in expected.items()):
        raise ValueError("H2 cache sample counts do not match training metadata")


def base_column_indices(
    faint_feature_names: list[str], h2_feature_names: list[str]
) -> list[int]:
    if len(set(faint_feature_names)) != len(faint_feature_names):
        raise ValueError("Faint feature signature contains duplicate names")
    faint_positions = {name: index for index, name in enumerate(faint_feature_names)}
    missing = [name for name in h2_feature_names if name not in faint_positions]
    if missing:
        raise ValueError(f"Faint feature signature is missing {len(missing)} H2 base columns")
    return [faint_positions[name] for name in h2_feature_names]


def assert_h2_samples_match(
    faint_x: np.ndarray,
    faint_y: np.ndarray,
    faint_weights: np.ndarray,
    h2_x: np.ndarray,
    h2_y: np.ndarray,
    h2_weights: np.ndarray,
    faint_names: list[str],
    h2_names: list[str],
) -> dict[str, Any]:
    indices = base_column_indices(faint_names, h2_names)
    if faint_x.shape != (len(faint_y), len(faint_names)):
        raise ValueError("Faint fit arrays do not match their feature signature")
    if h2_x.shape != (len(h2_y), len(h2_names)):
        raise ValueError("H2 cache matrix does not match its feature signature")
    if not np.array_equal(faint_y, h2_y):
        raise ValueError("Faint fit labels differ from the H2 cache")
    if not np.array_equal(faint_weights, h2_weights):
        raise ValueError("Faint fit sample weights differ from the H2 cache")
    if len(faint_x) != len(h2_x):
        raise ValueError("Faint fit and H2 cache contain different sample counts")
    chunk_rows = 4096
    for start in range(0, len(faint_x), chunk_rows):
        stop = min(start + chunk_rows, len(faint_x))
        if not np.array_equal(faint_x[start:stop, indices], h2_x[start:stop]):
            raise ValueError(f"Faint base feature samples differ from H2 in rows {start}:{stop}")
    return {
        "matches": True,
        "rows": len(faint_x),
        "base_columns": len(h2_names),
        "label_matches": True,
        "weight_matches": True,
        "comparison_chunk_rows": chunk_rows,
    }


def _validation_manifest(data_root: Path) -> Path:
    candidates = [data_root / "val.json", Path("dataset/val.json")]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("Cannot verify split isolation without a validation manifest for document IDs")


def run(args: argparse.Namespace) -> dict[str, Any]:
    start_time = time.monotonic()
    h2_metadata = json.loads(args.h2_metadata.read_text(encoding="utf-8"))
    if not isinstance(h2_metadata, dict):
        raise ValueError("H2 metadata must be a JSON object")
    h2_names = validate_h2_metadata(h2_metadata, args.fit_manifest)
    fit_rows = load_manifest(args.fit_manifest, require_labels=True)
    development_rows = load_manifest(args.development_manifest, require_labels=True)
    _validate_h2_coverage(h2_metadata, fit_rows)
    validation_manifest = _validation_manifest(args.data_root)
    split_documents = validate_fit_development_split(
        fit_rows, development_rows, args.fit_manifest, args.development_manifest, args.data_root
    )
    train_texture.validate_split_safety(args.fit_manifest, args.data_root, fit_rows)
    train_texture.validate_split_safety(
        args.development_manifest, args.data_root, development_rows
    )
    h2_x, h2_y, h2_weights = read_h2_cache(args.h2_cache)
    _validate_cache_coverage(h2_metadata, h2_x, h2_y)

    collect_start = time.monotonic()
    faint_names = pair_feature_names(hatch_texture.texture_feature_names("faint"), "distribution")
    faint_x, faint_y, faint_weights = train_texture.collect_training_dataset(
        fit_rows,
        args.data_root,
        stride=STRIDE,
        samples_per_class=SAMPLES_PER_CLASS,
        base_seed=SEED,
        query_source="distribution",
        feature_set="faint",
    )
    collect_seconds = time.monotonic() - collect_start
    sample_check = assert_h2_samples_match(
        faint_x, faint_y, faint_weights, h2_x, h2_y, h2_weights, faint_names, h2_names
    )
    del h2_x, h2_y, h2_weights

    fit_start = time.monotonic()
    model = train_texture.train_classifier(faint_x, faint_y, faint_weights, MAX_ITER, SEED)
    fit_seconds = time.monotonic() - fit_start
    if getattr(model, "n_features_in_", None) != len(faint_names):
        raise ValueError("Fitted H5 model feature count does not match its metadata signature")
    model_metadata = train_texture.build_metadata(
        argparse.Namespace(
            query_source="distribution",
            features="faint",
            max_iter=MAX_ITER,
            seed=SEED,
            samples_per_class=SAMPLES_PER_CLASS,
        ),
        sha256_file(args.fit_manifest),
        hatch_texture.texture_feature_names("faint"),
        fit_rows,
        faint_y,
        time.monotonic() - start_time,
    )
    model_metadata["h5_control"] = {
        "reference_h2_cache_sha256": sha256_file(args.h2_cache),
        "reference_h2_metadata_sha256": sha256_file(args.h2_metadata),
        "sample_equality": sample_check,
    }
    save_texture_model(model, model_metadata, args.output_model)
    report = {
        "schema": "faint-control-training/v1",
        "status": "SUCCESS",
        "fit": {
            "queries": len(fit_rows),
            "documents": len(split_documents["fit"]),
            "samples": len(faint_y),
            "feature_columns": faint_x.shape[1],
            "collection_seconds": collect_seconds,
            "fit_seconds": fit_seconds,
            "fit_cache_written": False,
        },
        "development_split": {
            "queries": len(development_rows),
            "documents": len(split_documents["development"]),
            "document_overlap_with_fit": 0,
            "public_validation_document_overlap": 0,
            "validation_document_id_source": str(validation_manifest),
            "validation_manifest_sha256": sha256_file(validation_manifest),
        },
        "sample_equality": sample_check,
        "model_path": str(args.output_model),
        "model_sha256": sha256_file(args.output_model),
        "source_hashes": _source_hashes(args),
        "elapsed_seconds": time.monotonic() - start_time,
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _source_hashes(args: argparse.Namespace) -> dict[str, str]:
    source_paths = {
        "driver": Path(__file__).resolve(),
        "hatch_texture.py": Path(hatch_texture.__file__).resolve(),
        "texture_learning.py": Path(texture_learning.__file__).resolve(),
        "train_texture.py": Path(train_texture.__file__).resolve(),
        "fit_manifest": args.fit_manifest,
        "development_manifest": args.development_manifest,
        "h2_cache": args.h2_cache,
        "h2_metadata": args.h2_metadata,
    }
    return {key: sha256_file(path) for key, path in source_paths.items()}


def main() -> None:
    report = run(parse_args())
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
