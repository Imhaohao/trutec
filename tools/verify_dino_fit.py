"""Verify a fitted DINO classifier using fit labels and its recorded sampling audit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dino_matching import FEATURE_SIGNATURE, MODEL_PARAMS, SAMPLES_PER_CLASS, load_dino_model
from evaluate import EvaluationError, data_path, load_manifest, reviewed_domains
from tools.dino_patch_support import FIT_MANIFEST_SHA256, verify_fit_manifest
from train_dino_matching import verify_fit_assets, verify_support_evidence
from train_hatchfinder import check_split_safety

PROTOCOL = ROOT / "artifacts/research-v4/dino-classifier-protocol.json"
ASSET_PROOF = ROOT / "artifacts/research-v4/fit-input-assets.json"
SUPPORT = ROOT / "artifacts/research-v4/dino-patch-support-fit.json"
PATCH = 14


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"Cannot read JSON file: {path}") from exc
    if not isinstance(value, dict):
        raise EvaluationError(f"Expected JSON object: {path}")
    return value


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _same(a: Any, b: Any, description: str) -> None:
    if a != b:
        raise EvaluationError(f"DINO fit {description} mismatch")


def _verify_model(model_path: Path) -> tuple[Any, dict[str, Any]]:
    try:
        model, metadata = load_dino_model(model_path)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"Cannot verify DINO artifact: {exc}") from exc
    if type(model) is not HistGradientBoostingClassifier:
        raise EvaluationError("Estimator is not the locked HistGradientBoostingClassifier")
    _same(model.classes_.tolist(), [0, 1], "class ordering")
    _same(model.n_features_in_, 769, "feature count")
    actual_params = model.get_params(deep=False)
    for key, expected in MODEL_PARAMS.items():
        _same(actual_params.get(key), expected, f"classifier parameter {key}")
    _same(metadata.get("classifier_params"), MODEL_PARAMS, "metadata parameters")
    _same(metadata.get("feature_signature"), FEATURE_SIGNATURE, "feature signature")
    _same(metadata.get("fit_manifest_sha256"), FIT_MANIFEST_SHA256, "manifest hash")
    if model.n_iter_ != MODEL_PARAMS["max_iter"]:
        raise EvaluationError("Classifier iteration count differs from the locked maximum")
    leaves = [tree.nodes["value"] for stage in model._predictors for tree in stage]
    if (not leaves or any(not np.isfinite(value).all() for value in leaves)
            or not np.isfinite(model._baseline_prediction).all()):
        raise EvaluationError("Classifier leaf values are missing or non-finite")
    return model, metadata


def _check_protocol(protocol: dict[str, Any], plan: dict[str, Any]) -> None:
    _same(plan.get("schema"), "dino-evaluation-plan/v4", "evaluation plan schema")
    _same(plan.get("protocol"), "artifacts/research-v4/dino-classifier-protocol.json", "protocol path")
    _same(_sha(PROTOCOL), plan.get("source_sha256", {}).get("artifacts/research-v4/dino-classifier-protocol.json"), "protocol source hash")
    _same(protocol.get("schema"), "dino-classifier-protocol/v4", "protocol schema")
    locked = protocol["locked_inputs_and_provenance"]
    _same(locked.get("training_manifest_sha256"), FIT_MANIFEST_SHA256, "protocol manifest hash")
    _same(protocol["classifier"]["initial_parameters"], MODEL_PARAMS, "protocol classifier parameters")
    for relative, expected in plan.get("source_sha256", {}).items():
        if not isinstance(expected, str) or _sha(ROOT / relative) != expected:
            raise EvaluationError(f"Evaluation-plan source hash mismatch: {relative}")


def _indices(pixels: np.ndarray) -> np.ndarray:
    return pixels // PATCH


def _centers(indices: np.ndarray, row: dict[str, Any]) -> list[list[int]]:
    return [[min(int(y) * PATCH + 7, row["height"] - 1),
             min(int(x) * PATCH + 7, row["width"] - 1)] for y, x in indices]


def _pairs(value: Any, name: str, row: dict[str, Any]) -> np.ndarray:
    try:
        result = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise EvaluationError(f"Invalid {name} coordinates for {row['id']}") from exc
    if result.size == 0:
        return np.empty((0, 2), dtype=np.int32)
    if result.ndim != 2 or result.shape[1:] != (2,) or result.dtype.kind not in "iu":
        raise EvaluationError(f"Invalid {name} coordinates for {row['id']}")
    if len(result) and (np.any(result < 0) or np.any(result[:, 0] >= row["height"]) or np.any(result[:, 1] >= row["width"])):
        raise EvaluationError(f"Out-of-range {name} coordinate for {row['id']}")
    return result.astype(np.int32, copy=False)


def _coverage_sets(coverage: dict[str, Any], fit_ids: set[str], audit: Any) -> tuple[dict[str, set[str]], dict[str, dict[str, Any]]]:
    if not isinstance(audit, list):
        raise EvaluationError("Sampling audit is missing")
    by_id = {str(record.get("query_id")): record for record in audit if isinstance(record, dict)}
    if len(by_id) != len(audit):
        raise EvaluationError("Sampling audit IDs are invalid or duplicated")
    categories = ("sampled_query_ids", "dual_class_query_ids", "positive_only_query_ids", "skipped_query_ids")
    partitions = {key: coverage.get(key, []) for key in categories}
    if any(not isinstance(value, list) for value in partitions.values()):
        raise EvaluationError("Sampling coverage IDs must be lists")
    sets = {key: set(value) for key, value in partitions.items()}
    if any(len(sets[key]) != len(partitions[key]) for key in categories):
        raise EvaluationError("Sampling coverage IDs contain duplicates")
    _check_coverage(sets, fit_ids, by_id)
    return sets, by_id


def _check_coverage(sets: dict[str, set[str]], fit_ids: set[str], by_id: dict[str, dict[str, Any]]) -> None:
    sampled, dual, positive_only, skipped = (sets[key] for key in (
        "sampled_query_ids", "dual_class_query_ids", "positive_only_query_ids", "skipped_query_ids"))
    if dual & positive_only or skipped & sampled:
        raise EvaluationError("Sampling coverage categories overlap incorrectly")
    if sampled != dual | positive_only or sampled | skipped != fit_ids:
        raise EvaluationError("Sampling coverage categories do not partition fit IDs")
    if set(by_id) != sampled:
        raise EvaluationError("Sampling audit does not cover sampled query IDs")


def _verify_record(row: dict[str, Any], record: dict[str, Any], seed: int, root: Path) -> dict[str, int]:
    ident = str(row["id"])
    if record.get("document_id") != str(row["document_id"]) or record.get("seed") != seed:
        raise EvaluationError(f"Sampling audit provenance mismatch: {ident}")
    positive, known, _, _ = reviewed_domains(row, root)
    pos, neg = sample_pure_patch_pixels(positive, known, seed, SAMPLES_PER_CLASS)
    cells: dict[str, np.ndarray] = {}
    for cls, expected, domain in (("positive", pos, positive), ("negative", neg, known & ~positive)):
        field = cls + "_native_pixels"
        actual = _pairs(record.get(field), field, row)
        if not np.array_equal(actual, expected):
            raise EvaluationError(f"Recomputed {field} differs for {ident}")
        if len(actual) and not bool(domain[actual[:, 0], actual[:, 1]].all()):
            raise EvaluationError(f"Sampled {field} includes unknown or wrong-class pixels: {ident}")
        cells[cls] = _indices(actual)
        if len({tuple(point) for point in cells[cls].tolist()}) != len(cells[cls]):
            raise EvaluationError(f"Sampling repeats a patch token: {ident}")
        _check_pure_cells(cells[cls], cls, positive, known, ident)
        if not np.array_equal(_pairs(record.get(cls + "_patch_indices"), cls + " patch indices", row), cells[cls]):
            raise EvaluationError(f"Patch indices mismatch: {ident}")
        if record.get("class_counts", {}).get(cls) != len(cells[cls]):
            raise EvaluationError(f"Class sample count mismatch: {ident}")
        _same(record.get(cls + "_patch_centers_yx"), _centers(cells[cls], row), f"{cls} centers for {ident}")
    expected_weights = {"positive": (0.5 if len(neg) else 1.0) / len(pos),
                        "negative": 0.5 / len(neg) if len(neg) else 0.0}
    _same(record.get("class_weights"), expected_weights, f"sample weights for {ident}")
    return {cls: len(value) for cls, value in cells.items()}


def _check_pure_cells(cells: np.ndarray, cls: str, positive: np.ndarray, known: np.ndarray, ident: str) -> None:
    negative = known & ~positive
    for patch_y, patch_x in cells:
        y0, x0 = int(patch_y) * PATCH, int(patch_x) * PATCH
        pos_patch = positive[y0:y0 + PATCH, x0:x0 + PATCH]
        neg_patch = negative[y0:y0 + PATCH, x0:x0 + PATCH]
        mixed = bool(pos_patch.any() and neg_patch.any())
        if mixed or (not bool(pos_patch.any()) if cls == "positive" else not bool(neg_patch.any())):
            raise EvaluationError(f"Sampled token is not class-pure: {ident}")


def _verify_audit(rows: list[dict[str, Any]], root: Path, metadata: dict[str, Any]) -> dict[str, int]:
    ordered = sorted(rows, key=lambda r: (str(r["document_id"]), str(r["id"])))
    expected_seeds = {str(row["id"]): 45 + index for index, row in enumerate(ordered)}
    fit_ids = set(expected_seeds)
    coverage = metadata.get("sampled_training_coverage")
    if not isinstance(coverage, dict):
        raise EvaluationError("Sampled training coverage is missing")
    sets, by_id = _coverage_sets(coverage, fit_ids, metadata.get("sampling_audit"))
    rows_by_id = {str(row["id"]): row for row in rows}
    for ident, record in by_id.items():
        _verify_record(rows_by_id[ident], record, expected_seeds[ident], root)
    if sets["dual_class_query_ids"] != {i for i, r in by_id.items() if r["class_counts"]["negative"] > 0}:
        raise EvaluationError("Dual-class query IDs disagree with audited class counts")
    if sets["positive_only_query_ids"] != {i for i, r in by_id.items() if r["class_counts"]["negative"] == 0}:
        raise EvaluationError("Positive-only query IDs disagree with audited class counts")
    class_totals = {"positive": sum(r["class_counts"]["positive"] for r in by_id.values()),
                    "negative": sum(r["class_counts"]["negative"] for r in by_id.values())}
    _same(coverage.get("positive_samples"), class_totals["positive"], "positive sample total")
    _same(coverage.get("negative_samples"), class_totals["negative"], "negative sample total")
    _same(coverage.get("samples"), sum(class_totals.values()), "sample total")
    _same(coverage.get("queries"), len(sets["sampled_query_ids"]), "sampled query count")
    _same(coverage.get("documents"), len({str(rows_by_id[i]["document_id"]) for i in sets["sampled_query_ids"]}), "sampled document count")
    return class_totals


def sample_pure_patch_pixels(positive: np.ndarray, known: np.ndarray, seed: int, cap: int) -> tuple[np.ndarray, np.ndarray]:
    from dino_matching import sample_pure_patch_pixels as sample
    return sample(positive, known, seed, cap)


def _runtime(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise EvaluationError(f"Cannot read runtime log: {path}") from exc
    # Require a complete, single /usr/bin/time -l block. This metric is process-level;
    # a host-wide or historical swap counter cannot be attributed to the fit process.
    blocks = re.findall(r"(?ms)^\s*([\d.]+)\s+real\s+([\d.]+)\s+user\s+([\d.]+)\s+sys\s*\n(.*?)(?=^\s*[\d.]+\s+real\b|\Z)", text)
    if len(blocks) != 1:
        raise EvaluationError("Runtime log must contain exactly one process measurement block")
    wall, _, _, body = blocks[0]
    swap = re.findall(r"^\s*(\d+)\s+swaps\s*$", body, re.MULTILINE)
    rss = re.findall(r"^\s*(\d+)\s+maximum resident set size\s*$", body, re.MULTILINE)
    footprint = re.findall(r"^\s*(\d+)\s+peak memory footprint\s*$", body, re.MULTILINE)
    if len(swap) != 1 or len(rss) != 1 or len(footprint) != 1:
        raise EvaluationError("Runtime log lacks complete unambiguous process resource measurements")
    if not np.isfinite(float(wall)) or float(wall) <= 0:
        raise EvaluationError("Runtime wall time must be finite and positive")
    if int(swap[0]) != 0 or not 0 < int(footprint[0]) < 16 * 1024**3:
        raise EvaluationError("DINO fit exceeds the zero-process-swap or under-16-GiB resource gate")
    return {"measurement": "/usr/bin/time -l", "process_attributed": True,
            "external_wall_seconds": float(wall), "process_swaps": int(swap[0]),
            "maximum_resident_set_bytes": int(rss[0]),
            "peak_memory_footprint_bytes": int(footprint[0]),
            "global_swap_claim": None}


def verify_fit(model_path: Path, manifest: Path, data_root: Path, evaluation_plan: Path,
               output: Path, runtime_log: Path) -> dict[str, Any]:
    manifest_sha = verify_fit_manifest(manifest)
    rows = load_manifest(manifest, require_labels=True)
    check_split_safety(rows, data_root)
    protocol = _json(PROTOCOL)
    plan = _json(evaluation_plan)
    _check_protocol(protocol, plan)
    support = verify_support_evidence(SUPPORT, manifest_sha)
    asset_mapping = verify_fit_assets(ASSET_PROOF, manifest_sha, rows, data_root)
    model, metadata = _verify_model(model_path)
    _same(metadata.get("training_ids"), sorted(str(r["id"]) for r in rows), "training IDs")
    _same(metadata.get("training_documents"), sorted({str(r["document_id"]) for r in rows}), "training documents")
    _same(metadata.get("classifier_protocol_sha256"), _sha(PROTOCOL), "protocol metadata hash")
    for relative, digest in metadata.get("implementation_sha256", {}).items():
        _same(_sha(ROOT / relative), digest, f"implementation hash for {relative}")
    _same(metadata.get("support_evidence_sha256"), _sha(SUPPORT), "support evidence hash")
    _same(metadata.get("fit_asset_mapping_sha256"), asset_mapping, "asset mapping hash")
    _same(metadata.get("fit_asset_proof_sha256"), _sha(ASSET_PROOF), "fit asset proof hash")
    locked = protocol["locked_inputs_and_provenance"]
    _same(metadata.get("source_commit"), locked.get("dino_source_commit"), "DINO source commit")
    _same(metadata.get("checkpoint_sha256"), locked.get("dino_checkpoint_sha256"), "DINO checkpoint hash")
    if metadata.get("dino_eval_mode") is not True or metadata.get("dino_all_parameters_frozen") is not True:
        raise EvaluationError("DINO extractor was not proven frozen and in eval mode")
    if int(metadata.get("support_evidence_dual_queries", 0)) < 30 or int(metadata.get("support_evidence_dual_documents", 0)) < 10:
        raise EvaluationError("Recorded support proof does not meet the locked minimum")
    counts = _verify_audit(rows, data_root, metadata)
    measured = _runtime(runtime_log)
    proof = {"schema": "dino-fit-proof/v1", "manifest_sha256": manifest_sha,
             "model_sha256": _sha(model_path), "model_signature": metadata["model_signature"],
             "protocol_sha256": _sha(PROTOCOL), "evaluation_plan_sha256": _sha(evaluation_plan),
             "fit_queries": len(rows), "fit_documents": len(metadata["training_documents"]),
             "sampled_class_counts": counts, "support_gate_passed": support["support_gate"]["passed"],
             "fit_asset_mapping_sha256": asset_mapping, "estimator": type(model).__name__,
             "features": model.n_features_in_, "iterations": model.n_iter_, "runtime": measured}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(proof, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return proof


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "manifest", "data-root", "evaluation-plan", "output", "runtime-log"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        proof = verify_fit(args.model, args.manifest, args.data_root, args.evaluation_plan, args.output, args.runtime_log)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps(proof, sort_keys=True))


if __name__ == "__main__":
    main()
