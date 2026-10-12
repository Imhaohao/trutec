"""Count reviewed fit-label ambiguity at the proposed DINO patch stride."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np

PATCH_SIZE = 14
FIT_MANIFEST_SHA256 = "62f99269f09c510870ecb8442479a33d7d04e49ca8b1bbd4274049162b9869d0"
MIN_DUAL_QUERIES = 30
MIN_DUAL_DOCUMENTS = 10
ADDITIVE_COUNT_KEYS = (
    "positive_pixels", "negative_pixels", "known_pixels", "pure_positive_patch_cells",
    "pure_negative_patch_cells", "mixed_label_patch_cells", "empty_patch_cells",
    "pure_positive_pixels", "pure_negative_pixels", "mixed_positive_pixels", "mixed_negative_pixels",
)


def verify_fit_manifest(path: Path) -> str:
    """Require the exact fit manifest before reading reviewed annotations."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != FIT_MANIFEST_SHA256:
        raise ValueError(f"Fit manifest SHA-256 mismatch: expected {FIT_MANIFEST_SHA256}, got {digest}")
    return digest


def count_patch_support(positive: np.ndarray, known: np.ndarray) -> dict[str, int]:
    """Count pure, mixed, and empty patch cells using labels only."""
    positive = np.asarray(positive, dtype=bool)
    known = np.asarray(known, dtype=bool)
    if positive.ndim != 2 or known.shape != positive.shape:
        raise ValueError("positive and known masks must be matching 2D arrays")
    if bool((positive & ~known).any()):
        raise ValueError("positive pixels must be a subset of known pixels")

    height, width = positive.shape
    grid_h = (height + PATCH_SIZE - 1) // PATCH_SIZE
    grid_w = (width + PATCH_SIZE - 1) // PATCH_SIZE
    padded_shape = (grid_h * PATCH_SIZE, grid_w * PATCH_SIZE)
    positive_pad = np.zeros(padded_shape, dtype=bool)
    negative_pad = np.zeros(padded_shape, dtype=bool)
    positive_pad[:height, :width] = positive
    negative_pad[:height, :width] = known & ~positive
    positive_counts = positive_pad.reshape(grid_h, PATCH_SIZE, grid_w, PATCH_SIZE).sum(axis=(1, 3))
    negative_counts = negative_pad.reshape(grid_h, PATCH_SIZE, grid_w, PATCH_SIZE).sum(axis=(1, 3))

    pure_positive = (positive_counts > 0) & (negative_counts == 0)
    pure_negative = (negative_counts > 0) & (positive_counts == 0)
    mixed = (positive_counts > 0) & (negative_counts > 0)
    occupied = (positive_counts + negative_counts) > 0
    return {
        "positive_pixels": int(positive_counts.sum()),
        "negative_pixels": int(negative_counts.sum()),
        "known_pixels": int(positive_counts.sum() + negative_counts.sum()),
        "patch_rows": grid_h,
        "patch_columns": grid_w,
        "pure_positive_patch_cells": int(pure_positive.sum()),
        "pure_negative_patch_cells": int(pure_negative.sum()),
        "mixed_label_patch_cells": int(mixed.sum()),
        "empty_patch_cells": int((~occupied).sum()),
        "pure_positive_pixels": int(positive_counts[pure_positive].sum()),
        "pure_negative_pixels": int(negative_counts[pure_negative].sum()),
        "mixed_positive_pixels": int(positive_counts[mixed].sum()),
        "mixed_negative_pixels": int(negative_counts[mixed].sum()),
    }


def analyze_fit_rows(rows: list[dict[str, Any]], data_root: Path) -> dict[str, Any]:
    """Count patches after the evaluator excludes query and context boxes."""
    from evaluate import reviewed_domains

    per_query = []
    for row in rows:
        positive, known, _, metadata = reviewed_domains(row, data_root)
        per_query.append({
            "query_id": str(row["id"]),
            "document_id": str(row["document_id"]),
            "positive_only": bool(metadata["positive_only"]),
            "original_known_pixels": int(metadata["original_known_pixels"]),
            "excluded_known_pixels": int(metadata["excluded_known_pixels"]),
            **count_patch_support(positive, known),
        })

    totals = {key: sum(cast(int, query[key]) for query in per_query) for key in ADDITIVE_COUNT_KEYS}
    dual = [q for q in per_query if q["pure_positive_patch_cells"] and q["pure_negative_patch_cells"]]
    positive_eligible = [q for q in per_query if q["pure_positive_patch_cells"]]
    dual_documents = {q["document_id"] for q in dual}
    return {
        "schema": "dino-patch-support/v1",
        "partition": "pinned fit manifest only",
        "patch_size": PATCH_SIZE,
        "labels_only_no_image_or_model_features": True,
        "domain_source": "evaluate.reviewed_domains excludes query_box and context_boxes before counting",
        "queries": len(per_query),
        "documents": len({q["document_id"] for q in per_query}),
        "eligible_positive_queries": len(positive_eligible),
        "dual_pure_queries": len(dual),
        "positive_only_queries": sum(bool(q["positive_only"]) for q in per_query),
        "dual_pure_documents": len(dual_documents),
        "support_gate": {
            "minimum_dual_queries": MIN_DUAL_QUERIES,
            "minimum_dual_documents": MIN_DUAL_DOCUMENTS,
            "passed": len(dual) >= MIN_DUAL_QUERIES and len(dual_documents) >= MIN_DUAL_DOCUMENTS,
        },
        "totals": totals,
        "per_query": per_query,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/splits/fit.json"))
    parser.add_argument("--data-root", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, help="Optional JSON destination; defaults to stdout")
    args = parser.parse_args()
    try:
        manifest_sha256 = verify_fit_manifest(args.manifest)
        from evaluate import load_manifest
        from train_hatchfinder import check_split_safety

        rows = load_manifest(args.manifest, require_labels=True)
        check_split_safety(rows, args.data_root)
        report = analyze_fit_rows(rows, args.data_root)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    report["manifest_path"] = str(args.manifest.resolve())
    report["manifest_sha256"] = manifest_sha256
    serialized = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(serialized, encoding="utf-8")
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
