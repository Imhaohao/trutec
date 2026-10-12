"""Measure the frozen cadence support gate using fit crops and reviewed fit domains."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from evaluate import load_manifest, reviewed_domains
from hatch_cadence import estimate_query_lag
from inference import read_drawing
from tools.dino_patch_support import FIT_MANIFEST_SHA256
from train_texture import validate_split_safety


def support_gate(records: list[dict[str, Any]]) -> dict[str, Any]:
    stable = [record for record in records if record["stable"]]
    dual = [record for record in stable if record["dual_class"]]
    documents = sorted({record["document_id"] for record in dual})
    return {
        "stable_queries": len(stable), "stable_dual_queries": len(dual),
        "stable_dual_documents": len(documents), "document_ids": documents,
        "minimum_dual_queries": 30, "minimum_dual_documents": 10,
        "passed": len(dual) >= 30 and len(documents) >= 10,
    }


def crop_record(gray: np.ndarray, row: dict[str, Any], data_root: Path) -> dict[str, Any]:
    if gray.shape != (row["height"], row["width"]):
        raise ValueError(f"native shape differs for {row['id']}")
    x0, y0, x1, y1 = row["query_box"]
    crop = gray[y0:y1, x0:x1]
    if crop.shape != (y1 - y0, x1 - x0):
        raise ValueError(f"invalid query crop for {row['id']}")
    lag = estimate_query_lag(crop)
    positive, known, _, _ = reviewed_domains(row, data_root)
    return {
        "id": row["id"], "document_id": row["document_id"],
        "crop_shape": list(crop.shape),
        "crop_pixel_sha256": hashlib.sha256(crop.tobytes()).hexdigest(),
        "stable": lag.available, "lag": asdict(lag),
        "dual_class": bool(positive.any() and (known & ~positive).any()),
        "positive_reviewed_pixels": int(positive.sum()),
        "negative_reviewed_pixels": int((known & ~positive).sum()),
    }


def collect_support(rows: list[dict[str, Any]], data_root: Path) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["image"]), []).append(row)
    records = []
    for group in groups.values():
        gray = read_drawing(data_root, group[0])
        for row in group:
            records.append(crop_record(gray, row, data_root))
        print(json.dumps({"drawing": group[0]["image"], "processed_queries": len(records)}), flush=True)
    return records


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path("artifacts/research-v4/cadence-feature-protocol.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest_sha = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    if manifest_sha != FIT_MANIFEST_SHA256:
        raise ValueError("cadence support requires the pinned fit manifest")
    rows = load_manifest(args.manifest, require_labels=True)
    validate_split_safety(args.manifest, args.data_root, rows)
    records = collect_support(rows, args.data_root)
    source_paths = (Path("hatch_cadence.py"), Path("tools/cadence_fit_support.py"))
    report = {
        "schema": "cadence-fit-support/v1", "manifest_sha256": manifest_sha,
        "protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        "support_gate": support_gate(records), "queries": len(rows), "per_query": records,
        "label_use": "fit reviewed domains determine class availability only; query pixels determine lag",
        "quality": "unmeasured; no model fit or prediction",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["support_gate"]), flush=True)


if __name__ == "__main__":
    main()
