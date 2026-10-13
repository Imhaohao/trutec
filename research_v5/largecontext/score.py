"""Score a checkpoint on the 55 development queries and fuse it with the HGB cache."""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_v5.devharness import HGB_CACHE, development_queries, load_rgb, report  # noqa: E402

from .geom import pack_half_rgb, query_pack
from .infer import load_model, lookup_sigmoid, pick_device, stitch_many


def main() -> None:
    args = parse_args()
    torch.set_num_threads(4)
    device = pick_device(args.device)
    model = load_model(args.checkpoint, device)
    scores = score_development(model, device)
    _write_reports(args, scores)
    print(f"wrote {args.output}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    return parser.parse_args()


def score_development(model, device) -> dict[str, np.ndarray]:
    grouped: dict[str, list] = {}
    for query in development_queries():
        grouped.setdefault(query.image, []).append(query)
    scores: dict[str, np.ndarray] = {}
    for image, queries in grouped.items():
        _score_sheet(model, device, image, queries, scores)
        print(f"scored {image} queries {len(queries)} total {len(scores)}", flush=True)
        if device.type == "mps":
            torch.mps.empty_cache()
    return scores


def _score_sheet(model, device, image: str, queries, scores: dict[str, np.ndarray]) -> None:
    rgb = load_rgb(image)
    packed = pack_half_rgb(rgb)
    bank = {query.id: query_pack(rgb, query.query_box) for query in queries}
    logits = stitch_many(model, packed, bank, device)
    for query in queries:
        scores[query.id] = lookup_sigmoid(logits[query.id], query.y, query.x)


def _write_reports(args, scores: dict[str, np.ndarray]) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    own = report("largecontext", scores, args.output)
    hgb = _hgb_scores()
    fused_min = {key: np.minimum(scores[key], hgb[key]) for key in scores}
    fused_geo = {key: np.sqrt(np.clip(scores[key], 0, 1) * np.clip(hgb[key], 0, 1)).astype(np.float32) for key in scores}
    min_report = report("largecontext_min_hgb", fused_min, args.output.with_name(args.output.stem + "_min_hgb.json"))
    geo_report = report("largecontext_geomean_hgb", fused_geo, args.output.with_name(args.output.stem + "_geomean_hgb.json"))
    point = own["selections"]["best_precision_at_recall_ge_0_98"]
    summary = {
        "largecontext": own["selections"],
        "min_hgb": min_report["selections"],
        "geomean_hgb": geo_report["selections"],
        "per_query_recall_at_precision_point": _per_query(scores, point),
    }
    args.output.with_name(args.output.stem + "_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _hgb_scores() -> dict[str, np.ndarray]:
    return {query.id: np.load(HGB_CACHE / f"{query.id}.npz")["probability"] for query in development_queries()}


def _per_query(scores: dict[str, np.ndarray], point: dict | None) -> list[dict]:
    if point is None:
        return []
    threshold = float(point["threshold"])
    rows = []
    for query in development_queries():
        rows.append(_one_recall(query, scores[query.id] >= threshold, threshold))
    return rows


def _one_recall(query, decision: np.ndarray, threshold: float) -> dict:
    positive = query.positive.astype(bool)
    tp = int(np.count_nonzero(decision & positive))
    fn = int(np.count_nonzero(~decision & positive))
    recall = None if tp + fn == 0 else tp / (tp + fn)
    return {"id": query.id, "document_id": query.document_id, "threshold": threshold, "recall": recall, "tp": tp, "fn": fn}


if __name__ == "__main__":
    main()
