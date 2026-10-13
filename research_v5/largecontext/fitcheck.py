"""Score ten real fit queries with the development inference path.

Also reports whether an unaugmented training crop separates positive cells.
Labels are read only here. infer.py still sees the image and the query box.
"""

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

from evaluate import aggregate, load_manifest, metrics_from_counts, reviewed_domains  # noqa: E402
from research_v5.devharness import THRESHOLDS, load_rgb, selections  # noqa: E402

from .constants import KNOWN_FRACTION, NATIVE_CROP
from .geom import area_half_rgb, cell_fractions, pack_half_rgb, pack_hwc, query_pack, square_window, tile_repeat
from .infer import forward_logits, load_model, lookup_sigmoid, pick_device, stitch_many

FIT_MANIFEST = ROOT / "artifacts/splits/fit.json"
DATA_ROOT = (ROOT / "dataset" / "assets").resolve().parent
KEYS = (
    "document_macro_iou",
    "global_recall",
    "global_precision",
    "tp",
    "fp",
    "fn",
    "tn",
)


def main() -> None:
    args = parse_args()
    torch.set_num_threads(4)
    device = pick_device(args.device)
    model = load_model(args.checkpoint, device)
    rows = choose_rows(args.count, args.seed)
    crops = [_crop_report(model, row, device) for row in rows]
    queries = score_rows(model, rows, device)
    payload = {"crops": crops, "queries": _public_queries(queries), "summary": _summary(queries)}
    _write(args.output, payload)
    _print(payload)
    print(f"wrote {args.output}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    return parser.parse_args()


def choose_rows(count: int, seed: int) -> list[dict]:
    """One real query per document, skipping positive-only queries.

    A positive-only query has no reviewed negatives, so predicting every
    pixel scores IoU 1 and cannot show a coordinate bug.
    """
    grouped = _by_document(_real_rows())
    docs = sorted(grouped)
    rng = np.random.default_rng(seed)
    rng.shuffle(docs)
    picked = []
    for doc in docs:
        row = _first_scored(grouped[doc])
        if row is None:
            continue
        picked.append(row)
        if len(picked) == count:
            return picked
    raise RuntimeError("not enough fit queries with reviewed negatives")


def score_rows(model, rows: list[dict], device) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["image"], []).append(row)
    scored = []
    for image, group in grouped.items():
        scored.extend(_score_sheet(model, image, group, device))
        print(f"scored {image} queries {len(group)}", flush=True)
        if device.type == "mps":
            torch.mps.empty_cache()
    return scored


def _crop_report(model, row: dict, device) -> dict:
    rgb = load_rgb(row["image"])
    positive, known, _, _ = reviewed_domains(row, DATA_ROOT)
    center = _median_point(positive)
    window, known_c, pos_c = square_window(rgb, known, positive, int(center[0]), int(center[1]), NATIVE_CROP)
    known_frac, pos_frac = cell_fractions(known_c, pos_c)
    drawing = pack_hwc(area_half_rgb(window))
    query = _query_from_box(rgb, row["query_box"])
    logits = forward_logits(model, drawing, query, device)
    return _crop_stats(row, logits, known_frac, pos_frac, drawing)


def _score_sheet(model, image: str, rows: list[dict], device) -> list[dict]:
    rgb = load_rgb(image)
    packed = pack_half_rgb(rgb)
    bank = {row["id"]: query_pack(rgb, tuple(int(v) for v in row["query_box"])) for row in rows}
    logits = stitch_many(model, packed, bank, device)
    return [_query_stats(row, logits[row["id"]]) for row in rows]


def _query_stats(row: dict, logits: np.ndarray) -> dict:
    positive, known, _, _ = reviewed_domains(row, DATA_ROOT)
    y, x = np.nonzero(known)
    scores = lookup_sigmoid(logits, y, x)
    labels = positive[y, x]
    return {
        "id": row["id"],
        "document_id": row["document_id"],
        "known": int(y.shape[0]),
        "positive_rate": float(labels.mean()) if labels.size else None,
        **_means(scores, labels),
        "quads": [_counts(scores >= threshold, labels) for threshold in THRESHOLDS],
    }


def _crop_stats(row, logits, known_frac, pos_frac, drawing) -> dict:
    prob = _sigmoid(logits)
    known = known_frac >= KNOWN_FRACTION
    pos = known & (pos_frac >= 0.5)
    neg = known & (pos_frac < 0.5)
    return {
        "id": row["id"],
        "known_cells": int(known.sum()),
        "center_pos_fraction": _center_value(pos_frac),
        "mean_pos": _mean(prob[pos]),
        "mean_neg": _mean(prob[neg]),
        "score_std": _mean_std(prob[known]),
        "target_std": _mean_std(pos_frac[known]),
        "ink_target_corr": _corr(_cell_ink(drawing)[known], pos_frac[known]),
        "score_target_corr": _corr(prob[known], pos_frac[known]),
    }


def _summary(queries: list[dict]) -> dict:
    points = [_threshold_point(queries, index, threshold) for index, threshold in enumerate(THRESHOLDS)]
    return {"selections": selections(points), "per_query_best_iou": [_best_iou(query) for query in queries]}


def _threshold_point(queries: list[dict], index: int, threshold: float) -> dict:
    rows = [_metric_row(query, index) for query in queries]
    summary = aggregate(rows)
    point = {key: summary[key] for key in KEYS}
    point["threshold"] = float(threshold)
    negatives = point["fp"] + point["tn"]
    point["reviewed_negative_fpr"] = point["fp"] / negatives if negatives else None
    point["false_discovery_rate"] = None if point["global_precision"] is None else 1.0 - point["global_precision"]
    return point


def _metric_row(query: dict, index: int) -> dict:
    tp, fp, fn, tn = query["quads"][index]
    return {"document_id": query["document_id"], "metrics": metrics_from_counts(tp, fp, fn, tn)}


def _best_iou(query: dict) -> dict:
    best = None
    threshold = None
    for index, value in enumerate(THRESHOLDS):
        iou = _metric_row(query, index)["metrics"]["iou"]
        if iou is None or (best is not None and iou <= best):
            continue
        best = iou
        threshold = float(value)
    return {"id": query["id"], "best_iou": best, "threshold": threshold, **_means_view(query)}


def _public_queries(queries: list[dict]) -> list[dict]:
    return [{key: value for key, value in query.items() if key != "quads"} for query in queries]


def _real_rows() -> list[dict]:
    return [row for row in load_manifest(FIT_MANIFEST) if row.get("kind") == "real"]


def _by_document(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["document_id"], []).append(row)
    return grouped


def _first_scored(rows: list[dict]) -> dict | None:
    for row in sorted(rows, key=lambda item: item["id"]):
        _positive, _known, _blank, meta = reviewed_domains(row, DATA_ROOT)
        if not meta["positive_only"]:
            return row
    return None


def _query_from_box(rgb: np.ndarray, box) -> np.ndarray:
    x0, y0, x1, y1 = (int(value) for value in box)
    return pack_hwc(tile_repeat(area_half_rgb(rgb[y0:y1, x0:x1])))


def _median_point(mask: np.ndarray) -> np.ndarray:
    points = np.argwhere(mask)
    if len(points) == 0:
        raise RuntimeError("fit query has no positive reviewed pixels")
    return points[len(points) // 2]


def _counts(decision: np.ndarray, positive: np.ndarray) -> tuple[int, int, int, int]:
    yes = decision.astype(bool)
    pos = positive.astype(bool)
    tp = int(np.count_nonzero(yes & pos))
    fp = int(np.count_nonzero(yes & ~pos))
    fn = int(np.count_nonzero(~yes & pos))
    tn = int(np.count_nonzero(~yes & ~pos))
    return tp, fp, fn, tn


def _means(scores: np.ndarray, positive: np.ndarray) -> dict:
    pos = positive.astype(bool)
    return {"mean_pos": _mean(scores[pos]), "mean_neg": _mean(scores[~pos])}


def _means_view(query: dict) -> dict:
    return {"mean_pos": query["mean_pos"], "mean_neg": query["mean_neg"], "positive_rate": query["positive_rate"]}


def _cell_ink(drawing: np.ndarray) -> np.ndarray:
    ink = drawing[3]
    cells_y, cells_x = ink.shape[0] // 2, ink.shape[1] // 2
    blocks = ink[: cells_y * 2, : cells_x * 2].reshape(cells_y, 2, cells_x, 2)
    return blocks.mean(axis=(1, 3))


def _center_value(values: np.ndarray) -> float:
    y = values.shape[0] // 2
    x = values.shape[1] // 2
    return float(values[y, x])


def _corr(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2 or float(left.std()) == 0.0 or float(right.std()) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _mean(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return float(values.mean())


def _mean_std(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return float(values.std())


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values.astype(np.float32), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _print(payload: dict) -> None:
    chosen = payload["summary"]["selections"]
    print("fit_selections " + json.dumps(chosen), flush=True)
    for row in payload["crops"]:
        print(
            f"crop {row['id']} corr {row['score_target_corr']} ink {row['ink_target_corr']} "
            f"pos {row['mean_pos']} neg {row['mean_neg']} center {row['center_pos_fraction']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
