"""Shared development-split scoring harness for overnight research.

Reviewed-known coordinates and labels come from the existing exact sparse
development cache (row-major known pixels after query/context exclusion), so
every experiment is scored on the identical evaluator domain without reopening
annotation masks. Scores are compared with the unchanged evaluator's
metrics_from_counts/aggregate functions.

Inference code must never read labels; only the scorer here does.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluate import aggregate, metrics_from_counts  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
DATA_ROOT = ROOT / "dataset"
DEVELOPMENT_MANIFEST = ROOT / "artifacts/splits/development-real.json"
FIT_MANIFEST = ROOT / "artifacts/splits/fit.json"
DOMAIN_CACHE = ROOT / "artifacts/hatchfinder-adapt-development-v1/probabilities"
HGB_CACHE = ROOT / "artifacts/texture-crop-development-v1/probabilities"
THRESHOLDS = np.linspace(0.0, 1.0, 101)
GOALS = {"document_macro_iou": 0.95, "global_recall": 0.98, "global_precision": 0.95}


@dataclass(frozen=True)
class DevQuery:
    id: str
    document_id: str
    image: str
    width: int
    height: int
    query_box: tuple[int, int, int, int]
    context_boxes: tuple[tuple[int, int, int, int], ...]
    y: np.ndarray
    x: np.ndarray
    positive: np.ndarray


def _rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["examples"] if isinstance(payload, dict) else payload


def request_view(row: dict) -> dict:
    """Label-free request fields; the only fields inference code may consume."""
    keys = ("id", "image", "width", "height", "query_box", "context_boxes")
    return {key: row[key] for key in keys if key in row}


def fit_rows(real_only: bool = False) -> list[dict]:
    rows = _rows(FIT_MANIFEST)
    return [row for row in rows if row.get("kind") == "real"] if real_only else rows


@lru_cache(maxsize=1)
def development_queries() -> tuple[DevQuery, ...]:
    queries = []
    for row in _rows(DEVELOPMENT_MANIFEST):
        cache = np.load(DOMAIN_CACHE / f"{row['id']}.npz")
        queries.append(
            DevQuery(
                id=row["id"],
                document_id=row["document_id"],
                image=row["image"],
                width=int(row["width"]),
                height=int(row["height"]),
                query_box=tuple(row["query_box"]),
                context_boxes=tuple(tuple(box) for box in row.get("context_boxes", [])),
                y=cache["y"],
                x=cache["x"],
                positive=cache["positive"],
            )
        )
    return tuple(queries)


def load_rgb(relative_image: str) -> np.ndarray:
    with Image.open(DATA_ROOT / relative_image) as image:
        return np.asarray(image.convert("RGB"))


def load_gray(relative_image: str) -> np.ndarray:
    with Image.open(DATA_ROOT / relative_image) as image:
        return np.asarray(image.convert("L"))


def cached_scores(cache_dir: Path) -> dict[str, np.ndarray]:
    return {q.id: np.load(cache_dir / f"{q.id}.npz")["probability"] for q in development_queries()}


def summarize_decisions(decisions: dict[str, np.ndarray]) -> dict:
    """Official aggregation for boolean decisions at each reviewed-known pixel."""
    rows = []
    for query in development_queries():
        decision = decisions[query.id].astype(bool)
        if decision.shape != query.positive.shape:
            raise ValueError(f"{query.id}: decision shape {decision.shape} != {query.positive.shape}")
        tp = int(np.count_nonzero(decision & query.positive))
        fp = int(np.count_nonzero(decision & ~query.positive))
        fn = int(np.count_nonzero(~decision & query.positive))
        tn = int(np.count_nonzero(~decision & ~query.positive))
        rows.append({"document_id": query.document_id, "metrics": metrics_from_counts(tp, fp, fn, tn)})
    summary = aggregate(rows)
    negatives = summary["fp"] + summary["tn"]
    summary["reviewed_negative_fpr"] = summary["fp"] / negatives if negatives else None
    summary["false_discovery_rate"] = (
        None if summary["global_precision"] is None else 1.0 - summary["global_precision"]
    )
    return summary


def sweep_scores(scores: dict[str, np.ndarray], thresholds: Iterable[float] = THRESHOLDS) -> list[dict]:
    points = []
    for threshold in thresholds:
        summary = summarize_decisions({key: value >= threshold for key, value in scores.items()})
        points.append({"threshold": float(threshold), **_compact(summary)})
    return points


def _compact(summary: dict) -> dict:
    keys = (
        "document_macro_iou", "global_recall", "global_precision", "reviewed_negative_fpr",
        "false_discovery_rate", "tp", "fp", "fn", "tn",
    )
    return {key: summary[key] for key in keys}


def selections(points: list[dict]) -> dict:
    def best(candidates: list[dict], key: str) -> dict | None:
        return max(candidates, key=lambda p: p[key] or 0.0) if candidates else None

    recall_ok = [p for p in points if (p["global_recall"] or 0.0) >= GOALS["global_recall"]]
    return {
        "best_macro_iou": best(points, "document_macro_iou"),
        "best_precision_at_recall_ge_0_98": best(recall_ok, "global_precision"),
        "best_macro_iou_at_recall_ge_0_98": best(recall_ok, "document_macro_iou"),
        "all_goals_met": any(
            p["document_macro_iou"] >= GOALS["document_macro_iou"]
            and (p["global_recall"] or 0) >= GOALS["global_recall"]
            and (p["global_precision"] or 0) >= GOALS["global_precision"]
            and (p["reviewed_negative_fpr"] or 0) <= 0.05
            for p in points
        ),
    }


def report(name: str, scores: dict[str, np.ndarray], output: Path | None = None) -> dict:
    points = sweep_scores(scores)
    result = {"name": name, "selections": selections(points), "points": points}
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_predictor(
    predict: Callable[[dict, np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    name: str,
    output: Path | None = None,
) -> dict:
    """predict(request, rgb, y, x) -> float scores in [0,1] at the given coordinates.

    The request carries no labels. y/x are query-independent scoring locations;
    a valid predictor must give the same score at a pixel regardless of whether
    it is queried, i.e. it must equal a full-image prediction sampled there.
    """
    rows = {row["id"]: row for row in _rows(DEVELOPMENT_MANIFEST)}
    scores = {}
    for query in development_queries():
        rgb = load_rgb(query.image)
        scores[query.id] = np.asarray(
            predict(request_view(rows[query.id]), rgb, query.y, query.x), dtype=np.float32
        )
    return report(name, scores, output)


if __name__ == "__main__":
    baseline = report("original_crop_hgb", cached_scores(HGB_CACHE))
    print(json.dumps(baseline["selections"], indent=2))
