"""Oracle upper bounds for label-free region partitions.

A region is called positive when strictly more than half of its reviewed pixels
are positive. Ties stay negative. Development labels are used only in this vote
and in summarize_decisions. The partition itself is label-free.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research_v5"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from devharness import development_queries, load_rgb, summarize_decisions  # noqa: E402
from partitions import (  # noqa: E402
    area_stats,
    boundary_enclosures,
    filled_closing,
    paper_after_thin_removal,
    texture_superpixels,
)

OUT = Path(__file__).resolve().parent / "oracle_summary.json"


@dataclass(frozen=True)
class Variant:
    name: str
    family: str
    open_radius: int = 0
    close_radius: int = 0
    cell: int = 0
    blur_radius: int = 17
    deficit_threshold: int = 32


def variants() -> tuple[Variant, ...]:
    paper = [
        Variant(f"paper_open_r{radius}", "paper", open_radius=radius)
        for radius in (1, 2, 3)
    ]
    enclosures = [
        Variant(
            f"enclosure_open{opened}_close{closed}",
            "enclosure",
            open_radius=opened,
            close_radius=closed,
        )
        for opened in (1, 2)
        for closed in (8, 16, 32)
    ]
    filled = [
        Variant(f"filled_close_r{radius}", "filled", close_radius=radius)
        for radius in (4, 8, 16, 32)
    ]
    pixels = [
        Variant(f"superpixel_cell{cell}", "superpixel", cell=cell)
        for cell in (48, 96, 192)
    ]
    return tuple(paper + enclosures + filled + pixels)


def build_labels(variant: Variant, rgb: np.ndarray) -> np.ndarray:
    if variant.family == "paper":
        return paper_after_thin_removal(rgb, variant.open_radius, variant.deficit_threshold)
    if variant.family == "enclosure":
        return boundary_enclosures(
            rgb, variant.open_radius, variant.close_radius, variant.deficit_threshold,
        )
    if variant.family == "filled":
        return filled_closing(rgb, variant.close_radius, variant.deficit_threshold)
    if variant.family == "superpixel":
        return texture_superpixels(rgb, variant.cell, variant.blur_radius)
    raise ValueError(f"unknown family {variant.family}")


def majority_positive(region_ids: np.ndarray, positive: np.ndarray) -> np.ndarray:
    ids = np.asarray(region_ids, dtype=np.int32)
    totals = np.bincount(ids)
    votes = np.bincount(ids, weights=positive.astype(np.float64))
    selected = votes * 2.0 > totals
    return selected[ids]


def purity_at_least(region_ids: np.ndarray, positive: np.ndarray, level: float) -> float:
    ids = np.asarray(region_ids, dtype=np.int32)
    totals = np.bincount(ids).astype(np.float64)
    votes = np.bincount(ids, weights=positive.astype(np.float64))
    majority = np.maximum(votes, totals - votes)
    pure = majority >= level * np.maximum(totals, 1.0)
    pure[totals == 0] = False
    return float(np.mean(pure[ids]))


def _group_queries():
    groups: dict[str, list] = {}
    for query in development_queries():
        groups.setdefault(query.image, []).append(query)
    return groups


def _empty_bucket() -> dict:
    return {"decisions": {}, "stats": [], "seconds": 0.0, "purity_weight": 0.0, "purity_sum": 0.0}


def _record_image(
    bucket: dict, image: str, labels: np.ndarray, queries: list, seconds: float
) -> None:
    stats = area_stats(labels)
    stats["image"] = image
    bucket["stats"].append(stats)
    bucket["seconds"] += seconds
    for query in queries:
        ids = labels[query.y, query.x]
        bucket["decisions"][query.id] = majority_positive(ids, query.positive)
        weight = float(ids.size)
        bucket["purity_weight"] += weight
        bucket["purity_sum"] += weight * purity_at_least(ids, query.positive, 0.9)


def evaluate_variant(variant: Variant, groups: dict[str, list]) -> dict:
    bucket = _empty_bucket()
    for image, queries in groups.items():
        rgb = load_rgb(image)
        started = time.perf_counter()
        labels = build_labels(variant, rgb)
        _record_image(bucket, image, labels, queries, time.perf_counter() - started)
        del rgb, labels
    summary = summarize_decisions(bucket["decisions"])
    return _payload(variant, bucket, summary)


def _payload(variant: Variant, bucket: dict, summary: dict) -> dict:
    counts = [item["region_count"] for item in bucket["stats"]]
    medians = [item["median_area"] for item in bucket["stats"]]
    return {
        "name": variant.name,
        "family": variant.family,
        "oracle": True,
        "rule": "region positive iff strictly more than half of its reviewed pixels are positive",
        "settings": {
            "open_radius": variant.open_radius,
            "close_radius": variant.close_radius,
            "cell": variant.cell,
            "blur_radius": variant.blur_radius,
            "deficit_threshold": variant.deficit_threshold,
            "paper_connectivity": 4,
            "filled_blob_connectivity": 8,
            "structuring_element": "ellipse",
        },
        "partition_seconds": bucket["seconds"],
        "mean_region_count": float(np.mean(counts)),
        "mean_median_area": float(np.mean(medians)),
        "reviewed_pixel_fraction_in_regions_purity_ge_0_9": (
            bucket["purity_sum"] / bucket["purity_weight"] if bucket["purity_weight"] else None
        ),
        "per_document": bucket["stats"],
        "summary": _compact(summary),
    }


def _compact(summary: dict) -> dict:
    keys = (
        "document_macro_iou",
        "global_recall",
        "global_precision",
        "reviewed_negative_fpr",
        "false_discovery_rate",
        "tp",
        "fp",
        "fn",
        "tn",
        "mean_query_iou",
    )
    return {key: _number(summary[key]) for key in keys}


def _number(value: object) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("metric value must be numeric")
    if isinstance(value, (int, np.integer)) and not isinstance(value, (float, np.floating)):
        return int(value)
    return float(value)


def _line(result: dict) -> str:
    summary = result["summary"]
    return (
        f"{result['name']}: iou={summary['document_macro_iou']:.4f} "
        f"recall={summary['global_recall']:.4f} precision={summary['global_precision']:.4f} "
        f"regions={result['mean_region_count']:.0f} median_area={result['mean_median_area']:.0f} "
        f"purity90={result['reviewed_pixel_fraction_in_regions_purity_ge_0_9']:.4f} "
        f"sec={result['partition_seconds']:.1f}"
    )


def main() -> None:
    groups = _group_queries()
    results = []
    for variant in variants():
        result = evaluate_variant(variant, groups)
        results.append(result)
        print(_line(result), flush=True)
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
