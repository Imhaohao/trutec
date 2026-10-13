"""Region scores that compare a region's texture profile with the query crop.

Aggregation reads the full feature grid and the cached full probability grid.
Reviewed coordinates are used only to sample the finished scores for the harness.
Choosing among these formulas is development-selected.
"""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research_v5"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aggregate import (  # noqa: E402
    center_labels,
    cosine_similarity,
    feature_columns,
    gabor_angle_correlation,
    gabor_wavelength_correlation,
    lookup,
    mass_rank,
    profile_correlation,
    sample_pixels,
)
from deploy import (  # noqa: E402
    RESULTS,
    _brief,
    _cache_gap,
    _load_probability,
    _require_cached,
    _require_same_grid,
    gabor_index,
)
from devharness import load_gray, load_rgb, report  # noqa: E402
import hatch_texture  # noqa: E402
from oracle import Variant, build_labels  # noqa: E402

STRIDE = 4
MIXED = (
    "cosine_mass_rank",
    "corr_angle_mass_rank",
    "corr_gabor_mass_rank",
    "corr_angle",
    "cosine_full",
)


def pattern_variants() -> tuple[Variant, ...]:
    return (
        Variant("superpixel_cell48", "superpixel", cell=48),
        Variant("superpixel_cell96", "superpixel", cell=96),
    )


def pattern_columns(medians: np.ndarray, query: np.ndarray, gabor_columns: np.ndarray, counts: np.ndarray) -> dict[str, np.ndarray]:
    """Label-free region scores. Each column aligns with region ids."""
    every = np.arange(medians.shape[1], dtype=np.int32)
    columns = {
        "cosine_full": cosine_similarity(medians, query, every, counts),
        "corr_full": profile_correlation(medians, query, every, counts),
        "corr_gabor": profile_correlation(medians, query, gabor_columns, counts),
        "corr_angle": gabor_angle_correlation(medians, query, gabor_columns, counts),
        "corr_wavelength": gabor_wavelength_correlation(medians, query, gabor_columns, counts),
    }
    columns["cosine_mass_rank"] = mass_rank(columns["cosine_full"], counts)
    columns["corr_angle_mass_rank"] = mass_rank(columns["corr_angle"], counts)
    columns["corr_gabor_mass_rank"] = mass_rank(columns["corr_gabor"], counts)
    return columns


def pixel_mixes(pixel: np.ndarray, region: np.ndarray, name: str) -> dict[str, np.ndarray]:
    return {
        f"max_pixel__{name}": np.maximum(pixel, region),
        f"product_pixel__{name}": (pixel * region).astype(np.float32),
    }


def score_patterns(variant: Variant, gabor_columns: np.ndarray) -> dict:
    from deploy import grouped_queries

    bucket: dict[str, dict] = {}
    gaps: list[float] = []
    started = time.perf_counter()
    for image, queries in grouped_queries().items():
        _require_cached(queries)
        _score_pattern_image(bucket, gaps, variant, gabor_columns, image, queries)
        print(f"{variant.name} finished {image}", flush=True)
    selections = _write_pattern_scores(variant.name, bucket)
    payload = {
        "partition": variant.name,
        "seconds": time.perf_counter() - started,
        "max_abs_pixel_gap_vs_hgb_cache": max(gaps) if gaps else None,
        "median_of_per_query_max_abs_gap": _median_gap(gaps),
        "selections": selections,
    }
    destination = RESULTS / f"{variant.name}_pattern_selections.json"
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {destination}", flush=True)
    return payload


def _score_pattern_image(bucket, gaps, variant, gabor_columns, image, queries) -> None:
    rgb = load_rgb(image)
    gray = load_gray(image)
    labels = build_labels(variant, rgb)
    features = hatch_texture.texture_features(gray, stride=STRIDE)
    sampled, _rows, _cols = center_labels(labels, STRIDE)
    _require_same_grid(image, sampled, features)
    _means, medians, counts = feature_columns(features, sampled)
    width = int(labels.max()) + 1
    for query in queries:
        _score_pattern_query(
            bucket, gaps, gabor_columns, query, labels, sampled, gray, features, medians, counts, width,
        )
    del rgb, gray, labels, features


def _score_pattern_query(
    bucket, gaps, gabor_columns, query, labels, sampled, gray, features, medians, counts, width,
) -> None:
    probability = _load_probability(query.id, sampled.shape)
    descriptor = hatch_texture.descriptor_for_query(gray, features, query.query_box, STRIDE, "crop")
    columns = pattern_columns(medians, descriptor, gabor_columns, counts)
    pixel = sample_pixels(probability, query.y, query.x, STRIDE, query.height, query.width)
    gaps.append(_cache_gap(pixel, query.id))
    ids = labels[query.y, query.x]
    _store_pattern(bucket, query.id, pixel, ids, columns, width)


def _store_pattern(bucket, query_id, pixel, ids, columns, width) -> None:
    looked = {name: lookup(_pad_column(column, width), ids) for name, column in columns.items()}
    for name in MIXED:
        looked.update(pixel_mixes(pixel, looked[name], name))
    for name, values in looked.items():
        bucket.setdefault(name, {})[query_id] = values


def _pad_column(column: np.ndarray, width: int) -> np.ndarray:
    if column.shape[0] == width:
        return column
    if column.shape[0] > width:
        raise ValueError("aggregated column is longer than the label image")
    padded = np.zeros(width, dtype=np.float32)
    padded[:column.shape[0]] = column
    return padded


def _write_pattern_scores(partition: str, bucket: dict[str, dict]) -> dict:
    selections = {}
    for name, scores in bucket.items():
        result = report(f"{partition}__{name}", scores, RESULTS / f"{partition}__{name}.json")
        selections[name] = result["selections"]
        best = result["selections"]["best_macro_iou"]
        held = result["selections"]["best_precision_at_recall_ge_0_98"]
        print(f"{partition}__{name}: best_iou={_brief(best)} at_recall_0.98={_brief(held)}", flush=True)
    return selections


def _median_gap(gaps: list[float]) -> float | None:
    if not gaps:
        return None
    return float(np.median(np.asarray(gaps, dtype=np.float64)))


def main() -> None:
    cv2.setNumThreads(4)
    requested = set(sys.argv[1:])
    chosen = [item for item in pattern_variants() if not requested or item.name in requested]
    if not chosen:
        raise SystemExit(f"no matching partition in {sorted(requested)}")
    gabor_columns = gabor_index()
    for variant in chosen:
        score_patterns(variant, gabor_columns)


if __name__ == "__main__":
    main()
