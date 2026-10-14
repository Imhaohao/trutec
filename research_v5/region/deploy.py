"""Score label-free region aggregates of the crop texture model.

Choosing among these global formulas is development-selected. Aggregation uses
the full probability grid and the full feature grid, never reviewed labels.
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

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research_v5"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aggregate import (  # noqa: E402
    center_labels,
    feature_columns,
    gabor_similarity,
    lookup,
    mix_pixel_and_region,
    probability_columns,
    sample_pixels,
    score_descriptors,
    spread_present,
    weighted_mean,
)
from build_probs import MODEL, cache_path, grouped_queries  # noqa: E402
from devharness import HGB_CACHE, cached_scores, load_gray, load_rgb, report  # noqa: E402
import hatch_texture  # noqa: E402
from inference_texture import verify_model_metadata  # noqa: E402
from oracle import Variant, build_labels  # noqa: E402
from texture_learning import load_texture_model  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
STRIDE = 4
MIX_BASES = ("prob_mean", "median_feature_hgb", "gabor_similarity", "gabor_spread")


def deploy_variants() -> tuple[Variant, ...]:
    return (
        Variant("superpixel_cell96", "superpixel", cell=96),
        Variant("superpixel_cell48", "superpixel", cell=48),
        Variant("superpixel_cell192", "superpixel", cell=192),
        Variant("enclosure_open1_close8", "enclosure", open_radius=1, close_radius=8),
    )


def gabor_index() -> np.ndarray:
    names = hatch_texture.texture_feature_names()
    return np.array([i for i, name in enumerate(names) if name.startswith("gabor_")], dtype=np.int32)


def ink_channel() -> int:
    return hatch_texture.texture_feature_names().index("ink_density_w16")


def _require_cached(queries: list) -> None:
    missing = [query.id for query in queries if not cache_path(query.id).is_file()]
    if missing:
        raise FileNotFoundError(f"missing probability cache for {missing[:3]}")


def _load_probability(query_id: str, shape: tuple[int, int]) -> np.ndarray:
    probability = np.load(cache_path(query_id)).astype(np.float32)
    if probability.shape != shape:
        raise ValueError(f"{query_id}: grid {probability.shape} != {shape}")
    return probability


def _query_columns(
    model, descriptor, sampled, probability, ink, means, medians, counts, gabor_columns, width,
) -> dict[str, np.ndarray]:
    columns = probability_columns(sampled, probability)
    columns["prob_ink_weighted_mean"] = weighted_mean(sampled, probability, ink)
    columns["mean_feature_hgb"] = score_descriptors(model, means, descriptor, counts)
    columns["median_feature_hgb"] = score_descriptors(model, medians, descriptor, counts)
    columns["gabor_similarity"] = gabor_similarity(medians, descriptor, gabor_columns, counts)
    columns["gabor_spread"] = spread_present(columns["gabor_similarity"], counts)
    columns["median_hgb_spread"] = spread_present(columns["median_feature_hgb"], counts)
    return {name: _pad(column, width) for name, column in columns.items()}


def _pad(column: np.ndarray, width: int) -> np.ndarray:
    if column.shape[0] == width:
        return column
    if column.shape[0] > width:
        raise ValueError("aggregated column is longer than the label image")
    padded = np.zeros(width, dtype=np.float32)
    padded[:column.shape[0]] = column
    return padded


def _store_query(bucket: dict[str, dict], query_id: str, pixel: np.ndarray, ids: np.ndarray, columns: dict) -> None:
    looked = {name: lookup(column, ids) for name, column in columns.items()}
    looked["pixel_float16"] = pixel
    for base in MIX_BASES:
        for mix_name, values in mix_pixel_and_region(pixel, looked[base]).items():
            looked[f"{mix_name}__{base}"] = values
    for name, values in looked.items():
        bucket.setdefault(name, {})[query_id] = values


def _cache_gap(pixel: np.ndarray, query_id: str) -> float:
    cached = np.load(HGB_CACHE / f"{query_id}.npz")["probability"]
    if cached.shape != pixel.shape:
        raise ValueError(f"{query_id}: cache length {cached.shape} != {pixel.shape}")
    return float(np.max(np.abs(pixel.astype(np.float32) - cached.astype(np.float32))))


def score_variant(variant: Variant, model, gabor_columns: np.ndarray, ink_axis: int) -> dict:
    groups = grouped_queries()
    bucket: dict[str, dict] = {}
    gaps: list[float] = []
    started = time.perf_counter()
    for image, queries in groups.items():
        _require_cached(queries)
        _score_image(bucket, gaps, variant, model, gabor_columns, ink_axis, image, queries)
        print(f"{variant.name} finished {image}", flush=True)
    selections = _write_scores(variant.name, bucket)
    payload = {
        "partition": variant.name,
        "seconds": time.perf_counter() - started,
        "max_abs_pixel_gap_vs_hgb_cache": max(gaps) if gaps else None,
        "selections": selections,
    }
    destination = RESULTS / f"{variant.name}_selections.json"
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {destination}", flush=True)
    return payload


def _score_image(bucket, gaps, variant, model, gabor_columns, ink_axis, image, queries) -> None:
    rgb = load_rgb(image)
    gray = load_gray(image)
    labels = build_labels(variant, rgb)
    features = hatch_texture.texture_features(gray, stride=STRIDE)
    sampled, _rows, _cols = center_labels(labels, STRIDE)
    _require_same_grid(image, sampled, features)
    means, medians, counts = feature_columns(features, sampled)
    ink = features[:, :, ink_axis]
    width = int(labels.max()) + 1
    for query in queries:
        _score_one_query(
            bucket, gaps, model, gabor_columns, query, labels, sampled,
            gray, features, means, medians, counts, ink, width,
        )
    del rgb, gray, labels, features


def _require_same_grid(image: str, sampled: np.ndarray, features: np.ndarray) -> None:
    if sampled.shape != features.shape[:2]:
        raise ValueError(f"{image}: label grid {sampled.shape} != features {features.shape[:2]}")


def _score_one_query(
    bucket, gaps, model, gabor_columns, query, labels, sampled,
    gray, features, means, medians, counts, ink, width,
) -> None:
    probability = _load_probability(query.id, sampled.shape)
    descriptor = hatch_texture.descriptor_for_query(
        gray, features, query.query_box, STRIDE, "crop",
    )
    columns = _query_columns(
        model, descriptor, sampled, probability, ink, means, medians, counts, gabor_columns, width,
    )
    pixel = sample_pixels(probability, query.y, query.x, STRIDE, query.height, query.width)
    gaps.append(_cache_gap(pixel, query.id))
    _store_query(bucket, query.id, pixel, labels[query.y, query.x], columns)


def _write_scores(partition: str, bucket: dict[str, dict]) -> dict:
    RESULTS.mkdir(parents=True, exist_ok=True)
    selections = {}
    for name, scores in bucket.items():
        result = report(f"{partition}__{name}", scores, RESULTS / f"{partition}__{name}.json")
        selections[name] = result["selections"]
        best = result["selections"]["best_macro_iou"]
        held = result["selections"]["best_precision_at_recall_ge_0_98"]
        print(
            f"{partition}__{name}: best_iou={_brief(best)} at_recall_0.98={_brief(held)}",
            flush=True,
        )
    return selections


def _brief(point: dict | None) -> str:
    if point is None:
        return "none"
    return (
        f"thr={point['threshold']:.2f} iou={point['document_macro_iou']:.4f} "
        f"recall={point['global_recall']:.4f} precision={point['global_precision']:.4f}"
    )


def _write_pixel_baseline() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    result = report("hgb_cache_pixel", cached_scores(HGB_CACHE), RESULTS / "hgb_cache_pixel.json")
    best = result["selections"]["best_macro_iou"]
    held = result["selections"]["best_precision_at_recall_ge_0_98"]
    print(f"hgb_cache_pixel: best_iou={_brief(best)} at_recall_0.98={_brief(held)}", flush=True)


def main() -> None:
    requested = set(sys.argv[1:])
    model, metadata = load_texture_model(MODEL)
    verify_model_metadata(metadata)
    _write_pixel_baseline()
    chosen = [item for item in deploy_variants() if not requested or item.name in requested]
    if not chosen:
        raise SystemExit(f"no matching partition in {sorted(requested)}")
    for variant in chosen:
        score_variant(variant, model, gabor_index(), ink_channel())


if __name__ == "__main__":
    main()
