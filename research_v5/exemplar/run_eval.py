"""Score label-free test-time exemplar variants on the development split.

Labels never enter the descriptor or the classifier. devharness.report is the
only consumer of development labels.
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research_v5"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from descriptors import (  # noqa: E402
    SheetFeatures,
    channel_table,
    describe_sheet,
    positive_centers,
    self_check_geometry,
    support_window,
)
from models import (  # noqa: E402
    blur_score,
    close_score,
    dilate_score,
    kept_columns,
    knn_ratio,
    logreg_grid_scale,
    logreg_quantiles,
    logreg_scores,
    mahalanobis_ratio,
    pack_training,
    prototype_ratio,
    sample_bilinear,
    svm_scores,
    trimmed_knn_ratio,
    trimmed_logreg_scores,
    unlabeled_rows,
)

import devharness  # noqa: E402

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
FINDINGS = HERE / "findings.txt"
FEATURE_VERSION = "ex1-stride8-periods4to48"
CORE_FAMILIES = frozenset(
    {
        "ink_mean",
        "ink_std",
        "chroma_mean",
        "chroma_std",
        "color",
        "grad_energy",
        "coherence",
        "cos2",
        "sin2",
        "tensor_energy",
        "spectral",
    }
)
FUSE_BASES = (
    "logreg_bal_C0.3_q05",
    "logreg_bal_C0.3_q20",
    "logreg_core_bal_C0.3_q05",
    "logreg_trim_bal_C0.3_q05",
    "prototype_ratio",
    "knn_k5_ratio",
    "mahalanobis_pca16_ratio",
)
FUSE_KINDS = ("geo", "min", "gate035")
SHARED = (
    "stride=8; density=ink mean/std and chroma mean/std at 8/16/32/64; "
    "color=rgb mean at 8/16/32; hog=8 unsigned bins at 8/16/32; "
    "grad energy at 8/16/32/64; structure coherence/cos2/sin2/energy at 16/32/64; "
    "spectral=mean-removed local DFT amplitude/(ink_std+0.02), clipped to 8, "
    "periods 4/8/12/16/24/32/48, 8 angles step 22.5 deg, windows 16/32/64 with period<=window; "
    "positives=centers whose support window lies fully inside the query, step 2 px (shift "
    "augmentation), cap 480; support=largest of 64/32/16/8 with at least 12 such centers; "
    "unlabeled=up to 6000 grid cells with centers outside the query, spatially strided; "
    "channels with window>support dropped; scores calibrated with "
    "sigmoid((s-percentile(train positives, q))/max(IQR, std)); "
    "pixel score=bilinear sample of the full-grid score; "
    "development-selected among these global heads, not fit per query on labels"
)
HEADS = {
    "hgb_cache": "existing crop-descriptor HistGradientBoosting probabilities from devharness.HGB_CACHE",
    "logreg_bal_C0.3_q05": "LogisticRegression C=0.3 class_weight=balanced solver=lbfgs max_iter=400; anchor q=5",
    "logreg_bal_C0.3_q20": "LogisticRegression C=0.3 class_weight=balanced solver=lbfgs max_iter=400; anchor q=20",
    "logreg_bal_C0.3_q05_blur3": "same model as logreg_bal_C0.3_q05, then 3x3 box blur on the score grid",
    "logreg_bal_C1_q05": "LogisticRegression C=1 class_weight=balanced solver=lbfgs max_iter=400; anchor q=5",
    "logreg_unweighted_C1_q05": "LogisticRegression C=1 class_weight=None solver=lbfgs max_iter=400; anchor q=5",
    "logreg_trim_bal_C0.3_q05": "drop the farthest 30% of positives from their median, then balanced logreg C=0.3; anchor q=5",
    "svm_bal_C0.3_q05": "LinearSVC C=0.3 class_weight=balanced dual=False max_iter=4000; anchor q=5",
    "logreg_core_bal_C0.3_q05": "balanced logreg C=0.3 on density, color, grad energy, structure, and spectral only (no HOG); anchor q=5",
    "prototype_ratio": "diagonal Mahalanobis ratio: dist(unlabeled mean)/(dist(positive median)+dist(unlabeled mean)), whitened by unlabeled std",
    "knn_k5_ratio": "k=5 mean distance to at most 96 positive rows vs 96 unlabeled rows, same whitening; ratio of those distances",
    "mahalanobis_pca16_ratio": "PCA 16 fit on whitened unlabeled cells, ridge 1e-2 covariance, dist to unlabeled mean vs dist to positive median",
    "knn_k1_ratio": "k=1 distance to at most 96 positive rows vs 96 unlabeled rows, whitened; distance ratio",
    "knn_k15_ratio": "k=15 mean distance, same cap and whitening as knn_k5_ratio",
    "knn_k1_trim_ratio": "drop farthest 30% of positives from their median, then k=1 distance ratio",
    "logreg_wide_gridscale": "balanced LogisticRegression C=0.3; sigmoid anchor=min(train positive decision), spread=std of the full score grid",
    "knn_k1_max5": "knn_k1_ratio then 5x5 maximum filter on the stride-8 score grid (40 px)",
    "knn_k1_max9": "knn_k1_ratio then 9x9 maximum filter on the stride-8 score grid (72 px)",
    "knn_k1_max15": "knn_k1_ratio then 15x15 maximum filter on the stride-8 score grid (120 px)",
    "knn_k5_max5": "knn_k5_ratio then 5x5 maximum filter (40 px)",
    "knn_k5_max9": "knn_k5_ratio then 9x9 maximum filter (72 px)",
    "knn_k5_max15": "knn_k5_ratio then 15x15 maximum filter (120 px)",
    "knn_k1_trim_max9": "knn_k1_trim_ratio then 9x9 maximum filter (72 px)",
    "maha_max9": "mahalanobis_pca16_ratio then 9x9 maximum filter (72 px)",
    "maha_max15": "mahalanobis_pca16_ratio then 15x15 maximum filter (120 px)",
    "knn_k1_trim_max7": "trimmed k=1 ratio, then 7x7 maximum filter (56 px)",
    "knn_k1_trim_max11": "trimmed k=1 ratio, then 11x11 maximum filter (88 px)",
    "knn_k1_close9": "k=1 ratio, then grayscale closing with a 9x9 kernel (72 px)",
    "knn_k5_close9": "k=5 ratio, then grayscale closing with a 9x9 kernel (72 px)",
    "knn_k1_trim_close5": "trimmed k=1 ratio, then grayscale closing with a 5x5 kernel (40 px)",
    "knn_k1_trim_close9": "trimmed k=1 ratio, then grayscale closing with a 9x9 kernel (72 px)",
    "knn_k1_trim_close15": "trimmed k=1 ratio, then grayscale closing with a 15x15 kernel (120 px)",
    "knn_k1_trim_close9_max3": "trimmed k=1 closing 9x9, then a 3x3 maximum filter (24 px)",
}


def main() -> None:
    if "--check" in sys.argv:
        self_check()
        print("self_check ok")
        return
    if "--round2" in sys.argv:
        run_round2()
        return
    if "--round3" in sys.argv:
        run_round3()
        return
    begin_findings()
    self_check()
    hgb = devharness.cached_scores(devharness.HGB_CACHE)
    record("hgb_cache", hgb)
    collected = score_development()
    for name in list(collected):
        record(name, collected[name])
    report_fusions(collected, hgb)
    print("done", FINDINGS)


def self_check() -> None:
    self_check_geometry()
    _check_flat_spectrum()
    _check_grating_separates()


def begin_findings() -> None:
    FINDINGS.write_text(
        "Test-time exemplar matcher on the 55-query development split.\n"
        "Each block is one harness sweep. all_goals_met uses macro IoU>=0.95, "
        "recall>=0.98, precision>=0.95, reviewed-negative FPR<=0.05.\n\n",
        encoding="utf-8",
    )


def score_development() -> dict[str, dict[str, np.ndarray]]:
    collected: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    grouped = _queries_by_image(devharness.development_queries())
    for image, queries in grouped.items():
        started = time.perf_counter()
        sheet, boxes, supports = _sheet_for(image, queries)
        print(
            f"{image} {sheet.grid.shape[1]}x{sheet.grid.shape[0]} cells "
            f"in {time.perf_counter() - started:.1f}s",
            flush=True,
        )
        _score_image(collected, sheet, boxes, supports, queries)
        del sheet
    return collected


def record(name: str, scores: dict[str, np.ndarray]) -> None:
    started = time.perf_counter()
    result = devharness.report(name, scores, HERE / f"{name}.json")
    _append_finding(name, result)
    best = result["selections"]["best_macro_iou"]
    at_recall = result["selections"]["best_precision_at_recall_ge_0_98"]
    print(
        f"{name} iou={_num(best, 'document_macro_iou')} "
        f"p_at_r98={_num(at_recall, 'global_precision')} "
        f"({time.perf_counter() - started:.1f}s)",
        flush=True,
    )


def report_fusions(collected: dict[str, dict[str, np.ndarray]], hgb: dict[str, np.ndarray]) -> None:
    for base in FUSE_BASES:
        for kind in FUSE_KINDS:
            fused = {
                key: combine(kind, collected[base][key], hgb[key]) for key in collected[base]
            }
            record(f"fuse_{kind}__{base}", fused)


def settings_for(name: str) -> str:
    if name.startswith("fuse_"):
        kind, base = name.split("__", 1)
        return f"{kind} fusion of [{base}] with HGB cache probabilities. {settings_for(base)}"
    head = HEADS.get(name, name)
    return f"{SHARED} Head: {head}"


def combine(kind: str, exemplar: np.ndarray, hgb: np.ndarray) -> np.ndarray:
    left = np.clip(exemplar, 0.0, 1.0)
    right = np.clip(hgb, 0.0, 1.0)
    if kind == "geo":
        return np.sqrt(left * right).astype(np.float32)
    if kind == "min":
        return np.minimum(left, right).astype(np.float32)
    return (right * np.clip(left / np.float32(0.35), 0.0, 1.0)).astype(np.float32)


def _queries_by_image(queries):
    grouped = {}
    for query in queries:
        grouped.setdefault(query.image, []).append(query)
    return grouped


def _sheet_for(image: str, queries):
    boxes, supports, groups = _sites(queries)
    cached = _read_cache(image, boxes)
    if cached is not None:
        return cached, boxes, supports
    sheet = describe_sheet(devharness.load_rgb(image), groups)
    _write_cache(image, boxes, sheet)
    return sheet, boxes, supports


def _sites(queries):
    boxes, supports, groups = [], [], []
    for query in queries:
        box = tuple(int(value) for value in query.query_box)
        window = support_window(box)
        boxes.append(box)
        supports.append(window)
        groups.append(positive_centers(box, window))
    return boxes, supports, tuple(groups)


def _score_image(collected, sheet, boxes, supports, queries) -> None:
    for index, query in enumerate(queries):
        sampled = _sample_query(sheet, index, boxes[index], supports[index], query.y, query.x)
        for name, values in sampled.items():
            collected[name][query.id] = values
        print(f"  support={supports[index]} positives={sheet.points[index].shape[0]}", flush=True)


def _sample_query(sheet, index, box, support, y, x):
    maps = _full_maps(sheet, index, box, support)
    maps.update(_core_map(sheet, index, box, support))
    sampled = {}
    for name, score in maps.items():
        values = sample_bilinear(score, sheet.grid_y, sheet.grid_x, y, x)
        if values.shape != y.shape:
            raise RuntimeError(f"sampled {values.shape} != coordinates {y.shape}")
        sampled[name] = values
    return sampled


def _full_maps(sheet, index, box, support):
    flat, positives, unlabeled = _packed(sheet, index, box, support, None)
    height, width = sheet.grid.shape[:2]
    anchored = logreg_quantiles(flat, positives, unlabeled, height, width, 0.3, True, (5, 20))
    q05 = anchored[5]
    return {
        "logreg_bal_C0.3_q05": q05,
        "logreg_bal_C0.3_q20": anchored[20],
        "logreg_bal_C0.3_q05_blur3": blur_score(q05, 1),
        "logreg_bal_C1_q05": logreg_scores(flat, positives, unlabeled, height, width, 1.0, True, 5),
        "logreg_unweighted_C1_q05": logreg_scores(
            flat, positives, unlabeled, height, width, 1.0, False, 5
        ),
        "logreg_trim_bal_C0.3_q05": trimmed_logreg_scores(
            flat, positives, unlabeled, height, width, 0.3, 5
        ),
        "svm_bal_C0.3_q05": svm_scores(flat, positives, unlabeled, height, width, 0.3, 5),
        "prototype_ratio": prototype_ratio(flat, positives, unlabeled, height, width),
        "knn_k5_ratio": knn_ratio(flat, positives, unlabeled, height, width),
        "mahalanobis_pca16_ratio": mahalanobis_ratio(flat, positives, unlabeled, height, width),
    }


def _core_map(sheet, index, box, support):
    flat, positives, unlabeled = _packed(sheet, index, box, support, CORE_FAMILIES)
    height, width = sheet.grid.shape[:2]
    score = logreg_scores(flat, positives, unlabeled, height, width, 0.3, True, 5)
    return {"logreg_core_bal_C0.3_q05": score}


def _packed(sheet, index, box, support, families):
    columns = kept_columns(channel_table(), support, families)
    rows = unlabeled_rows(sheet.grid_y, sheet.grid_x, box)
    return pack_training(sheet.grid, sheet.points[index], rows, columns)


def _cache_paths(image: str) -> tuple[Path, Path, Path]:
    digest = hashlib.sha256(f"{FEATURE_VERSION}|{image}".encode()).hexdigest()[:20]
    CACHE.mkdir(parents=True, exist_ok=True)
    return CACHE / f"{digest}.npy", CACHE / f"{digest}_pos.npz", CACHE / f"{digest}_axes.npz"


def _read_cache(image: str, boxes):
    grid_path, pos_path, axes_path = _cache_paths(image)
    if not (grid_path.exists() and pos_path.exists() and axes_path.exists()):
        return None
    return _load_matching(grid_path, pos_path, axes_path, boxes)


def _load_matching(grid_path, pos_path, axes_path, boxes):
    stored = np.load(pos_path)
    if int(stored["count"][0]) != len(boxes):
        return None
    saved = [tuple(int(value) for value in stored[f"b{i}"]) for i in range(len(boxes))]
    if saved != list(boxes):
        return None
    grid = np.array(np.load(grid_path))
    if grid.shape[-1] != len(channel_table()):
        return None
    axes = np.load(axes_path)
    points = tuple(np.array(stored[f"p{index}"]) for index in range(len(boxes)))
    loaded = devharness_sheet(np.array(axes["y"]), np.array(axes["x"]), grid, points)
    stored.close()
    axes.close()
    return loaded


def devharness_sheet(grid_y, grid_x, grid, points):
    return SheetFeatures(grid_y, grid_x, grid, points)


def _write_cache(image: str, boxes, sheet) -> None:
    grid_path, pos_path, axes_path = _cache_paths(image)
    payload = {"count": np.array([len(boxes)], dtype=np.int32)}
    for index, box in enumerate(boxes):
        payload[f"b{index}"] = np.asarray(box, dtype=np.int32)
        payload[f"p{index}"] = sheet.points[index]
    np.save(grid_path, sheet.grid)
    np.savez(pos_path, **payload)
    np.savez(axes_path, y=sheet.grid_y, x=sheet.grid_x)


def _append_finding(name: str, result: dict) -> None:
    selections = result["selections"]
    text = (
        f"variant: {name}\n"
        f"settings: {settings_for(name)}\n"
        f"best_macro_iou: {_point(selections['best_macro_iou'])}\n"
        f"best_precision_at_recall_ge_0.98: {_point(selections['best_precision_at_recall_ge_0_98'])}\n"
        f"all_goals_met: {selections['all_goals_met']}\n\n"
    )
    with FINDINGS.open("a", encoding="utf-8") as handle:
        handle.write(text)


def _point(point: dict | None) -> str:
    if not point:
        return "none"
    return (
        f"iou={point['document_macro_iou']:.6f} precision={point['global_precision']:.6f} "
        f"recall={point['global_recall']:.6f} threshold={point['threshold']:.2f} "
        f"fpr={point['reviewed_negative_fpr']:.6f}"
    )


def _num(point: dict | None, key: str) -> str:
    if not point or point[key] is None:
        return "none"
    return f"{point[key]:.4f}"


def _check_flat_spectrum() -> None:
    flat = np.full((64, 64, 3), 180, np.uint8)
    sheet = describe_sheet(flat, tuple(), stride=8)
    spectral = [index for index, channel in enumerate(channel_table()) if channel.kind == "spectral"]
    peak = float(np.max(np.abs(sheet.grid[:, :, spectral])))
    if peak > 0.2:
        raise RuntimeError(f"flat sheet spectral peak {peak}")


def _check_grating_separates() -> None:
    image = np.full((256, 256, 3), 240, np.uint8)
    yy, xx = np.indices((256, 256))
    image[(xx < 128) & ((xx % 12) < 2)] = 0
    image[(xx >= 128) & ((yy % 8) < 2)] = 0
    box = (32, 32, 96, 96)
    window = support_window(box)
    centers = positive_centers(box, window)
    sheet = describe_sheet(image, (centers,), stride=8)
    flat, positives, unlabeled = _packed(sheet, 0, box, window, None)
    score = prototype_ratio(flat, positives, unlabeled, sheet.grid.shape[0], sheet.grid.shape[1])
    left = _mean_region(score, sheet, 40, 120, 100, 124)
    right = _mean_region(score, sheet, 40, 120, 160, 230)
    if left <= right:
        raise RuntimeError(f"grating prototype left {left:.3f} <= right {right:.3f}")


def _mean_region(score, sheet, y0, y1, x0, x1) -> float:
    rows = (sheet.grid_y >= y0) & (sheet.grid_y < y1)
    cols = (sheet.grid_x >= x0) & (sheet.grid_x < x1)
    return float(score[rows[:, None] & cols[None, :]].mean())


ROUND2_FUSE = (
    "knn_k1_ratio",
    "knn_k1_max9",
    "knn_k1_max15",
    "knn_k5_max9",
    "knn_k1_trim_ratio",
    "knn_k1_trim_max9",
    "maha_max9",
    "maha_max15",
    "knn_k15_ratio",
    "logreg_wide_gridscale",
)


def run_round2() -> None:
    with FINDINGS.open("a", encoding="utf-8") as handle:
        handle.write(
            "Round 2: neighborhood max on the score grid (paper between strokes inherits "
            "a nearby cell), k=1 and trimmed positives, and a wider logistic calibration.\n\n"
        )
    hgb = devharness.cached_scores(devharness.HGB_CACHE)
    collected = _score_round2()
    for name in list(collected):
        record(name, collected[name])
    for base in ROUND2_FUSE:
        for kind in ("geo", "min"):
            fused = {key: combine(kind, collected[base][key], hgb[key]) for key in collected[base]}
            record(f"fuse_{kind}__{base}", fused)
    print("round2 done", FINDINGS)


def _score_round2() -> dict[str, dict[str, np.ndarray]]:
    collected: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    grouped = _queries_by_image(devharness.development_queries())
    for image, queries in grouped.items():
        started = time.perf_counter()
        sheet, boxes, supports = _sheet_for(image, queries)
        print(f"{image} cache {time.perf_counter() - started:.1f}s", flush=True)
        _score_round2_image(collected, sheet, boxes, supports, queries)
        del sheet
    return collected


def _score_round2_image(collected, sheet, boxes, supports, queries) -> None:
    for index, query in enumerate(queries):
        sampled = _sample_round2(sheet, index, boxes[index], supports[index], query.y, query.x)
        for name, values in sampled.items():
            collected[name][query.id] = values


def _sample_round2(sheet, index, box, support, y, x):
    maps = _round2_maps(sheet, index, box, support)
    sampled = {}
    for name, score in maps.items():
        values = sample_bilinear(score, sheet.grid_y, sheet.grid_x, y, x)
        if values.shape != y.shape:
            raise RuntimeError(f"sampled {values.shape} != coordinates {y.shape}")
        sampled[name] = values
    return sampled


def _round2_maps(sheet, index, box, support):
    flat, positives, unlabeled = _packed(sheet, index, box, support, None)
    height, width = sheet.grid.shape[:2]
    k1 = knn_ratio(flat, positives, unlabeled, height, width, 1)
    k5 = knn_ratio(flat, positives, unlabeled, height, width, 5)
    k15 = knn_ratio(flat, positives, unlabeled, height, width, 15)
    trimmed = trimmed_knn_ratio(flat, positives, unlabeled, height, width, 1)
    maha = mahalanobis_ratio(flat, positives, unlabeled, height, width)
    wide = logreg_grid_scale(flat, positives, unlabeled, height, width, 0.3)
    maps = {
        "knn_k1_ratio": k1,
        "knn_k15_ratio": k15,
        "knn_k1_trim_ratio": trimmed,
        "logreg_wide_gridscale": wide,
        "knn_k1_trim_max9": dilate_score(trimmed, 9),
        "maha_max9": dilate_score(maha, 9),
        "maha_max15": dilate_score(maha, 15),
    }
    for size in (5, 9, 15):
        maps[f"knn_k1_max{size}"] = dilate_score(k1, size)
        maps[f"knn_k5_max{size}"] = dilate_score(k5, size)
    return maps


def run_round3() -> None:
    with FINDINGS.open("a", encoding="utf-8") as handle:
        handle.write(
            "Round 3: grayscale closing (max then min of the same size) and neighborhood "
            "sizes 7 and 11 around the round-2 peak at 9.\n\n"
        )
    hgb = devharness.cached_scores(devharness.HGB_CACHE)
    collected = _score_named(_round3_maps)
    for name in list(collected):
        record(name, collected[name])
    for base in ("knn_k1_trim_close9", "knn_k1_trim_close9_max3", "knn_k1_trim_max7"):
        for kind in ("geo", "min"):
            fused = {key: combine(kind, collected[base][key], hgb[key]) for key in collected[base]}
            record(f"fuse_{kind}__{base}", fused)
    print("round3 done", FINDINGS)


def _score_named(builder) -> dict[str, dict[str, np.ndarray]]:
    collected: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    grouped = _queries_by_image(devharness.development_queries())
    for image, queries in grouped.items():
        sheet, boxes, supports = _sheet_for(image, queries)
        _collect_builder(collected, builder, sheet, boxes, supports, queries)
        del sheet
    return collected


def _collect_builder(collected, builder, sheet, boxes, supports, queries) -> None:
    for index, query in enumerate(queries):
        maps = builder(sheet, index, boxes[index], supports[index])
        _store_query(collected, maps, sheet, query)


def _store_query(collected, maps, sheet, query) -> None:
    for name, score in maps.items():
        values = sample_bilinear(score, sheet.grid_y, sheet.grid_x, query.y, query.x)
        if values.shape != query.y.shape:
            raise RuntimeError(f"sampled {values.shape} != coordinates {query.y.shape}")
        collected[name][query.id] = values


def _round3_maps(sheet, index, box, support):
    flat, positives, unlabeled = _packed(sheet, index, box, support, None)
    height, width = sheet.grid.shape[:2]
    trimmed = trimmed_knn_ratio(flat, positives, unlabeled, height, width, 1)
    k1 = knn_ratio(flat, positives, unlabeled, height, width, 1)
    k5 = knn_ratio(flat, positives, unlabeled, height, width, 5)
    closed9 = close_score(trimmed, 9)
    return {
        "knn_k1_trim_max7": dilate_score(trimmed, 7),
        "knn_k1_trim_max11": dilate_score(trimmed, 11),
        "knn_k1_close9": close_score(k1, 9),
        "knn_k5_close9": close_score(k5, 9),
        "knn_k1_trim_close5": close_score(trimmed, 5),
        "knn_k1_trim_close9": closed9,
        "knn_k1_trim_close15": close_score(trimmed, 15),
        "knn_k1_trim_close9_max3": dilate_score(closed9, 3),
    }


if __name__ == "__main__":
    main()
