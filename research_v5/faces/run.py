"""Fit-split constant selection, development scoring, and face previews.

Inference goes through matcher.variant_scores. Fit labels are read only inside
the fit selector. Development labels are read only by devharness.report.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research_v5"))

import devharness  # noqa: E402
from evaluate import reviewed_domains  # noqa: E402

from barriers import _family_drop, build_faces  # noqa: E402
from matcher import clear_cache, variant_scores  # noqa: E402
from scoring import sample_maps, score_maps  # noqa: E402
from settings import PERIODIC, SOLID, DEFAULT, Settings  # noqa: E402
from signature import _measure, find_exemplar, measure_sheet  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PREVIEWS = HERE / "previews"
Image.MAX_IMAGE_PIXELS = None

CANDIDATES = (
    replace(DEFAULT, gap_radius=12, min_length=40, min_run=5, inherit_tiles=1),
    replace(DEFAULT, gap_radius=18, min_length=48, min_run=5, inherit_tiles=1),
    replace(DEFAULT, gap_radius=18, min_length=48, min_run=5, inherit_tiles=2),
    replace(DEFAULT, gap_radius=28, min_length=48, min_run=6, inherit_tiles=2),
    replace(DEFAULT, gap_radius=18, min_length=80, min_run=5, inherit_tiles=3),
    replace(DEFAULT, gap_radius=36, min_length=64, min_run=5, inherit_tiles=1),
)


def main() -> None:
    cv2.setNumThreads(4)
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("diag", "all"):
        run_diag()
    if mode in ("fit", "all"):
        choose_on_fit()
    if mode in ("dev", "all"):
        score_development(load_settings())


def run_diag() -> None:
    _assert_family_filter()
    _assert_grating()
    _print_exemplars()
    _preview_two_sheets()


def choose_on_fit() -> None:
    rows = _fit_sample()
    prepared = [item for item in (_prepare_fit(row) for row in rows) if float(item["positive"].mean()) < 0.85]
    if len(prepared) < 3:
        raise RuntimeError("fit sample has too few queries with reviewed negatives")
    kept = [item["row"]["id"] for item in prepared]
    print(f"fit sample {len(kept)}", kept, flush=True)
    table = [_fit_candidate(settings, prepared) for settings in CANDIDATES]
    winner = _select_candidate(table)
    payload = {"selected": winner, "candidates": table, "queries": kept}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "fit-selection.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(winner, indent=2), flush=True)


def score_development(settings: Settings) -> None:
    clear_cache()
    scores = _development_scores(settings)
    hgb = devharness.cached_scores(devharness.HGB_CACHE)
    variants = _fuse(scores, hgb)
    RESULTS.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for name, maps in variants.items():
        result = devharness.report(name, maps, RESULTS / f"{name}.json")
        summaries[name] = result["selections"]
        print(name, json.dumps(result["selections"]), flush=True)
    _write_findings(settings, summaries)
    clear_cache()


def _fit_note() -> str:
    path = RESULTS / "fit-selection.json"
    if not path.exists():
        return "No fit-selection.json was found; development used the default settings."
    payload = json.loads(path.read_text(encoding="utf-8"))
    chosen = payload["selected"]
    same = len({row["precision_at_recall"] for row in payload["candidates"]}) == 1
    detail = "Every fit candidate had the same precision at recall 0.98." if same else "Fit candidates differed."
    return (
        f"Fit queries: {payload['queries']}. {detail} "
        f"Selected precision {chosen['precision_at_recall']} at recall {chosen['recall_at_selection']} "
        f"(threshold {chosen['threshold']}). Best macro IoU on that sample was {chosen['best_macro_iou']} "
        f"at recall {chosen['best_macro_iou_recall']}."
    )


def load_settings() -> Settings:
    path = RESULTS / "fit-selection.json"
    if not path.exists():
        return DEFAULT
    raw = json.loads(path.read_text(encoding="utf-8"))["selected"]["settings"]
    return Settings(**raw)


def _development_scores(settings: Settings) -> dict[str, dict[str, np.ndarray]]:
    grouped: dict[str, list] = {}
    for query in devharness.development_queries():
        grouped.setdefault(query.image, []).append(query)
    scores: dict[str, dict[str, np.ndarray]] = {}
    preview_budget = 3
    for image, queries in grouped.items():
        started = time.perf_counter()
        rgb = devharness.load_rgb(image)
        if preview_budget > 0:
            _save_face_preview(rgb, settings, PREVIEWS / f"faces-{4 - preview_budget}.png")
            preview_budget -= 1
        for query in queries:
            request = _request(query)
            sampled = variant_scores(request, rgb, query.y, query.x, settings)
            scores[query.id] = sampled
        print(f"dev {image} queries={len(queries)} sec={time.perf_counter() - started:.1f}", flush=True)
    return scores


def _fuse(
    scores: dict[str, dict[str, np.ndarray]], hgb: dict[str, np.ndarray]
) -> dict[str, dict[str, np.ndarray]]:
    names = ("stats", "ncc", "stats_times_ncc")
    variants = {name: {} for name in names}
    variants.update({
        "stats_times_hgb": {},
        "max_stats_hgb": {},
        "ncc_times_hgb": {},
        "max_ncc_hgb": {},
        "combined_times_hgb": {},
        "max_combined_hgb": {},
    })
    for query_id, maps in scores.items():
        _store_fusions(variants, query_id, maps, hgb[query_id])
    return variants


def _store_fusions(
    variants: dict[str, dict[str, np.ndarray]],
    query_id: str,
    maps: dict[str, np.ndarray],
    hgb: np.ndarray,
) -> None:
    variants["stats"][query_id] = maps["stats"]
    variants["ncc"][query_id] = maps["ncc"]
    variants["stats_times_ncc"][query_id] = maps["stats_times_ncc"]
    variants["stats_times_hgb"][query_id] = maps["stats"] * hgb
    variants["max_stats_hgb"][query_id] = np.maximum(maps["stats"], hgb)
    variants["ncc_times_hgb"][query_id] = maps["ncc"] * hgb
    variants["max_ncc_hgb"][query_id] = np.maximum(maps["ncc"], hgb)
    variants["combined_times_hgb"][query_id] = maps["stats_times_ncc"] * hgb
    variants["max_combined_hgb"][query_id] = np.maximum(maps["stats_times_ncc"], hgb)


def _save_face_preview(rgb: np.ndarray, settings: Settings, path: Path) -> None:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = build_faces(gray, settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_colorize(faces[::4, ::4])).save(path)
    print(f"preview {path.name} faces={int(faces.max())} sec_done", flush=True)


def _colorize(labels: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(0)
    palette = rng.integers(32, 230, size=(int(labels.max()) + 1, 3), dtype=np.uint8)
    palette[0] = 255
    return palette[labels]


def _request(query: object) -> dict:
    return {
        "image": query.image,
        "width": query.width,
        "height": query.height,
        "query_box": query.query_box,
        "context_boxes": query.context_boxes,
    }


def _fit_sample() -> list[dict]:
    rows = [
        row for row in devharness.fit_rows(real_only=True)
        if _has_negative_domain(row)
    ]
    rows.sort(key=lambda row: (row["width"] * row["height"], row["document_id"], row["id"]))
    unique = _one_per_document(rows)
    if len(unique) <= 6:
        return unique
    indexes = np.linspace(0, len(unique) - 1, 6).astype(int)
    return [unique[int(index)] for index in indexes]


def _one_per_document(rows: list[dict]) -> list[dict]:
    chosen = []
    seen: set[str] = set()
    for row in rows:
        if row["document_id"] in seen:
            continue
        seen.add(row["document_id"])
        chosen.append(row)
    return chosen


def _fit_data_root() -> Path:
    # dataset/assets is a symlink. evaluate.data_path rejects the resolved target
    # unless the root is the directory that actually contains assets.
    return (devharness.DATA_ROOT / "assets").resolve().parent


def _has_negative_domain(row: dict) -> bool:
    labels = row.get("labels") or {}
    keys = ("negative_mask", "negative_boxes", "known_mask", "known_boxes", "blank_mask", "blank_boxes")
    return any(key in labels for key in keys)


def _prepare_fit(row: dict) -> dict:
    started = time.perf_counter()
    rgb = devharness.load_rgb(row["image"])
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    grid = measure_sheet(rgb, gray, DEFAULT)
    positive, known, _blank, _meta = reviewed_domains(row, _fit_data_root())
    ys, xs = np.nonzero(known)
    print(f"fit prep {row['id']} known={ys.size} sec={time.perf_counter() - started:.1f}", flush=True)
    return {
        "row": row,
        "rgb": rgb,
        "gray": gray,
        "grid": grid,
        "y": ys,
        "x": xs,
        "positive": positive[ys, xs],
    }


def _fit_candidate(settings: Settings, prepared: list[dict]) -> dict:
    started = time.perf_counter()
    parts = [_fit_part(settings, item) for item in prepared]
    summary = _sweep_parts(parts)
    summary["settings"] = asdict(settings)
    summary["seconds"] = round(time.perf_counter() - started, 2)
    print("fit", summary["precision_at_recall"], "recall", summary["recall_at_selection"], "sec", summary["seconds"], flush=True)
    return summary


def _fit_part(settings: Settings, item: dict) -> tuple[str, np.ndarray, np.ndarray]:
    faces = build_faces(item["gray"], settings)
    exemplar = find_exemplar(item["rgb"], item["gray"], _fit_boxes(item["row"]), settings)
    maps = score_maps(exemplar, item["grid"], faces, item["gray"], settings)
    sampled = sample_maps(maps, item["grid"], faces, item["y"], item["x"])
    return item["row"]["document_id"], sampled["stats"], item["positive"]


def _fit_boxes(row: dict) -> tuple[tuple[int, int, int, int], ...]:
    query = tuple(int(v) for v in row["query_box"])
    context = tuple(tuple(int(v) for v in box) for box in row.get("context_boxes") or ())
    return (query, *context)


def _sweep_parts(parts: list[tuple[str, np.ndarray, np.ndarray]]) -> dict:
    points = [_threshold_point(parts, float(threshold)) for threshold in np.linspace(0.0, 1.0, 101)]
    eligible = [point for point in points if point["recall"] >= 0.98]
    pool = eligible or points
    chosen = max(pool, key=lambda point: (point["precision"], point["macro_iou"]))
    best_iou = max(points, key=lambda point: point["macro_iou"])
    return {
        "precision_at_recall": chosen["precision"],
        "recall_at_selection": chosen["recall"],
        "macro_iou_at_selection": chosen["macro_iou"],
        "threshold": chosen["threshold"],
        "best_macro_iou": best_iou["macro_iou"],
        "best_macro_iou_recall": best_iou["recall"],
        "met_recall_gate": bool(eligible),
    }


def _threshold_point(parts: list[tuple[str, np.ndarray, np.ndarray]], threshold: float) -> dict:
    totals = {"tp": 0, "fp": 0, "fn": 0}
    per_document: dict[str, list[float]] = {}
    for document, scores, positive in parts:
        _accumulate(totals, per_document, document, scores >= threshold, positive)
    precision, recall = _rates(totals)
    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "macro_iou": _macro(per_document),
    }


def _accumulate(
    totals: dict[str, int],
    per_document: dict[str, list[float]],
    document: str,
    decision: np.ndarray,
    positive: np.ndarray,
) -> None:
    tp = int(np.count_nonzero(decision & positive))
    fp = int(np.count_nonzero(decision & ~positive))
    fn = int(np.count_nonzero(~decision & positive))
    totals["tp"] += tp
    totals["fp"] += fp
    totals["fn"] += fn
    if tp + fp + fn:
        per_document.setdefault(document, []).append(tp / (tp + fp + fn))


def _rates(totals: dict[str, int]) -> tuple[float, float]:
    tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return precision, recall


def _macro(per_document: dict[str, list[float]]) -> float:
    if not per_document:
        return 0.0
    means = [sum(values) / len(values) for values in per_document.values()]
    return sum(means) / len(means)


def _select_candidate(table: list[dict]) -> dict:
    gated = [row for row in table if row["met_recall_gate"]]
    pool = gated or table
    return max(pool, key=lambda row: (row["precision_at_recall"], row["macro_iou_at_selection"]))


def _write_findings(settings: Settings, summaries: dict) -> None:
    lines = [
        "Face matcher findings",
        "",
        "Development numbers below come from devharness.report on this run.",
        "Geometry constants were chosen on the fit sample in results/fit-selection.json.",
        "They were not changed after seeing development scores.",
        _fit_note(),
        "",
        "Settings",
        json.dumps(asdict(settings), indent=2),
        "",
        "Large faces",
        (
            "A face whose area is at least large_face_area keeps a separate score on each "
            f"{settings.tile}px tile (stride {settings.tile_stride}). Tiles with no repeated "
            f"period inherit the nearest measured tile of the same face within {settings.inherit_tiles} "
            "tiles, so paper and overlaid text inside a hatch take that hatch's score. A second "
            "pattern on a leaked face keeps its own tile score. Smaller faces are flattened to the "
            "median score of their measured tiles. A face with no periodic or solid tile stays at 0."
        ),
        "",
        "Barriers",
        (
            "Long dark strokes come from the OpenCV line-segment detector. Segments shorter than "
            f"min_length or lighter than gray {settings.black_level} are dropped. Thin segments that "
            f"sit in a regular parallel family of at least min_run={settings.min_run} lines, spacing "
            f"{settings.spacing_min:.0f}-{settings.spacing_max:.0f}px, coefficient of variation "
            f"<= {settings.spacing_cv}, are treated as hatch strokes and dropped. Thick segments "
            f"(distance-transform radius >= {settings.lsd_thick_dist}) are kept. Connected components "
            f"of dark pixels whose long side is at least min_length and whose area/length is between "
            f"{settings.component_min_thickness} and {settings.component_max_thickness} are added. "
            f"An elliptical closing of radius {settings.gap_radius} then bridges gaps up to about "
            "two radii. Faces are the 4-connected components of the complement; barrier pixels take "
            "the nearest face."
        ),
        "",
        "Exemplar",
        (
            "Inside the query crop, every subwindow of side 24, 32, 48, 64, or 96 is ranked by the "
            "normalized autocorrelation peak at a period of at least min_period. Peaks that are ridges "
            "(a single line or an edge, high at every lag along one ray) are rejected. Standard "
            "deviation is not used. The winning window supplies orientation, period, thickness, ink "
            "density, and ink hue/saturation. A nearly uniform nonwhite crop with almost no dark "
            "strokes is a solid fill and is matched on color instead of period. Normalized correlation "
            "uses that same window as the template, and the correlation is read on the measured tiles "
            "of each face, including inherited paper tiles."
        ),
        "",
        "Harness selections",
    ]
    for name, selection in summaries.items():
        lines.extend(_selection_lines(name, selection))
    (HERE / "findings.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _selection_lines(name: str, selection: dict) -> list[str]:
    best = selection["best_macro_iou"] or {}
    gated = selection["best_precision_at_recall_ge_0_98"] or {}
    return [
        "",
        name,
        f"  best macro IoU: {best.get('document_macro_iou')} at threshold {best.get('threshold')}",
        (
            f"  best precision at recall>=0.98: {gated.get('global_precision')} "
            f"(recall {gated.get('global_recall')}, macro IoU {gated.get('document_macro_iou')}, "
            f"threshold {gated.get('threshold')})"
        ),
    ]


def _print_exemplars() -> None:
    seen: set[str] = set()
    PREVIEWS.mkdir(parents=True, exist_ok=True)
    for query in devharness.development_queries():
        if query.image in seen:
            continue
        seen.add(query.image)
        _print_one_exemplar(query, len(seen))


def _print_one_exemplar(query: object, index: int) -> None:
    rgb = devharness.load_rgb(query.image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    box = tuple(int(value) for value in query.query_box)
    exemplar = find_exemplar(rgb, gray, (box,), DEFAULT)
    if exemplar is None:
        print(f"exemplar {index} {query.id} none", flush=True)
        return
    kind = {PERIODIC: "periodic", SOLID: "solid"}.get(exemplar.kind, "plain")
    print(
        f"exemplar {index} {query.id} {kind} strength={exemplar.strength:.3f} "
        f"period={exemplar.period:.1f} orient={exemplar.orient:.1f} "
        f"thick={exemplar.thickness:.2f} dens={exemplar.density:.3f} "
        f"hue={exemplar.hue:.1f} sat={exemplar.sat:.2f}",
        flush=True,
    )
    _save_template(exemplar.template, PREVIEWS / f"exemplar-{index}.png")


def _save_template(template: np.ndarray, path: Path) -> None:
    scale = max(1, 128 // max(template.shape))
    image = Image.fromarray(template).resize(
        (template.shape[1] * scale, template.shape[0] * scale), Image.Resampling.NEAREST
    )
    image.save(path)


def _preview_two_sheets() -> None:
    queries = devharness.development_queries()
    images = []
    for query in queries:
        if query.image not in images:
            images.append(query.image)
        if len(images) == 2:
            break
    for index, image in enumerate(images, start=1):
        started = time.perf_counter()
        rgb = devharness.load_rgb(image)
        _save_face_preview(rgb, DEFAULT, PREVIEWS / f"diag-faces-{index}.png")
        print(f"diag sheet {index} sec={time.perf_counter() - started:.1f}", flush=True)


def _assert_family_filter() -> None:
    hatch = np.array([[0, 10 + 12 * i, 200, 10 + 12 * i] for i in range(8)], dtype=np.float32)
    pair = np.array([[0, 10, 220, 10], [0, 18, 220, 18]], dtype=np.float32)
    if not _family_drop(hatch, DEFAULT).all():
        raise RuntimeError("parallel hatch family was kept")
    if _family_drop(pair, DEFAULT).any():
        raise RuntimeError("wall pair was dropped")


def _assert_grating() -> None:
    yy, xx = np.mgrid[0:64, 0:64]
    gray = np.clip(128 + 100 * np.sin(2 * np.pi * xx / 10.0), 0, 255).astype(np.uint8)
    rgb = np.dstack([gray, gray, gray])
    measured = _measure(rgb, gray, DEFAULT)
    period = measured.peaks[0][1] if measured.peaks else 0.0
    strength = measured.peaks[0][0] if measured.peaks else 0.0
    orient = measured.peaks[0][2] if measured.peaks else 0.0
    if measured.kind != PERIODIC or not (8.0 <= period <= 12.0):
        raise RuntimeError(
            f"grating not recovered: kind={measured.kind} period={period} strength={strength} orient={orient}"
        )
    print(f"grating period={period:.2f} orient={orient:.1f} strength={strength:.3f}", flush=True)


if __name__ == "__main__":
    main()
