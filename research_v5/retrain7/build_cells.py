"""Fit-split stage-2 cells from g2-cleaned stage-1 grids.

Sampling, folds, and structure channels match research_v5/structure/cells.py.
The 75 stacker channels are rebuilt from the cross-fitted stage-1 model and
the g2 descriptor. Base sampling is 2000 cells per class. The double sample
uses the same function with 4000.
"""
from __future__ import annotations

import gc
import time

import runtime

runtime.activate()

import cv2
import numpy as np

import hatch_texture
from banks import _load_bank, _save_bank, bank_path
from constants import CELLS_PER_CLASS, DATA_ROOT, GRID_STRIDE
from dataset import _gather, _population_weights, _sample_cells
from drawing import _load_rgb, _mismatches
from evaluate import reviewed_domains
from features import assemble, feature_names, fraction_maps
from folds import indexed_training_order
from guards import descriptor_for
from inference import read_drawing
from inference_texture import predict_probability_grid
from runtime import (
    BANK_CACHE,
    CELL_CACHE,
    CELL_VERSION,
    CLEANER,
    STACKER_CACHE,
    STACKER_VERSION,
    STRUCTURE_CELLS,
    atomic_savez,
    current_gb,
    guard_memory,
    log,
    require,
)
from stage1 import load_fold_payloads
from structure_features import (
    component_maps,
    feature_names as structure_names,
    query_reference,
    structure_volume,
)
from texture_learning import grid_centers

BASE_PER_CLASS = CELLS_PER_CLASS
DOUBLE_PER_CLASS = CELLS_PER_CLASS * 2


def build_all(rows: list[dict], document_fold: dict[str, int], real_ids: set[str], limit: int = 0) -> dict:
    _require_widths()
    require(BASE_PER_CLASS == 2000, f"cells per class {BASE_PER_CLASS}")
    payloads = load_fold_payloads()
    grouped = _groups(rows)
    images = _by_area(grouped)
    if limit:
        images = images[:limit]
    for image in images:
        _build_image(image, grouped[image], document_fold, real_ids, payloads)
        guard_memory()
    if limit:
        log(f"smoke built {len(images)} images")
        return {"smoke": len(images)}
    summary = verify_all(rows, document_fold, real_ids)
    log(
        f"cells queries={summary['queries']} base={summary['cells_base']} "
        f"double={summary['cells_double']} changed={summary['queries_cell_score_changed']}"
    )
    return summary


def verify_all(rows: list[dict], document_fold: dict[str, int], real_ids: set[str]) -> dict:
    totals = _empty_totals()
    for row in rows:
        _verify_one(row, document_fold, real_ids, totals)
    totals["queries"] = len(rows)
    return totals


def stacker_path(query_id: str):
    return STACKER_CACHE / f"{_safe(query_id)}.npz"


def cell_path(query_id: str):
    return CELL_CACHE / f"{_safe(query_id)}.npz"


def _build_image(image, group, document_fold, real_ids, payloads) -> None:
    document = _one_document(group)
    fold = int(document_fold[document])
    model = _fold_model(payloads, fold, document)
    pending_stacker = [pair for pair in group if not _stacker_ready(pair[1]["id"])]
    if pending_stacker:
        _stacker_pass(image, pending_stacker, model)
    pending_cells = [pair for pair in group if not _cell_ready(pair[1]["id"])]
    if pending_cells:
        _structure_pass(image, pending_cells, document, fold, real_ids)
    log(f"cells image {image} stacker={len(pending_stacker)} joined={len(pending_cells)} rss_gb={current_gb():.2f}")


def _stacker_pass(image: str, pending, model) -> None:
    started = time.perf_counter()
    spec = _spec(pending[0][1])
    boxes = [row["query_box"] for _index, row in pending]
    prepared = _prepare(spec, boxes)
    try:
        _save_stacker_queries(pending, prepared, model)
    finally:
        prepared.clear()
        gc.collect()
    log(f"stacker {image} queries={len(pending)} seconds={time.perf_counter() - started:.1f}")


def _save_stacker_queries(pending, prepared, model) -> None:
    probabilities = _probabilities(prepared, pending, model)
    for index, (global_index, row) in enumerate(pending):
        _save_one_stacker(global_index, row, prepared, probabilities[index], prepared["mismatches"][index])
        probabilities[index] = None


def _save_one_stacker(global_index: int, row: dict, prepared, probability, mismatch) -> None:
    volume = assemble(
        probability,
        _box(row["query_box"]),
        prepared["height"],
        prepared["width"],
        prepared["ink_maps"],
        prepared["chroma_maps"],
        mismatch,
    )
    try:
        base, double = _both_samples(row, global_index)
        features, labels = _gather(volume, base[0], base[1])
        features2, labels2 = _gather(volume, double[0], double[1])
    finally:
        del volume
    _require_domains(base, double, row["id"])
    _write_stacker(row, features, labels, features2, labels2, base[2], base[3])


def _structure_pass(image, pending, document: str, fold: int, real_ids: set[str]) -> None:
    started = time.perf_counter()
    gray, bank, area_map, side_map, ink = _structure_inputs(image)
    try:
        for global_index, row in pending:
            _join_one(global_index, row, gray, bank, area_map, side_map, ink, document, fold, real_ids)
    finally:
        del gray, bank, area_map, side_map, ink
        gc.collect()
    log(f"structure {image} queries={len(pending)} seconds={time.perf_counter() - started:.1f}")


def _structure_inputs(image: str):
    rgb = _load_rgb(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    bank = _read_bank(image, rgb)
    area_map, side_map, ink = component_maps(rgb)
    del rgb
    gc.collect()
    return gray, bank, area_map, side_map, ink


def _join_one(global_index, row, gray, bank, area_map, side_map, ink, document, fold, real_ids) -> None:
    reference = query_reference(gray, _box(row["query_box"]), area_map, side_map, ink)
    volume = structure_volume(bank, reference)
    try:
        base, double = _both_samples(row, global_index)
        struct, labels = _gather(volume, base[0], base[1])
        struct2, labels2 = _gather(volume, double[0], double[1])
    finally:
        del volume
    stacker = _read_stacker(row["id"])
    _require_labels(stacker, labels, labels2, row["id"])
    _assert_structure_cache(row["id"], labels, struct)
    features = _concat(stacker["features"], struct)
    features2 = _concat(stacker["features2"], struct2)
    _write_cell(row, features, labels, features2, labels2, stacker, document, fold, real_ids)


def _prepare(spec: dict, boxes: list) -> dict:
    gray = read_drawing(DATA_ROOT, spec)
    rgb = _load_rgb(spec["image"])
    features = hatch_texture.texture_features(gray, stride=GRID_STRIDE)
    signatures = [_signature(rgb, box) for box in boxes]
    ink, chroma, hue = _ink_chroma(rgb)
    height, width = int(gray.shape[0]), int(gray.shape[1])
    del rgb
    gc.collect()
    centers_y, centers_x = grid_centers(height, width, GRID_STRIDE)
    mismatches = _mismatches(spec["image"], signatures, ink, chroma, hue, centers_y, centers_x)
    prepared = {
        "gray": gray,
        "features": features,
        "ink_maps": fraction_maps(ink, centers_y, centers_x),
        "chroma_maps": fraction_maps(chroma, centers_y, centers_x),
        "height": height,
        "width": width,
        "mismatches": mismatches,
    }
    del ink, chroma, hue, signatures
    return prepared


def _probabilities(prepared, pending, model) -> list:
    grids = []
    for _index, row in pending:
        descriptor = descriptor_for(CLEANER, prepared["gray"], _box(row["query_box"]))
        grids.append(predict_probability_grid(model, prepared["features"], descriptor))
    del prepared["gray"], prepared["features"]
    gc.collect()
    return grids


def _both_samples(row: dict, global_index: int):
    positive, known, _blank, _meta = reviewed_domains(row, DATA_ROOT)
    centers_y, centers_x = grid_centers(int(row["height"]), int(row["width"]), GRID_STRIDE)
    base = _sample_cells(positive, known, centers_y, centers_x, BASE_PER_CLASS, global_index)
    double = _sample_cells(positive, known, centers_y, centers_x, DOUBLE_PER_CLASS, global_index)
    del positive, known
    return base, double


def _read_bank(image: str, rgb: np.ndarray) -> dict:
    cached = _load_bank(bank_path(image), image, rgb.shape)
    if cached is not None:
        return cached
    local = _local_bank(image)
    cached = _load_bank(local, image, rgb.shape)
    if cached is not None:
        return cached
    from structure_features import bank_from_rgb

    bank = bank_from_rgb(rgb)
    _save_bank(local, image, rgb.shape, bank)
    log(f"bank built locally {image}")
    return bank


def _write_stacker(row, features, labels, features2, labels2, n_pos: int, n_neg: int) -> None:
    require(features.shape[1] == len(feature_names()) or len(labels) == 0, f"stacker width {row['id']}")
    atomic_savez(
        stacker_path(row["id"]),
        X=np.asarray(features, dtype=np.float32),
        y=np.asarray(labels, dtype=np.int8),
        X2=np.asarray(features2, dtype=np.float32),
        y2=np.asarray(labels2, dtype=np.int8),
        n_pos_domain=np.int64(n_pos),
        n_neg_domain=np.int64(n_neg),
        cleaner=np.array(CLEANER),
        version=np.int32(STACKER_VERSION),
    )


def _write_cell(row, features, labels, features2, labels2, stacker, document, fold, real_ids) -> None:
    real = int(row["id"] in real_ids)
    atomic_savez(
        cell_path(row["id"]),
        X=np.asarray(features, dtype=np.float32),
        y=np.asarray(labels, dtype=np.int8),
        X2=np.asarray(features2, dtype=np.float32),
        y2=np.asarray(labels2, dtype=np.int8),
        document=np.array(document),
        fold=np.int8(fold),
        real=np.int8(real),
        n_pos_domain=np.int64(stacker["n_pos"]),
        n_neg_domain=np.int64(stacker["n_neg"]),
        cleaner=np.array(CLEANER),
        version=np.int32(CELL_VERSION),
    )


def _verify_one(row, document_fold, real_ids, totals) -> None:
    require(_cell_ready(row["id"]), f"missing cell {row['id']}")
    stored = _read_cell(row["id"])
    _require_cell_row(stored, row, document_fold, real_ids)
    gap = _stacker_gap(row["id"], stored["features"])
    struct_gap = _structure_gap(row["id"], stored["labels"], stored["features"])
    _add_totals(totals, stored, gap, struct_gap)


def _stacker_gap(query_id: str, features: np.ndarray) -> dict:
    cached = _structure_matrix(query_id)
    if len(features) == 0:
        return {"score": 0.0, "stacker": 0.0}
    width = len(feature_names())
    score = _max_gap(features[:, 0], cached[:, 0])
    stacker = _max_gap(features[:, :width], cached[:, :width])
    return {"score": score, "stacker": stacker}


def _structure_gap(query_id: str, labels: np.ndarray, features: np.ndarray) -> float:
    cached = _structure_matrix(query_id)
    cached_y = _structure_labels(query_id)
    require(np.array_equal(labels, cached_y), f"sampling drift {query_id}")
    width = len(feature_names())
    if len(labels) == 0:
        return 0.0
    return _max_gap(features[:, width:], cached[:, width:])


def _assert_structure_cache(query_id: str, labels, struct) -> None:
    gap = _structure_only_gap(query_id, labels, struct)
    require(gap <= 1e-4, f"structure gap {query_id} {gap}")


def _structure_only_gap(query_id: str, labels, struct) -> float:
    cached_y = _structure_labels(query_id)
    require(np.array_equal(np.asarray(labels), cached_y), f"sampling drift {query_id}")
    cached = _structure_matrix(query_id)
    width = len(feature_names())
    if len(cached_y) == 0:
        return 0.0
    return _max_gap(struct, cached[:, width:])


def _read_stacker(query_id: str) -> dict:
    require(_stacker_ready(query_id), f"missing stacker {query_id}")
    with np.load(stacker_path(query_id)) as stored:
        return {
            "features": np.array(stored["X"], dtype=np.float32),
            "labels": np.array(stored["y"], dtype=np.int8),
            "features2": np.array(stored["X2"], dtype=np.float32),
            "labels2": np.array(stored["y2"], dtype=np.int8),
            "n_pos": int(stored["n_pos_domain"]),
            "n_neg": int(stored["n_neg_domain"]),
        }


def _read_cell(query_id: str) -> dict:
    with np.load(cell_path(query_id)) as stored:
        return {
            "features": np.array(stored["X"], dtype=np.float32),
            "labels": np.array(stored["y"], dtype=np.int8),
            "features2": np.array(stored["X2"], dtype=np.float32),
            "labels2": np.array(stored["y2"], dtype=np.int8),
            "document": str(stored["document"].item()),
            "fold": int(stored["fold"]),
            "real": int(stored["real"]),
            "n_pos": int(stored["n_pos_domain"]),
            "n_neg": int(stored["n_neg_domain"]),
        }


def _groups(rows: list[dict]) -> dict[str, list]:
    grouped: dict[str, list] = {}
    for index, row in indexed_training_order(rows):
        grouped.setdefault(row["image"], []).append((index, row))
    return grouped


def _by_area(grouped: dict) -> list[str]:
    def area(image: str) -> int:
        row = grouped[image][0][1]
        return int(row["width"]) * int(row["height"])

    return sorted(grouped, key=area)


def _fold_model(payloads, fold: int, document: str):
    payload = payloads[fold]
    require(int(payload["held_out_fold"]) == fold, f"fold model {fold}")
    require(document not in set(payload["training_documents"]), f"stage1 saw {document}")
    return payload["model"]


def _one_document(group) -> str:
    document = group[0][1]["document_id"]
    for _index, row in group:
        require(row["document_id"] == document, "image spans documents")
    return document


def _require_domains(base, double, query_id: str) -> None:
    require(base[2] == double[2] and base[3] == double[3], f"domain counts {query_id}")


def _require_labels(stacker, labels, labels2, query_id: str) -> None:
    require(np.array_equal(labels, stacker["labels"]), f"base label drift {query_id}")
    require(np.array_equal(labels2, stacker["labels2"]), f"double label drift {query_id}")


def _require_cell_row(stored, row, document_fold, real_ids) -> None:
    require(stored["document"] == row["document_id"], f"document {row['id']}")
    require(stored["fold"] == int(document_fold[row["document_id"]]), f"fold {row['id']}")
    require(stored["real"] == int(row["id"] in real_ids), f"real flag {row['id']}")
    require(stored["features"].shape[1] == _joined_width() or len(stored["labels"]) == 0, f"width {row['id']}")
    require((row["kind"] == "real") == (row["id"] in real_ids), f"kind {row['id']}")


def _stacker_ready(query_id: str) -> bool:
    return _ready(stacker_path(query_id), STACKER_VERSION, len(feature_names()))


def _cell_ready(query_id: str) -> bool:
    return _ready(cell_path(query_id), CELL_VERSION, _joined_width())


def _ready(path, version: int, width: int) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path) as stored:
            return _stored_ok(stored, version, width)
    except (OSError, ValueError, KeyError):
        return False


def _stored_ok(stored, version: int, width: int) -> bool:
    if int(stored["version"]) != version or str(stored["cleaner"].item()) != CLEANER:
        return False
    return _width_ok(stored["X"], width) and _width_ok(stored["X2"], width)


def _width_ok(matrix, width: int) -> bool:
    return matrix.ndim == 2 and int(matrix.shape[1]) == width


def _concat(base, struct) -> np.ndarray:
    if len(base) == 0:
        return np.zeros((0, base.shape[1] + struct.shape[1]), np.float32)
    return np.concatenate([base, struct], axis=1)


def _structure_matrix(query_id: str) -> np.ndarray:
    with np.load(_structure_cell(query_id)) as stored:
        return np.array(stored["X"], dtype=np.float32)


def _structure_labels(query_id: str) -> np.ndarray:
    with np.load(_structure_cell(query_id)) as stored:
        return np.array(stored["y"], dtype=np.int8)


def _structure_cell(query_id: str):
    path = STRUCTURE_CELLS / f"{_safe(query_id)}.npz"
    require(path.is_file(), f"missing structure cell {query_id}")
    return path


def _max_gap(left, right) -> float:
    require(left.shape == right.shape, "gap shape")
    if left.size == 0:
        return 0.0
    return float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))


def _local_bank(image: str):
    return BANK_CACHE / bank_path(image).name


def _spec(row: dict) -> dict:
    return {key: row[key] for key in ("id", "image", "width", "height")}


def _box(box) -> tuple[int, int, int, int]:
    return tuple(int(value) for value in box)


def _safe(query_id: str) -> str:
    require(bool(query_id) and "/" not in query_id and "\\" not in query_id, "unsafe id")
    require(not query_id.startswith("."), "unsafe id")
    return query_id


def _signature(rgb, box):
    from color.color_features import extract_robust_query_signature

    return extract_robust_query_signature(rgb, _box(box))


def _ink_chroma(rgb):
    from color.color_features import extract_ink_and_chroma

    return extract_ink_and_chroma(rgb)


def _joined_width() -> int:
    return len(feature_names()) + len(structure_names())


def _require_widths() -> None:
    require(len(feature_names()) == 75, "stacker width")
    require(feature_names()[0] == "cell_score", "cell_score column")
    require(len(structure_names()) == len(set(structure_names())), "structure names")


def _empty_totals() -> dict:
    return {
        "queries": 0,
        "cells_base": 0,
        "cells_double": 0,
        "positive_base": 0,
        "negative_base": 0,
        "positive_double": 0,
        "negative_double": 0,
        "real_queries": 0,
        "queries_cell_score_changed": 0,
        "max_cell_score_abs_gap": 0.0,
        "max_stacker_abs_gap": 0.0,
        "max_structure_abs_gap": 0.0,
    }


def _add_totals(totals, stored, gap, struct_gap: float) -> None:
    labels = stored["labels"]
    labels2 = stored["labels2"]
    totals["cells_base"] += int(len(labels))
    totals["cells_double"] += int(len(labels2))
    totals["positive_base"] += int(np.count_nonzero(labels == 1))
    totals["negative_base"] += int(np.count_nonzero(labels == 0))
    totals["positive_double"] += int(np.count_nonzero(labels2 == 1))
    totals["negative_double"] += int(np.count_nonzero(labels2 == 0))
    totals["real_queries"] += int(stored["real"] == 1)
    totals["queries_cell_score_changed"] += int(gap["score"] > 1e-5)
    totals["max_cell_score_abs_gap"] = max(totals["max_cell_score_abs_gap"], gap["score"])
    totals["max_stacker_abs_gap"] = max(totals["max_stacker_abs_gap"], gap["stacker"])
    totals["max_structure_abs_gap"] = max(totals["max_structure_abs_gap"], struct_gap)
    require(struct_gap <= 1e-4, "structure verify gap")


def population(labels, n_pos: int, n_neg: int) -> np.ndarray:
    return _population_weights(labels, n_pos, n_neg)
