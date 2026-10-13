"""Fit-split crops. Half the windows sit on a positive pixel, half on a known negative."""

from __future__ import annotations

import gc
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate import data_path, load_manifest, reviewed_domains  # noqa: E402

from .constants import KNOWN_FRACTION, NATIVE_CROP, POINT_CAP
from .geom import apply_dihedral, area_half_rgb, cell_fractions, pack_hwc, square_window, tile_repeat

Image.MAX_IMAGE_PIXELS = None


@dataclass
class QueryPoints:
    row: dict
    box: tuple[int, int, int, int]
    pos_yx: np.ndarray
    neg_yx: np.ndarray


class FitSampler:
    """One drawing at a time. Each document is visited the same number of times."""

    def __init__(self, manifest: Path, data_root: Path, seed: int, half_crop: int = NATIVE_CROP // 2, per_visit: int = 4) -> None:
        self.rows = load_training_rows(manifest)
        self.groups = _group(self.rows)
        self.data_root = Path(data_root)
        self.rng = np.random.default_rng(seed)
        self.native = half_crop * 2
        self._cycle = _cycle(tuple(self.groups), per_visit, self.rng)
        self.doc_id: str | None = None
        self.image_name: str | None = None
        self.rgb: np.ndarray | None = None
        self.queries: list[QueryPoints] = []
        self.mask_id: str | None = None
        self.positive: np.ndarray | None = None
        self.known: np.ndarray | None = None

    def document_ids(self) -> tuple[str, ...]:
        return tuple(self.groups)

    def sample_batch(self, batch_size: int):
        doc_id = next(self._cycle)
        self._activate(doc_id)
        samples = [_sample_one(self, index % 2 == 0) for index in range(batch_size)]
        if any(item is None for item in samples):
            return None
        return _collate(samples)


def load_training_rows(path: Path) -> list[dict]:
    text = path.as_posix()
    if "development" in text or path.name.startswith("val"):
        raise SystemExit("training manifest must be the fit split or the synthetic manifest")
    return load_manifest(path)


def one_pass(doc_ids: tuple[str, ...], per_visit: int, rng: np.random.Generator):
    ids = list(doc_ids)
    rng.shuffle(ids)
    for doc_id in ids:
        for _ in range(per_visit):
            yield doc_id


def _cycle(doc_ids: tuple[str, ...], per_visit: int, rng: np.random.Generator):
    while True:
        yield from one_pass(doc_ids, per_visit, rng)


def _sample_one(sampler: FitSampler, want_positive: bool):
    for _ in range(8):
        packed = _try_once(sampler, want_positive)
        if packed is not None:
            return packed
    return None


def _try_once(sampler: FitSampler, want_positive: bool):
    query = _pick_query(sampler.queries, want_positive, sampler.rng)
    if query is None:
        return None
    _ensure_image(sampler, query)
    _load_masks(sampler, query)
    center = _center(query, want_positive, sampler.rng)
    if center is None or sampler.rgb is None or sampler.known is None or sampler.positive is None:
        return None
    return _crop_augment(sampler, query, int(center[0]), int(center[1]))


def _crop_augment(sampler: FitSampler, query: QueryPoints, cy: int, cx: int):
    window, known, positive = square_window(sampler.rgb, sampler.known, sampler.positive, cy, cx, sampler.native)
    query_rgb = _query_rgb(sampler.rgb, query.box)
    turns, flip_h, flip_v = _augment_choice(sampler.rng)
    window, known, positive, query_rgb = apply_dihedral([window, known, positive, query_rgb], turns, flip_h, flip_v)
    known_frac, pos_frac = cell_fractions(known, positive)
    if not bool(np.any(known_frac >= KNOWN_FRACTION)):
        return None
    drawing = pack_hwc(area_half_rgb(window))
    query_pack = pack_hwc(tile_repeat(area_half_rgb(query_rgb)))
    return {"drawing": drawing, "query": query_pack, "known": known_frac, "positive": pos_frac}


def _activate(self: FitSampler, doc_id: str) -> None:
    if self.doc_id == doc_id:
        return
    self.rgb = None
    self.image_name = None
    self.queries = []
    self._drop_masks()
    gc.collect()
    rows = self.groups[doc_id]
    self.doc_id = doc_id
    print(f"document {doc_id} queries {len(rows)}", flush=True)
    self.queries = [_points_for(self, row) for row in rows]


def _points_for(sampler: FitSampler, row: dict) -> QueryPoints:
    positive, known, _, _ = reviewed_domains(row, sampler.data_root)
    pos_yx = _take_points(positive, sampler.rng)
    neg_yx = _take_points(known & ~positive, sampler.rng)
    del positive, known
    return QueryPoints(row=row, box=tuple(int(v) for v in row["query_box"]), pos_yx=pos_yx, neg_yx=neg_yx)


def _load_masks(sampler: FitSampler, query: QueryPoints) -> None:
    if sampler.mask_id == query.row["id"]:
        return
    sampler._drop_masks()
    positive, known, _, _ = reviewed_domains(query.row, sampler.data_root)
    sampler.positive = positive
    sampler.known = known
    sampler.mask_id = query.row["id"]


def _take_points(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    flat = np.flatnonzero(np.asarray(mask))
    if flat.size == 0:
        return np.zeros((0, 2), np.int32)
    if flat.size > POINT_CAP:
        flat = rng.choice(flat, size=POINT_CAP, replace=False)
    ys, xs = np.divmod(flat, mask.shape[1])
    return np.stack((ys, xs), axis=1).astype(np.int32)


def _pick_query(queries: list[QueryPoints], want_positive: bool, rng: np.random.Generator):
    pool = [query for query in queries if _has_class(query, want_positive)]
    if not pool:
        pool = [query for query in queries if _has_class(query, True)]
    if not pool:
        return None
    return pool[int(rng.integers(len(pool)))]


def _has_class(query: QueryPoints, want_positive: bool) -> bool:
    points = query.pos_yx if want_positive else query.neg_yx
    return len(points) > 0


def _center(query: QueryPoints, want_positive: bool, rng: np.random.Generator):
    points = query.pos_yx if want_positive else query.neg_yx
    if len(points) == 0:
        points = query.pos_yx
    if len(points) == 0:
        return None
    return points[int(rng.integers(len(points)))]


def _augment_choice(rng: np.random.Generator):
    turns = int(rng.integers(0, 4))
    return turns, bool(rng.random() < 0.5), bool(rng.random() < 0.5)


def _query_rgb(rgb: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    return np.asarray(rgb[y0:y1, x0:x1])


def _collate(samples: list[dict]) -> dict:
    return {
        "drawing": np.stack([item["drawing"] for item in samples]),
        "query": np.stack([item["query"] for item in samples]),
        "known": np.stack([item["known"] for item in samples]),
        "positive": np.stack([item["positive"] for item in samples]),
    }


def _group(rows: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["document_id"], []).append(row)
    return groups


def _ensure_image(sampler: FitSampler, query: QueryPoints) -> None:
    name = query.row["image"]
    if sampler.image_name == name and sampler.rgb is not None:
        return
    sampler.rgb = _read_rgb(sampler.data_root, name)
    sampler.image_name = name
    height, width = int(sampler.rgb.shape[0]), int(sampler.rgb.shape[1])
    if height != int(query.row["height"]) or width != int(query.row["width"]):
        raise RuntimeError(query.row["id"] + " image shape does not match the manifest")


def _read_rgb(data_root: Path, relative: str) -> np.ndarray:
    with Image.open(data_path(data_root, relative)) as image:
        return np.asarray(image.convert("RGB"))


def _drop_masks(self: FitSampler) -> None:
    self.positive = None
    self.known = None
    self.mask_id = None


FitSampler._activate = _activate  # type: ignore[method-assign]
FitSampler._drop_masks = _drop_masks  # type: ignore[method-assign]
