"""Fit-split crops whose cell labels come only from reviewed pixels."""

from __future__ import annotations

import gc
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_cdt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate import data_path, load_manifest, reviewed_domains  # noqa: E402

from constants import CROP, NEG, POS, QUERY_HALO, STRIDE, UNK
from pixels import pack_channels

Image.MAX_IMAGE_PIXELS = None
cv2.setNumThreads(4)

SAMPLING_NOTES = (
    "Documents are shuffled, and each visit keeps one decoded sheet for several crops. Every document is offered the same number of visits, so a sheet with more queries is not sampled more often.",
    "A crop is kept only when it contains a reviewed positive cell and a reviewed-known non-positive cell. Unknown cells stay out of the loss.",
    "Rotation and scale are shared by the query crop and the target crop. Ink-color gains are shared too, so a colored hatch and a gray neighbor on the same sheet keep their color difference.",
    "Cell labels are the reviewed mask sampled at the center of each stride-8 cell. Scale jitter resamples that center grid with nearest neighbors.",
)


@dataclass(frozen=True)
class PreparedQuery:
    grid: np.ndarray
    pos_yx: np.ndarray
    neg_dist: np.ndarray
    query_box: tuple[int, int, int, int]


def load_fit_rows(path: Path) -> list[dict]:
    if path.name != "fit.json":
        raise SystemExit("training manifest must be named fit.json")
    return load_manifest(path)


def build_label_grid(positive: np.ndarray, known: np.ndarray, stride: int = STRIDE) -> np.ndarray:
    """0 unknown, 1 positive center, 2 reviewed-known non-positive center."""
    pos = positive[stride // 2 :: stride, stride // 2 :: stride]
    kn = known[stride // 2 :: stride, stride // 2 :: stride]
    grid = np.zeros(pos.shape, dtype=np.uint8)
    grid[kn & ~pos] = NEG
    grid[pos] = POS
    return grid


def labels_for_loss(grid: np.ndarray) -> np.ndarray:
    labels = np.full(grid.shape, -1, dtype=np.int64)
    labels[grid == POS] = 1
    labels[grid == NEG] = 0
    return np.ascontiguousarray(labels)


def cell_inside(mask: np.ndarray, stride: int = STRIDE) -> np.ndarray:
    return np.ascontiguousarray(mask[stride // 2 :: stride, stride // 2 :: stride])


class DocumentStore:
    """One decoded drawing at a time, reused for several crops."""

    def __init__(self, rows: list[dict], data_root: Path) -> None:
        self.groups = _group_rows(rows)
        self.data_root = Path(data_root)
        self.doc_id: str | None = None
        self.rgb: np.ndarray | None = None
        self.prepared: PreparedQuery | None = None

    def document_ids(self) -> tuple[str, ...]:
        return tuple(self.groups)

    def sample(self, doc_id: str, rng: np.random.Generator, crop: int = CROP):
        self._activate(doc_id, rng)
        if self.rgb is None or self.prepared is None:
            return None
        return sample_pair(self.rgb, self.prepared, rng, crop)

    def _activate(self, doc_id: str, rng: np.random.Generator) -> None:
        if self.doc_id == doc_id:
            return
        self.rgb = None
        self.prepared = None
        gc.collect()
        picked = _first_usable(self.groups[doc_id], self.data_root, rng)
        self.doc_id = doc_id
        if picked is None:
            print(f"skip {doc_id}: no query with positives and known negatives", flush=True)
            return
        self.prepared, image_name = picked
        self.rgb = _read_image(self.data_root, image_name)


def sample_pair(rgb: np.ndarray, prepared: PreparedQuery, rng: np.random.Generator, crop: int = CROP):
    for _ in range(12):
        batch = _one_attempt(rgb, prepared, rng, crop)
        if batch is not None:
            return batch
    return None


def iter_documents(doc_ids: tuple[str, ...], per_visit: int, rng: np.random.Generator):
    ids = list(doc_ids)
    while True:
        rng.shuffle(ids)
        for doc_id in ids:
            for _ in range(per_visit):
                yield doc_id


def _one_attempt(rgb, prepared, rng, crop):
    scale, turns, gains = _draw_augment(rng)
    target = _target_crop(rgb, prepared, scale, turns, gains, crop, rng)
    if target is None:
        return None
    query = _query_crop(rgb, prepared.query_box, scale, turns, gains)
    if query is None:
        return None
    return {"target": target[0], "labels": target[1], "query": query[0], "query_mask": query[1]}


def _draw_augment(rng: np.random.Generator):
    scale = float(rng.uniform(0.8, 1.25))
    turns = int(rng.integers(0, 4))
    gains = rng.uniform(0.8, 1.25, size=3).astype(np.float32)
    return scale, turns, gains


def _target_crop(rgb, prepared, scale, turns, gains, crop, rng):
    cells = _window_cells(scale, crop)
    if cells > prepared.grid.shape[0] or cells > prepared.grid.shape[1]:
        return None
    origin = _choose_origin(prepared, cells, rng)
    if origin is None:
        return None
    cropped = _slice_window(rgb, prepared.grid, origin[0], origin[1], cells)
    if cropped is None:
        return None
    image, grid = cropped
    image = cv2.resize(image, (crop, crop), interpolation=cv2.INTER_LINEAR)
    grid = cv2.resize(grid, (crop // STRIDE, crop // STRIDE), interpolation=cv2.INTER_NEAREST)
    image, grid = _rotate_pair(image, grid, turns)
    if not _has_both(grid):
        return None
    packed = pack_channels(_apply_gains(image, gains))
    labels = labels_for_loss(np.ascontiguousarray(grid))
    if labels.shape != (crop // STRIDE, crop // STRIDE):
        return None
    return packed, labels


def _query_crop(rgb, box, scale, turns, gains):
    canvas, inside = _query_canvas(rgb, box, QUERY_HALO)
    height = _snap_side(canvas.shape[0], scale)
    width = _snap_side(canvas.shape[1], scale)
    image, mask = _resize_query(canvas, inside, width, height)
    image, mask = _rotate_pair(image, mask, turns)
    image = np.ascontiguousarray(image)
    mask = np.ascontiguousarray(mask)
    packed = pack_channels(_apply_gains(image, gains))
    cells = cell_inside(mask)
    expected = (packed.shape[1] // STRIDE, packed.shape[2] // STRIDE)
    if cells.shape != expected or not bool(cells.any()):
        return None
    return packed, cells


def _window_cells(scale: float, crop: int) -> int:
    side = int(round((crop / scale) / STRIDE)) * STRIDE
    return max(side, STRIDE * 4) // STRIDE


def _snap_side(length: int, scale: float) -> int:
    snapped = int(round(length * scale / STRIDE)) * STRIDE
    return max(snapped, STRIDE)


def _choose_origin(prepared: PreparedQuery, cells: int, rng: np.random.Generator):
    distances = prepared.neg_dist[prepared.pos_yx[:, 0], prepared.pos_yx[:, 1]]
    pool = np.flatnonzero(distances <= (cells - 1))
    if pool.size == 0:
        return None
    for _ in range(8):
        origin = _origin_from_pool(prepared, cells, rng, pool)
        if origin is not None:
            return origin
    return None


def _origin_from_pool(prepared, cells, rng, pool):
    py, px = (int(v) for v in prepared.pos_yx[pool[int(rng.integers(pool.size))]])
    negative = _nearby_negative(prepared.grid, py, px, cells, rng)
    if negative is None:
        return None
    return _cover_both(py, px, negative[0], negative[1], cells, prepared.grid.shape, rng)


def _nearby_negative(grid, py, px, cells, rng):
    y0 = max(0, py - (cells - 1))
    y1 = min(grid.shape[0], py + cells)
    x0 = max(0, px - (cells - 1))
    x1 = min(grid.shape[1], px + cells)
    found = np.argwhere(grid[y0:y1, x0:x1] == NEG)
    if found.size == 0:
        return None
    pick = found[int(rng.integers(len(found)))]
    return int(pick[0]) + y0, int(pick[1]) + x0


def _cover_both(py, px, ny, nx, cells, shape, rng):
    y0 = max(0, max(py, ny) - cells + 1)
    y1 = min(py, ny, shape[0] - cells)
    x0 = max(0, max(px, nx) - cells + 1)
    x1 = min(px, nx, shape[1] - cells)
    if y0 > y1 or x0 > x1:
        return None
    return int(rng.integers(y0, y1 + 1)), int(rng.integers(x0, x1 + 1))


def _slice_window(rgb, grid, y0, x0, cells):
    y1 = (y0 + cells) * STRIDE
    x1 = (x0 + cells) * STRIDE
    if y1 > rgb.shape[0] or x1 > rgb.shape[1]:
        return None
    image = np.ascontiguousarray(rgb[y0 * STRIDE : y1, x0 * STRIDE : x1])
    labels = np.ascontiguousarray(grid[y0 : y0 + cells, x0 : x0 + cells])
    return image, labels


def _resize_query(image, mask, width, height):
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    scaled = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    return resized, scaled.astype(bool)


def _rotate_pair(image, mask, turns):
    if turns == 0:
        return image, mask
    return np.rot90(image, turns), np.rot90(mask, turns)


def _apply_gains(image: np.ndarray, gains: np.ndarray) -> np.ndarray:
    img = image.astype(np.float32)
    luma = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    paper = float(np.percentile(luma, 90))
    chroma = img.max(axis=-1) - img.min(axis=-1)
    ink = (luma + 8 < paper) | (chroma > 12)
    img[ink] *= gains
    np.clip(img, 0, 255, out=img)
    return img.astype(np.uint8)


def _query_canvas(rgb, box, halo):
    x0, y0, x1, y1 = box
    height, width = rgb.shape[:2]
    top, left = y0 - halo, x0 - halo
    bottom, right = y1 + halo, x1 + halo
    canvas = np.full((bottom - top, right - left, 3), 255, dtype=np.uint8)
    inside = np.zeros((bottom - top, right - left), dtype=bool)
    _paste_canvas(canvas, rgb, top, left, height, width)
    inside[y0 - top : y1 - top, x0 - left : x1 - left] = True
    return canvas, inside


def _paste_canvas(canvas, rgb, top, left, height, width):
    src_top, src_left = max(top, 0), max(left, 0)
    src_bottom = min(top + canvas.shape[0], height)
    src_right = min(left + canvas.shape[1], width)
    dst_top, dst_left = src_top - top, src_left - left
    dst_bottom = dst_top + (src_bottom - src_top)
    dst_right = dst_left + (src_right - src_left)
    canvas[dst_top:dst_bottom, dst_left:dst_right] = rgb[src_top:src_bottom, src_left:src_right]


def _has_both(grid: np.ndarray) -> bool:
    return bool(np.any(grid == POS) and np.any(grid == NEG))


def _group_rows(rows: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["document_id"], []).append(row)
    return groups


def _read_image(data_root: Path, image_name: str) -> np.ndarray:
    with Image.open(data_path(data_root, image_name)) as image:
        return np.array(image.convert("RGB"))


def _first_usable(rows: list[dict], data_root: Path, rng: np.random.Generator):
    order = list(range(len(rows)))
    rng.shuffle(order)
    for index in order:
        prepared = prepare_query(rows[index], data_root)
        if prepared is not None:
            return prepared, rows[index]["image"]
    return None


def prepare_query(row: dict, data_root: Path) -> PreparedQuery | None:
    positive, known, _, _ = reviewed_domains(row, data_root)
    grid = build_label_grid(positive, known, STRIDE)
    del positive, known
    if not np.any(grid == POS) or not np.any(grid == NEG):
        return None
    return PreparedQuery(
        grid=grid,
        pos_yx=np.argwhere(grid == POS).astype(np.int32),
        neg_dist=_chebyshev_to_negatives(grid),
        query_box=tuple(int(v) for v in row["query_box"]),
    )


def _chebyshev_to_negatives(grid: np.ndarray) -> np.ndarray:
    field = np.ones(grid.shape, dtype=np.uint8)
    field[grid == NEG] = UNK
    return distance_transform_cdt(field, metric="chessboard").astype(np.int16)
