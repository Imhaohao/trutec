"""Label-free tiled embedding and cosine scores.

A cell is copied from a tile only when its receptive field lies inside that
tile, or on the drawing border where the full forward sees the same padding.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch.nn import functional as F

from constants import EMBED_DIM, HALO, STRIDE, TILE
from model import HatchEncoder
from pixels import pack_channels


def pick_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if name == "mps":
        raise SystemExit("MPS was requested but is not available")
    raise SystemExit("device must be cpu or mps")


def load_encoder(path, device: torch.device) -> HatchEncoder:
    model = HatchEncoder()
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def predict(model, request: dict, rgb: np.ndarray, y, x, device: torch.device, tile: int = TILE, halo: int = HALO):
    box = tuple(int(v) for v in request["query_box"])
    return score_rgb(model, rgb, box, y, x, tile, halo, device)


def score_rgb(model, rgb, query_box, y, x, tile: int, halo: int, device: torch.device):
    embedding = embed_tiled(model, rgb_tensor(rgb), tile, halo, device)
    prototype = prototype_from_embedding(embedding, query_box)
    return scores_at_pixels(embedding, prototype, y, x)


def rgb_tensor(rgb: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(pack_channels(rgb)).unsqueeze(0)


def embed_whole(model, image: torch.Tensor, device: torch.device) -> torch.Tensor:
    return _forward_cpu(model, pad_to_stride(image, 1.0), device)


def embed_tiled(model, image: torch.Tensor, tile: int, halo: int, device: torch.device) -> torch.Tensor:
    check_tile(tile, halo)
    padded = pad_to_stride(image, 1.0)
    if padded.shape[-2] <= tile and padded.shape[-1] <= tile:
        return _forward_cpu(model, padded, device)
    canvas = _nan_canvas(padded)
    _fill_tiles(model, padded, canvas, tile, halo, device)
    _require_covered(canvas)
    return canvas


def prototype_from_embedding(embedding: torch.Tensor, query_box) -> torch.Tensor:
    pooled = _box_cells(embedding, query_box).mean(dim=1)
    return F.normalize(pooled, dim=0)


def scores_at_pixels(embedding: torch.Tensor, prototype: torch.Tensor, y, x) -> np.ndarray:
    vectors = embedding.detach().cpu().numpy()
    weights = prototype.detach().cpu().numpy()
    rows = np.clip(np.asarray(y, dtype=np.int64) // STRIDE, 0, vectors.shape[1] - 1)
    cols = np.clip(np.asarray(x, dtype=np.int64) // STRIDE, 0, vectors.shape[2] - 1)
    cosine = vectors[:, rows, cols].T @ weights
    return ((cosine + 1.0) * 0.5).astype(np.float32)


def pad_to_stride(image: torch.Tensor, value: float) -> torch.Tensor:
    height, width = int(image.shape[-2]), int(image.shape[-1])
    pad_h = (STRIDE - height % STRIDE) % STRIDE
    pad_w = (STRIDE - width % STRIDE) % STRIDE
    if pad_h == 0 and pad_w == 0:
        return image
    return F.pad(image, (0, pad_w, 0, pad_h), value=value)


def check_tile(tile: int, halo: int) -> None:
    if tile % STRIDE or halo % STRIDE:
        raise ValueError("tile and halo must be multiples of the stride")
    if tile <= 2 * halo:
        raise ValueError("tile must be larger than twice the halo")


def _forward_cpu(model, image: torch.Tensor, device: torch.device) -> torch.Tensor:
    with torch.no_grad():
        return model(image.to(device))[0].detach().cpu()


def _nan_canvas(image: torch.Tensor) -> torch.Tensor:
    height = int(image.shape[-2]) // STRIDE
    width = int(image.shape[-1]) // STRIDE
    return torch.full((EMBED_DIM, height, width), float("nan"))


def _fill_tiles(model, padded, canvas, tile, halo, device) -> None:
    height, width = int(padded.shape[-2]), int(padded.shape[-1])
    for y0, yspan in _axis_windows(height, tile, halo):
        for x0, xspan in _axis_windows(width, tile, halo):
            _place_window(model, padded, canvas, y0, x0, yspan, xspan, height, width, halo, device)


def _place_window(model, padded, canvas, y0, x0, yspan, xspan, height, width, halo, device) -> None:
    window = padded[:, :, y0 : y0 + yspan, x0 : x0 + xspan]
    local = _forward_cpu(model, window, device)
    gy0, gy1 = _cell_span(y0, yspan, height, halo, canvas.shape[1])
    gx0, gx1 = _cell_span(x0, xspan, width, halo, canvas.shape[2])
    ly0, lx0 = gy0 - y0 // STRIDE, gx0 - x0 // STRIDE
    ly1, lx1 = gy1 - y0 // STRIDE, gx1 - x0 // STRIDE
    if ly0 < 0 or lx0 < 0 or ly1 > local.shape[1] or lx1 > local.shape[2] or gy0 >= gy1 or gx0 >= gx1:
        raise RuntimeError("tile keep range fell outside the window")
    canvas[:, gy0:gy1, gx0:gx1] = local[:, ly0:ly1, lx0:lx1]


def _axis_windows(length: int, tile: int, halo: int) -> list[tuple[int, int]]:
    if length <= tile:
        return [(0, length)]
    core = tile - 2 * halo
    starts = list(range(0, length - tile + 1, core))
    last = length - tile
    if starts[-1] != last:
        starts.append(last)
    return [(start, tile) for start in starts]


def _cell_span(start: int, span: int, length: int, halo: int, limit: int) -> tuple[int, int]:
    keep0, keep1 = _keep_bounds(start, span, length, halo)
    return _cells_covering(keep0, keep1, limit)


def _keep_bounds(start: int, span: int, length: int, halo: int) -> tuple[int, int]:
    end = start + span
    keep0 = start if start == 0 else start + halo
    keep1 = end if end == length else end - halo
    return keep0, keep1


def _cells_covering(keep0: int, keep1: int, limit: int) -> tuple[int, int]:
    half = STRIDE // 2
    if keep0 <= half:
        start = 0
    else:
        start = math.ceil((keep0 - half) / STRIDE)
    stop = math.floor((keep1 - 1 - half) / STRIDE) + 1
    return max(0, start), min(limit, stop)


def _require_covered(canvas: torch.Tensor) -> None:
    if not bool(torch.isfinite(canvas).all()):
        raise RuntimeError("tiled embedding left uncovered cells")


def _box_cells(embedding: torch.Tensor, query_box) -> torch.Tensor:
    mask = _inside_mask(int(embedding.shape[1]), int(embedding.shape[2]), query_box)
    if bool(mask.any()):
        return embedding[:, mask]
    x0, y0, x1, y1 = (int(v) for v in query_box)
    row = _nearest_index(y0, y1, int(embedding.shape[1]))
    col = _nearest_index(x0, x1, int(embedding.shape[2]))
    return embedding[:, row, col].unsqueeze(1)


def _inside_mask(height: int, width: int, query_box) -> torch.Tensor:
    x0, y0, x1, y1 = (int(v) for v in query_box)
    ys = torch.arange(height) * STRIDE + (STRIDE // 2)
    xs = torch.arange(width) * STRIDE + (STRIDE // 2)
    row = (ys >= y0) & (ys < y1)
    col = (xs >= x0) & (xs < x1)
    return row[:, None] & col[None, :]


def _nearest_index(start: int, stop: int, limit: int) -> int:
    center = (start + stop) / 2
    index = int(round((center - (STRIDE / 2)) / STRIDE))
    return min(max(index, 0), limit - 1)
