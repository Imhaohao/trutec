"""Label-free tiled logits. A native pixel always maps with y//4, x//4."""

from __future__ import annotations

import numpy as np
import torch

from .constants import OUTPUT_DIVISOR, OVERLAP, TILE
from .geom import pack_half_rgb, query_pack
from .model import HatchContextNet


def pick_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if name == "mps":
        raise SystemExit("MPS was requested but is not available")
    raise SystemExit("device must be cpu or mps")


def load_model(path, device: torch.device) -> HatchContextNet:
    model = HatchContextNet()
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def predict(model, request: dict, rgb: np.ndarray, y, x, device: torch.device, tile: int = TILE, overlap: int = OVERLAP):
    packed = pack_half_rgb(rgb)
    query = query_pack(rgb, tuple(int(value) for value in request["query_box"]))
    logits = stitch_logits(model, packed, query, device, tile, overlap)
    return lookup_sigmoid(logits, y, x)


def forward_logits(model, packed: np.ndarray, query: np.ndarray, device: torch.device) -> np.ndarray:
    padded = pad_spatial(np.asarray(packed, dtype=np.float32), OUTPUT_DIVISOR)
    logits = _run_forward(model, padded, query, device)
    return _crop_valid(logits, int(packed.shape[-2]), int(packed.shape[-1]))


def stitch_logits(model, packed: np.ndarray, query: np.ndarray, device: torch.device, tile: int = TILE, overlap: int = OVERLAP):
    bank = stitch_many(model, packed, {"only": query}, device, tile, overlap)
    return bank["only"]


def stitch_many(model, packed: np.ndarray, queries: dict[str, np.ndarray], device: torch.device, tile: int = TILE, overlap: int = OVERLAP):
    _check_tile(tile, overlap)
    array = np.asarray(packed, dtype=np.float32)
    orig_h, orig_w = int(array.shape[-2]), int(array.shape[-1])
    padded = pad_spatial(array, OUTPUT_DIVISOR)
    if int(padded.shape[-2]) <= tile and int(padded.shape[-1]) <= tile:
        return {key: _crop_valid(_run_forward(model, padded, query, device), orig_h, orig_w) for key, query in queries.items()}
    return _blend_tiles(model, padded, queries, device, tile, overlap, orig_h, orig_w)


def lookup_sigmoid(logits: np.ndarray, y, x) -> np.ndarray:
    """Native (y, x) -> output cell (y // 4, x // 4), then sigmoid.

    Half-resolution drops odd native pixels with a 2x2 mean, and the network
    emits one logit per 2x2 half-resolution block, so the cell is the native
    pixel floor-divided by 4. The same formula is used for every pixel.
    """
    rows = np.clip(np.asarray(y, dtype=np.int64) // 4, 0, logits.shape[0] - 1)
    cols = np.clip(np.asarray(x, dtype=np.int64) // 4, 0, logits.shape[1] - 1)
    return _sigmoid(logits[rows, cols]).astype(np.float32)


def pad_spatial(packed: np.ndarray, multiple: int, value: float = 0.0) -> np.ndarray:
    channels, height, width = packed.shape
    out_h = _round_up(height, multiple)
    out_w = _round_up(width, multiple)
    if out_h == height and out_w == width:
        return packed
    canvas = np.full((channels, out_h, out_w), value, np.float32)
    canvas[:, :height, :width] = packed
    return canvas


def _blend_tiles(model, padded, queries, device, tile, overlap, orig_h, orig_w):
    out_h, out_w = int(padded.shape[-2]) // 2, int(padded.shape[-1]) // 2
    numerators = {key: np.zeros((out_h, out_w), np.float32) for key in queries}
    denominators = {key: np.zeros((out_h, out_w), np.float32) for key in queries}
    codes = _maybe_codes(model, queries, device)
    for window in _windows(int(padded.shape[-2]), int(padded.shape[-1]), tile, overlap):
        _accumulate_window(model, padded, window, queries, codes, numerators, denominators, device, tile, overlap)
    return {key: _crop_valid(_divide(numerators[key], denominators[key]), orig_h, orig_w) for key in queries}


def _accumulate_window(model, padded, window, queries, codes, numerators, denominators, device, tile, overlap):
    y0, x0, valid_h, valid_w = window
    canvas = _canvas(padded, y0, x0, valid_h, valid_w, tile)
    local = _window_logits(model, canvas, queries, codes, device)
    y_out, x_out = y0 // 2, x0 // 2
    out_h, out_w = valid_h // 2, valid_w // 2
    weight = _window_weight(y_out, x_out, out_h, out_w, numerators[next(iter(numerators))].shape, overlap // 2)
    for key, logit in zip(queries, local):
        numerators[key][y_out : y_out + out_h, x_out : x_out + out_w] += logit[:out_h, :out_w] * weight
        denominators[key][y_out : y_out + out_h, x_out : x_out + out_w] += weight


def _window_logits(model, canvas, queries, codes, device):
    drawing = torch.from_numpy(np.ascontiguousarray(canvas)).unsqueeze(0).to(device)
    with torch.inference_mode():
        if codes is None:
            return [_forward_numpy(model, drawing, query, device) for query in queries.values()]
        front = model.encoder(drawing)
        return [_decode_numpy(model, front, code) for code in codes]


def _forward_numpy(model, drawing, query, device) -> np.ndarray:
    query_b = torch.from_numpy(np.ascontiguousarray(query)).unsqueeze(0).to(device)
    return model(drawing, query_b)[0, 0].detach().cpu().numpy()


def _decode_numpy(model, front, code) -> np.ndarray:
    vec, feat = code
    return model.decode(front, vec, feat)[0, 0].detach().cpu().numpy()


def _run_forward(model, packed, query, device) -> np.ndarray:
    drawing = torch.from_numpy(np.ascontiguousarray(packed)).unsqueeze(0).to(device)
    query_b = torch.from_numpy(np.ascontiguousarray(query)).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(drawing, query_b)
    return logits[0, 0].detach().cpu().numpy()


def _maybe_codes(model, queries, device):
    if len(queries) < 2 or not hasattr(model, "query_encoder"):
        return None
    codes = []
    for query in queries.values():
        tensor = torch.from_numpy(np.ascontiguousarray(query)).unsqueeze(0).to(device)
        with torch.inference_mode():
            codes.append(model.query_encoder(tensor))
    return codes


def _windows(height: int, width: int, tile: int, overlap: int):
    for y0 in _axis_starts(height, tile, overlap):
        valid_h = min(tile, height - y0)
        for x0 in _axis_starts(width, tile, overlap):
            yield y0, x0, valid_h, min(tile, width - x0)


def _axis_starts(length: int, tile: int, overlap: int) -> list[int]:
    if length <= tile:
        return [0]
    stride = tile - overlap
    starts = list(range(0, length - tile + 1, stride))
    final = length - tile
    if starts[-1] != final:
        starts.append(final)
    return starts


def _canvas(packed, y0, x0, valid_h, valid_w, tile):
    canvas = np.zeros((packed.shape[0], tile, tile), np.float32)
    canvas[:, :valid_h, :valid_w] = packed[:, y0 : y0 + valid_h, x0 : x0 + valid_w]
    return canvas


def _window_weight(y0, x0, valid_h, valid_w, shape, overlap):
    height, width = shape
    wy = _blend_axis(y0, valid_h, height, overlap)
    wx = _blend_axis(x0, valid_w, width, overlap)
    return wy[:, None] * wx[None, :]


def _blend_axis(start: int, valid: int, length: int, overlap: int) -> np.ndarray:
    weights = np.ones(valid, dtype=np.float32)
    if overlap <= 0:
        return weights
    if start > 0:
        _fade(weights, overlap, at_start=True)
    if start + valid < length:
        _fade(weights, overlap, at_start=False)
    return weights


def _fade(weights: np.ndarray, overlap: int, at_start: bool) -> None:
    ramp = min(overlap, int(weights.shape[0]))
    curve = _rising_hann(ramp)
    if at_start:
        weights[:ramp] *= curve
        return
    weights[-ramp:] *= curve[::-1]


def _rising_hann(count: int) -> np.ndarray:
    step = (np.arange(count, dtype=np.float32) + np.float32(1.0)) / np.float32(count)
    curve = np.float32(0.5) * (np.float32(1.0) - np.cos(np.float32(np.pi) * step))
    return curve.astype(np.float32)


def _divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    if not bool(np.all(denominator > 0)):
        raise RuntimeError("tile stitch left uncovered pixels")
    return numerator / denominator


def _crop_valid(logits: np.ndarray, height: int, width: int) -> np.ndarray:
    return logits[: height // 2, : width // 2]


def _round_up(length: int, multiple: int) -> int:
    return ((length + multiple - 1) // multiple) * multiple


def _check_tile(tile: int, overlap: int) -> None:
    if tile % OUTPUT_DIVISOR or overlap % 2 or overlap < 0 or overlap >= tile:
        raise ValueError("tile must be a multiple of 32 and overlap an even number smaller than the tile")


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values.astype(np.float32), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))
