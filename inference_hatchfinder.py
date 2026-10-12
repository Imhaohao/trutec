"""Native overlapping-tile adapter for a pretrained HatchFinder model.

The challenge does not provide a search mask. Each tile gets an all-ones mask
whose shape comes only from the padded tile. RGB stays in the original 0-1
scale. Accumulators stay float32 on CPU so a 7200x4800 sheet stays under 500MB.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

PAD_MULTIPLE = 64
MIN_QUERY_SIDE = 8
WHITE = 255


class AdapterError(ValueError):
    """The adapter input, weights, or model output cannot be used."""


def axis_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    """Window origins that reach both edges, with steps no larger than the stride."""
    _validate_axis(length, tile_size, overlap)
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    starts = list(range(0, length - tile_size + 1, stride))
    final = length - tile_size
    if starts[-1] != final:
        starts.append(final)
    return _dedupe_starts(starts)


def axis_blend_weights(start: int, valid: int, length: int, overlap: int) -> np.ndarray:
    """Separable Hann ramp. Image edges stay at 1, and every sample stays above 0."""
    weights = np.ones(valid, dtype=np.float32)
    if overlap <= 0:
        return weights
    if start > 0:
        _fade_edge(weights, overlap, at_start=True)
    if start + valid < length:
        _fade_edge(weights, overlap, at_start=False)
    return weights


def extract_query(rgb: np.ndarray, query_box: tuple[int, int, int, int]) -> np.ndarray:
    """Exact crop. A side shorter than 8 is padded with white, never resampled."""
    _check_rgb(rgb)
    x0, y0, x1, y1 = query_box
    height, width = rgb.shape[0], rgb.shape[1]
    if not _box_inside(x0, y0, x1, y1, width, height):
        raise AdapterError("query box is empty or outside the drawing")
    crop = rgb[y0:y1, x0:x1]
    target_h = max(int(crop.shape[0]), MIN_QUERY_SIDE)
    target_w = max(int(crop.shape[1]), MIN_QUERY_SIDE)
    return _pad_white(crop, target_h, target_w)


def tiled_probability(
    model: Any,
    rgb: np.ndarray,
    query_box: tuple[int, int, int, int],
    tile_size: int,
    overlap: int,
    device: str,
) -> np.ndarray:
    """Float32 heatmap at the drawing's own height and width."""
    _check_rgb(rgb)
    _validate_tile_params(tile_size, overlap)
    hatch = rgb_to_tensor(extract_query(rgb, query_box), device)
    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    numerator = np.zeros((height, width), dtype=np.float32)
    denominator = np.zeros((height, width), dtype=np.float32)
    _stitch_all(model, rgb, hatch, numerator, denominator, tile_size, overlap, device)
    return _normalize(numerator, denominator)


def load_requests(path: Path) -> list[dict[str, Any]]:
    """Label-free requests. Annotation fields are rejected by the shared helper."""
    from inference import read_requests

    return read_requests(path)


def select_device(requested: str | None, mps_available: bool) -> str:
    """Pick MPS or CPU. An explicit MPS request is never rewritten to CPU."""
    if requested is None:
        return "mps" if mps_available else "cpu"
    if requested == "cpu":
        return "cpu"
    if requested == "mps" and mps_available:
        return "mps"
    if requested == "mps":
        raise AdapterError("MPS was requested but is not available")
    raise AdapterError("device must be mps or cpu")


def validate_settings(threshold: float, tile_size: int, overlap: int, model_path: Path) -> None:
    if not _threshold_ok(threshold):
        raise AdapterError("threshold must lie in (0, 1)")
    _validate_tile_params(tile_size, overlap)
    _require_weights(model_path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        _hash_chunks(digest, handle)
    return digest.hexdigest()


def package_version() -> str:
    try:
        return version("hatchfinder")
    except PackageNotFoundError:
        return "not-installed"


def run_adapter(
    inputs: Path,
    root: Path,
    output: Path,
    model_path: Path,
    device: str,
    tile_size: int,
    overlap: int,
    threshold: float,
) -> dict[str, Any]:
    """Load the model once, then time load, tiles, stitch, and PNG writing together."""
    started = time.monotonic()
    digest = sha256_file(model_path)
    nbytes = model_path.stat().st_size
    model = load_hatchfinder(model_path, device)
    records = run_queries(
        model, load_requests(inputs), root, output, tile_size, overlap, threshold, device
    )
    _sync(device)
    return build_report(
        time.monotonic() - started,
        device,
        records,
        tile_size,
        overlap,
        threshold,
        digest,
        nbytes,
    )


def run_queries(
    model: Any,
    rows: list[dict[str, Any]],
    root: Path,
    output: Path,
    tile_size: int,
    overlap: int,
    threshold: float,
    device: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for group in _group_by_image(rows).values():
        rgb = read_rgb_drawing(root, group[0])
        records.extend(
            _predict_group(model, rgb, group, output, tile_size, overlap, threshold, device)
        )
    return records


def read_rgb_drawing(root: Path, row: dict[str, Any]) -> np.ndarray:
    from evaluate import data_path

    path = data_path(root, row["image"])
    with Image.open(path) as image:
        rgb = np.array(image.convert("RGB"), dtype=np.uint8)
    expected = (int(row["height"]), int(row["width"]), 3)
    if rgb.shape != expected:
        raise AdapterError("drawing " + str(row["id"]) + " does not match its declared size")
    return rgb


def build_report(
    elapsed: float,
    device: str,
    records: list[dict[str, Any]],
    tile_size: int,
    overlap: int,
    threshold: float,
    digest: str,
    nbytes: int,
) -> dict[str, Any]:
    return {
        "method": "hatchfinder-native-tiles-v1",
        "elapsed_seconds": elapsed,
        "timing_coverage": (
            "weights hash, model loading, image loading, crop, tiling, "
            "forward, stitch, and PNG writing"
        ),
        "hardware": hardware_facts(device),
        "queries": len(records),
        "tile": {"tile_size": tile_size, "overlap": overlap, "threshold": threshold},
        "weights_sha256": digest,
        "weights_bytes": nbytes,
        "package_version": package_version(),
        "examples": records,
    }


def hardware_facts(device: str) -> dict[str, Any]:
    torch = _torch()
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "device": device,
        "torch": torch.__version__,
        "mps_available": bool(torch.backends.mps.is_available()),
    }


def load_hatchfinder(path: Path, device: str) -> Any:
    """Load on CPU, then move to MPS only when that device was selected."""
    from hatchfinder_mps import MPSHatchFinder

    model = MPSHatchFinder(load_model_path=path, device="cpu")
    if device == "mps":
        model = model.to("mps")
    model.eval()
    return model


def resolve_device(requested: str | None) -> str:
    torch = _torch()
    return select_device(requested, bool(torch.backends.mps.is_available()))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--device", choices=("mps", "cpu"), default=None)
    parser.add_argument("--tile-size", type=int, default=896)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(None if argv is None else list(argv))
    device = _cli_device(parser, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = run_adapter(
        args.inputs,
        args.data_root,
        args.output_dir,
        args.model,
        device,
        int(args.tile_size),
        int(args.overlap),
        float(args.threshold),
    )
    destination = args.output_dir / "runtime.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def _cli_device(parser: argparse.ArgumentParser, args: argparse.Namespace) -> str:
    try:
        validate_settings(float(args.threshold), int(args.tile_size), int(args.overlap), args.model)
        return resolve_device(args.device)
    except AdapterError as exc:
        parser.error(str(exc))
        raise


def _validate_axis(length: int, tile_size: int, overlap: int) -> None:
    if length <= 0:
        raise AdapterError("axis length must be positive")
    _validate_tile_params(tile_size, overlap)


def _validate_tile_params(tile_size: int, overlap: int) -> None:
    if tile_size <= 0 or tile_size % PAD_MULTIPLE != 0:
        raise AdapterError("tile size must be a positive multiple of 64")
    if overlap < 0 or overlap >= tile_size:
        raise AdapterError("overlap must be >= 0 and smaller than tile size")


def _threshold_ok(threshold: float) -> bool:
    return bool(threshold > 0.0 and threshold < 1.0)


def _require_weights(path: Path) -> None:
    try:
        with path.open("rb") as handle:
            readable = handle.read(1) != b""
    except OSError as exc:
        raise AdapterError("unreadable weights: " + str(path)) from exc
    if not readable:
        raise AdapterError("unreadable weights: " + str(path))


def _dedupe_starts(starts: list[int]) -> list[int]:
    unique: list[int] = []
    seen: set[int] = set()
    for start in starts:
        if start in seen:
            continue
        seen.add(start)
        unique.append(start)
    return unique


def _fade_edge(weights: np.ndarray, overlap: int, at_start: bool) -> None:
    ramp = min(overlap, int(weights.shape[0]))
    curve = _rising_hann(ramp)
    if at_start:
        weights[:ramp] *= curve
        return
    weights[-ramp:] *= curve[::-1]


def _rising_hann(count: int) -> np.ndarray:
    # The first sample is 1/count, so the ramp never places a zero on a pixel.
    step = np.arange(count, dtype=np.float32) + np.float32(1.0)
    step /= np.float32(count)
    pi = np.float32(np.pi)
    curve = np.float32(0.5) * (np.float32(1.0) - np.cos(pi * step))
    return curve.astype(np.float32)


def _check_rgb(rgb: np.ndarray) -> None:
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise AdapterError("drawing must be a uint8 RGB array")
    if min(rgb.shape[0], rgb.shape[1]) <= 0:
        raise AdapterError("drawing must have positive dimensions")


def _box_inside(x0: int, y0: int, x1: int, y1: int, width: int, height: int) -> bool:
    return bool(0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height)


def _pad_white(image: np.ndarray, height: int, width: int) -> np.ndarray:
    src_h, src_w = int(image.shape[0]), int(image.shape[1])
    if src_h == height and src_w == width:
        return image
    canvas = np.full((height, width, int(image.shape[2])), WHITE, dtype=image.dtype)
    canvas[:src_h, :src_w] = image
    return canvas


def _canvas_side(length: int, tile_size: int) -> int:
    multiple = ((length + PAD_MULTIPLE - 1) // PAD_MULTIPLE) * PAD_MULTIPLE
    if multiple < tile_size:
        return tile_size
    return multiple


def _stitch_all(
    model: Any,
    rgb: np.ndarray,
    hatch: Any,
    numerator: np.ndarray,
    denominator: np.ndarray,
    tile_size: int,
    overlap: int,
    device: str,
) -> None:
    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    y_starts = axis_starts(height, tile_size, overlap)
    x_starts = axis_starts(width, tile_size, overlap)
    for y0 in y_starts:
        _stitch_row(
            model, rgb, hatch, numerator, denominator, y0, x_starts, tile_size, overlap, device
        )


def _stitch_row(
    model: Any,
    rgb: np.ndarray,
    hatch: Any,
    numerator: np.ndarray,
    denominator: np.ndarray,
    y0: int,
    x_starts: list[int],
    tile_size: int,
    overlap: int,
    device: str,
) -> None:
    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    valid_h = min(tile_size, height - y0)
    for x0 in x_starts:
        _blend_tile(
            model, rgb, hatch, numerator, denominator,
            y0, x0, valid_h, width, tile_size, overlap, device,
        )


def _blend_tile(
    model: Any,
    rgb: np.ndarray,
    hatch: Any,
    numerator: np.ndarray,
    denominator: np.ndarray,
    y0: int,
    x0: int,
    valid_h: int,
    width: int,
    tile_size: int,
    overlap: int,
    device: str,
) -> None:
    valid_w = min(tile_size, width - x0)
    canvas = _tile_canvas(rgb[y0:y0 + valid_h, x0:x0 + valid_w], tile_size)
    local = _infer_tile(model, canvas, hatch, valid_h, valid_w, device)
    window = _window(y0, x0, valid_h, valid_w, int(rgb.shape[0]), width, overlap)
    numerator[y0:y0 + valid_h, x0:x0 + valid_w] += local * window
    denominator[y0:y0 + valid_h, x0:x0 + valid_w] += window


def _tile_canvas(crop: np.ndarray, tile_size: int) -> np.ndarray:
    canvas_h = _canvas_side(int(crop.shape[0]), tile_size)
    canvas_w = _canvas_side(int(crop.shape[1]), tile_size)
    return _pad_white(crop, canvas_h, canvas_w)


def _window(
    y0: int, x0: int, valid_h: int, valid_w: int, height: int, width: int, overlap: int
) -> np.ndarray:
    wy = axis_blend_weights(y0, valid_h, height, overlap)
    wx = axis_blend_weights(x0, valid_w, width, overlap)
    return wy[:, None] * wx[None, :]


def _normalize(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    if numerator.shape != denominator.shape or not bool(np.all(denominator > 0)):
        raise AdapterError("stitch left uncovered pixels")
    return numerator / denominator


def _infer_tile(
    model: Any, canvas: np.ndarray, hatch: Any, valid_h: int, valid_w: int, device: str
) -> np.ndarray:
    drawing = rgb_to_tensor(canvas, device)
    mask = _ones_mask(int(canvas.shape[0]), int(canvas.shape[1]), device)
    try:
        return _sigmoid_crop(model, drawing, mask, hatch, canvas.shape[0], canvas.shape[1], valid_h, valid_w)
    finally:
        del drawing, mask


def _sigmoid_crop(
    model: Any,
    drawing: Any,
    mask: Any,
    hatch: Any,
    canvas_h: int,
    canvas_w: int,
    valid_h: int,
    valid_w: int,
) -> np.ndarray:
    torch = _torch()
    with torch.inference_mode():
        logits = model(drawing, mask, hatch)
    try:
        spatial = _finite_probability(logits, canvas_h, canvas_w)
        return spatial[:valid_h, :valid_w].copy()
    finally:
        del logits


def _finite_probability(logits: Any, height: int, width: int) -> np.ndarray:
    torch = _torch()
    shape = tuple(int(dim) for dim in logits.shape)
    if shape != (1, 1, height, width):
        raise AdapterError("unexpected logits shape " + str(shape))
    if not bool(torch.isfinite(logits).all()):
        raise AdapterError("model logits are not finite")
    probs = torch.sigmoid(logits).detach().to("cpu")
    array = np.array(probs.numpy()[0, 0], dtype=np.float32, copy=True)
    if not bool(np.isfinite(array).all()):
        raise AdapterError("model probabilities are not finite")
    return array


def rgb_to_tensor(image: np.ndarray, device: str) -> Any:
    """Pack HWC uint8 into float32 NCHW in [0, 1] with no channel rescaling."""
    torch = _torch()
    scaled = image.astype(np.float32)
    scaled /= np.float32(255.0)
    channels = np.ascontiguousarray(np.transpose(scaled, (2, 0, 1)))
    tensor = torch.from_numpy(channels).unsqueeze(0)
    return tensor.to(device=device, dtype=torch.float32)


def _ones_mask(height: int, width: int, device: str) -> Any:
    torch = _torch()
    return torch.ones((1, 1, height, width), dtype=torch.float32, device=device)


def _torch() -> Any:
    import torch

    return torch


def _hash_chunks(digest: Any, handle: Any) -> None:
    while True:
        chunk = handle.read(1024 * 1024)
        if not chunk:
            return
        digest.update(chunk)


def _group_by_image(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["image"]), []).append(row)
    return grouped


def _predict_group(
    model: Any,
    rgb: np.ndarray,
    group: list[dict[str, Any]],
    output: Path,
    tile_size: int,
    overlap: int,
    threshold: float,
    device: str,
) -> list[dict[str, Any]]:
    records = []
    for row in group:
        _same_shape(rgb, row)
        records.append(_predict_one(model, rgb, row, output, tile_size, overlap, threshold, device))
    return records


def _same_shape(rgb: np.ndarray, row: dict[str, Any]) -> None:
    if rgb.shape[0] != int(row["height"]) or rgb.shape[1] != int(row["width"]):
        raise AdapterError("inconsistent dimensions for a shared drawing")


def _predict_one(
    model: Any,
    rgb: np.ndarray,
    row: dict[str, Any],
    output: Path,
    tile_size: int,
    overlap: int,
    threshold: float,
    device: str,
) -> dict[str, Any]:
    started = time.monotonic()
    probability = tiled_probability(model, rgb, _query_tuple(row["query_box"]), tile_size, overlap, device)
    selected = _write_mask(output, row, probability, threshold)
    _sync(device)
    record = {"id": row["id"], "seconds": time.monotonic() - started, "selected_pixels": selected}
    print(json.dumps(record), flush=True)
    return record


def _query_tuple(box: Sequence[int]) -> tuple[int, int, int, int]:
    if len(box) != 4:
        raise AdapterError("query box must be four integers")
    x0, y0, x1, y1 = box[0], box[1], box[2], box[3]
    if type(x0) is not int or type(y0) is not int or type(x1) is not int or type(y1) is not int:
        raise AdapterError("query box must be four integers")
    return (x0, y0, x1, y1)


def _write_mask(output: Path, row: dict[str, Any], probability: np.ndarray, threshold: float) -> int:
    from inference import save_prediction

    mask = probability >= threshold
    save_prediction(output / (str(row["id"]) + ".png"), mask)
    return int(np.count_nonzero(mask))


def _sync(device: str) -> None:
    if device != "mps":
        return
    _torch().mps.synchronize()


if __name__ == "__main__":
    main()
