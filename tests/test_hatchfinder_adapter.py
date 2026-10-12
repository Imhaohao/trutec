"""Geometry and fake-model checks for the HatchFinder tile adapter."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import inference_hatchfinder as adapter


def test_axis_starts_cover_edges_without_gaps() -> None:
    lengths = (1, 2, 3, 7, 8, 9, 15, 31, 63, 64, 65, 127, 128, 129, 255, 257, 897, 1000, 7200)
    tiles = ((64, 0), (64, 1), (64, 63), (128, 0), (128, 127), (256, 64), (896, 128))
    for length in lengths:
        for tile_size, overlap in tiles:
            _assert_axis(length, tile_size, overlap)


def test_known_axis_starts_include_the_final_edge() -> None:
    assert adapter.axis_starts(1, 64, 16) == [0]
    assert adapter.axis_starts(64, 64, 0) == [0]
    assert adapter.axis_starts(65, 64, 0) == [0, 1]
    assert adapter.axis_starts(128, 64, 0) == [0, 64]
    assert adapter.axis_starts(100, 64, 16) == [0, 36]
    assert adapter.axis_starts(897, 896, 128) == [0, 1]


def test_hann_weights_stay_positive_and_keep_image_edges() -> None:
    tiny = adapter.axis_blend_weights(0, 5, 5, 16)
    assert np.all(tiny == np.float32(1))
    left = adapter.axis_blend_weights(0, 64, 200, 16)
    middle = adapter.axis_blend_weights(48, 64, 200, 16)
    right = adapter.axis_blend_weights(136, 64, 200, 16)
    assert left[0] == np.float32(1) and right[-1] == np.float32(1)
    assert np.all(left > 0) and np.all(middle > 0) and np.all(right > 0)
    assert middle[0] < np.float32(1) and middle[-1] < np.float32(1)


def test_parser_defaults_and_rejected_settings(tmp_path: Path) -> None:
    parser = adapter.build_parser()
    args = parser.parse_args(["--inputs", "a", "--data-root", "b", "--output-dir", "c", "--model", "d"])
    assert args.device is None
    assert (args.tile_size, args.overlap, args.threshold) == (896, 128, 0.5)
    _assert_cli_rejects(tmp_path, {"--threshold": "0"})
    _assert_cli_rejects(tmp_path, {"--threshold": "1"})
    _assert_cli_rejects(tmp_path, {"--tile-size": "100"})
    _assert_cli_rejects(tmp_path, {"--tile-size": "64", "--overlap": "64"})
    _assert_cli_rejects(tmp_path, {"--model": str(tmp_path / "missing.pt")})
    assert not (tmp_path / "predictions").exists()


def test_device_selection_does_not_fall_back_to_cpu() -> None:
    assert adapter.select_device(None, False) == "cpu"
    assert adapter.select_device(None, True) == "mps"
    assert adapter.select_device("cpu", True) == "cpu"
    assert adapter.select_device("mps", True) == "mps"
    with pytest.raises(adapter.AdapterError):
        adapter.select_device("mps", False)
    with pytest.raises(adapter.AdapterError):
        adapter.select_device("cuda", False)


def test_annotation_fields_rejected_via_helper(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    from evaluate import EvaluationError

    clean = tmp_path / "clean.json"
    dirty = tmp_path / "dirty.json"
    _write_manifest(clean, _request())
    assert adapter.load_requests(clean)[0]["id"] == "q1"
    row = _request()
    row["labels"] = {"positive_boxes": [[0, 0, 1, 1]]}
    row["expected_counts"] = {"positive": 1}
    _write_manifest(dirty, row)
    with pytest.raises(EvaluationError):
        adapter.load_requests(dirty)


def test_constant_probability_is_exact_on_odd_and_tiny_shapes() -> None:
    torch = pytest.importorskip("torch")
    model = _ConstantLogits(torch, 0.0)
    shapes = ((1, 1), (1, 2), (3, 3), (7, 5), (9, 15), (63, 63), (65, 63), (63, 65), (100, 7), (7, 100), (129, 65))
    for height, width in shapes:
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        probability = adapter.tiled_probability(model, rgb, (0, 0, 1, 1), 64, 16, "cpu")
        expected = np.full((height, width), np.float32(0.5), dtype=np.float32)
        assert probability.dtype == np.float32
        assert probability.shape == (height, width)
        assert np.array_equal(probability, expected)


def test_drawing_and_query_keep_native_pixels() -> None:
    torch = pytest.importorskip("torch")
    rgb = _marked_rgb(70, 65)
    model = _Recording(torch)
    adapter.tiled_probability(model, rgb, (2, 3, 14, 11), 64, 16, "cpu")
    assert len(model.drawings) == 4
    assert all(np.array_equal(hatch, model.hatches[0]) for hatch in model.hatches)
    _assert_tiles_match_source(model, rgb, 64, 16)
    _assert_query_tensor(model.hatches[0], rgb[3:11, 2:14])
    tiny = _Recording(torch)
    query = np.zeros((6, 9, 3), dtype=np.uint8)
    query[1, 2] = (0, 255, 17)
    query[1, 3] = (255, 0, 17)
    adapter.tiled_probability(tiny, query, (2, 1, 5, 3), 64, 0, "cpu")
    _assert_padded_query(tiny.hatches[0])
    _assert_padded_drawing(tiny.drawings[0], query)


def test_native_prediction_mask_output(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("cv2")
    from evaluate import load_binary_png

    rgb = _marked_rgb(13, 17)
    image_path = tmp_path / "sheet.png"
    _save_rgb(image_path, rgb)
    row = _request()
    row["image"] = "sheet.png"
    row["width"] = 17
    row["height"] = 13
    row["query_box"] = [1, 1, 5, 4]
    output = tmp_path / "predictions"
    output.mkdir()
    adapter.run_queries(_ConstantLogits(torch, 0.0), [row], tmp_path, output, 64, 0, 0.5, "cpu")
    mask = load_binary_png(output / "q1.png", 17, 13)
    assert mask.shape == (13, 17)
    assert bool(mask.all())


def test_malformed_model_output_fails() -> None:
    torch = pytest.importorskip("torch")
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    box = (0, 0, 4, 4)
    with pytest.raises(adapter.AdapterError):
        adapter.tiled_probability(_WrongShape(torch), rgb, box, 64, 0, "cpu")
    with pytest.raises(adapter.AdapterError):
        adapter.tiled_probability(_NonFinite(torch, float("nan")), rgb, box, 64, 0, "cpu")
    with pytest.raises(adapter.AdapterError):
        adapter.tiled_probability(_NonFinite(torch, float("inf")), rgb, box, 64, 0, "cpu")


def _assert_axis(length: int, tile_size: int, overlap: int) -> None:
    starts = adapter.axis_starts(length, tile_size, overlap)
    assert starts == sorted(set(starts))
    assert starts[0] == 0
    stride = tile_size - overlap
    previous = starts[0]
    for start in starts[1:]:
        assert 0 < start - previous <= stride
        previous = start
    if length <= tile_size:
        assert starts == [0]
    else:
        assert starts[-1] == length - tile_size
    covered = np.zeros(length, dtype=bool)
    for start in starts:
        covered[start:start + tile_size] = True
    assert bool(covered.all())


def _assert_cli_rejects(root: Path, overrides: dict[str, str]) -> None:
    weights = root / "weights.pt"
    weights.write_bytes(b"x")
    values = {
        "--inputs": str(root / "inputs.json"),
        "--data-root": str(root),
        "--output-dir": str(root / "predictions"),
        "--model": str(weights),
        "--device": "cpu",
        "--tile-size": "896",
        "--overlap": "128",
        "--threshold": "0.5",
    }
    values.update(overrides)
    argv: list[str] = []
    for flag, value in values.items():
        argv.extend((flag, value))
    with pytest.raises(SystemExit):
        adapter.main(argv)


def _request() -> dict:
    return {
        "id": "q1",
        "image": "drawings/a.png",
        "width": 32,
        "height": 16,
        "query_box": [1, 2, 9, 8],
        "context_boxes": [[10, 2, 14, 6]],
    }


def _write_manifest(path: Path, row: dict) -> None:
    payload = {"schema": "hatch-matching-inputs/v1", "examples": [row]}
    path.write_text(json.dumps(payload), encoding="utf-8")


def _marked_rgb(height: int, width: int) -> np.ndarray:
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[:, :, 0] = np.arange(height, dtype=np.uint8)[:, None]
    rgb[:, :, 1] = np.arange(width, dtype=np.uint8)[None, :]
    rgb[:, :, 2] = 17
    return rgb


def _save_rgb(path: Path, rgb: np.ndarray) -> None:
    from PIL import Image

    Image.fromarray(rgb, mode="RGB").save(path, format="PNG")


def _assert_tiles_match_source(model: _Recording, rgb: np.ndarray, tile_size: int, overlap: int) -> None:
    height, width = rgb.shape[0], rgb.shape[1]
    expected_scale = _scale(rgb)
    index = 0
    for y0 in adapter.axis_starts(height, tile_size, overlap):
        for x0 in adapter.axis_starts(width, tile_size, overlap):
            drawing = model.drawings[index]
            mask = model.masks[index]
            assert drawing.shape == (1, 3, tile_size, tile_size)
            assert drawing.dtype == np.float32
            assert mask.shape == (1, 1, tile_size, tile_size)
            assert np.all(mask == np.float32(1))
            assert np.array_equal(drawing[0], expected_scale[y0:y0 + tile_size, x0:x0 + tile_size].transpose(2, 0, 1))
            index += 1
    assert index == len(model.drawings)


def _assert_query_tensor(hatch: np.ndarray, crop: np.ndarray) -> None:
    assert hatch.shape == (1, 3, crop.shape[0], crop.shape[1])
    assert np.array_equal(hatch[0], _scale(crop).transpose(2, 0, 1))
    assert np.max(hatch) <= np.float32(1)


def _assert_padded_query(hatch: np.ndarray) -> None:
    assert hatch.shape == (1, 3, 8, 8)
    assert hatch[0, 0, 0, 0] == np.float32(0)
    assert hatch[0, 1, 0, 0] == np.float32(1)
    assert hatch[0, 2, 0, 0] == np.float32(17) / np.float32(255)
    assert hatch[0, 0, 0, 1] == np.float32(1)
    assert np.all(hatch[0, :, 2:, :] == np.float32(1))
    assert np.all(hatch[0, :, :, 3:] == np.float32(1))


def _assert_padded_drawing(drawing: np.ndarray, rgb: np.ndarray) -> None:
    assert drawing.shape == (1, 3, 64, 64)
    assert np.array_equal(drawing[0, :, : rgb.shape[0], : rgb.shape[1]], _scale(rgb).transpose(2, 0, 1))
    assert np.all(drawing[0, :, rgb.shape[0]:, :] == np.float32(1))
    assert np.all(drawing[0, :, :, rgb.shape[1]:] == np.float32(1))


def _scale(rgb: np.ndarray) -> np.ndarray:
    scaled = rgb.astype(np.float32)
    scaled /= np.float32(255.0)
    return scaled


class _ConstantLogits:
    def __init__(self, torch: object, value: float) -> None:
        self.torch = torch
        self.value = value

    def __call__(self, drawing: object, mask: object, hatch: object) -> object:
        del mask, hatch
        return self.torch.full(
            (1, 1, drawing.shape[-2], drawing.shape[-1]),
            self.value,
            dtype=self.torch.float32,
            device=drawing.device,
        )


class _Recording:
    def __init__(self, torch: object) -> None:
        self.torch = torch
        self.drawings: list[np.ndarray] = []
        self.masks: list[np.ndarray] = []
        self.hatches: list[np.ndarray] = []

    def __call__(self, drawing: object, mask: object, hatch: object) -> object:
        self.drawings.append(drawing.detach().cpu().numpy().copy())
        self.masks.append(mask.detach().cpu().numpy().copy())
        self.hatches.append(hatch.detach().cpu().numpy().copy())
        return self.torch.zeros(
            (1, 1, drawing.shape[-2], drawing.shape[-1]),
            dtype=self.torch.float32,
            device=drawing.device,
        )


class _WrongShape:
    def __init__(self, torch: object) -> None:
        self.torch = torch

    def __call__(self, drawing: object, mask: object, hatch: object) -> object:
        del drawing, mask, hatch
        return self.torch.zeros((1, 1, 4, 4), dtype=self.torch.float32)


class _NonFinite:
    def __init__(self, torch: object, value: float) -> None:
        self.torch = torch
        self.value = value

    def __call__(self, drawing: object, mask: object, hatch: object) -> object:
        del mask, hatch
        logits = self.torch.zeros(
            (1, 1, drawing.shape[-2], drawing.shape[-1]),
            dtype=self.torch.float32,
            device=drawing.device,
        )
        logits[0, 0, 0, 0] = self.value
        return logits
