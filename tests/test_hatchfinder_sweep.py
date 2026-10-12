"""Synthetic exactness tests for the pretrained HatchFinder development sweep."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evaluate_hatchfinder_sweep as sweep
import inference_hatchfinder as adapter


class NonconstantLogits:
    """Deterministic fake model with local tile seams and query dependence."""

    def __init__(self, torch: Any) -> None:
        self.torch = torch
        self.calls = 0

    def __call__(self, drawing: Any, mask: Any, hatch: Any) -> Any:
        torch = self.torch
        self.calls += 1
        assert bool(torch.all(mask == 1))
        height, width = int(drawing.shape[-2]), int(drawing.shape[-1])
        y = torch.arange(height, dtype=torch.float32, device=drawing.device).reshape(1, 1, height, 1)
        x = torch.arange(width, dtype=torch.float32, device=drawing.device).reshape(1, 1, 1, width)
        local = drawing[:, 0:1] * 1.7 - drawing[:, 1:2] * 0.8
        return local + 0.013 * y + 0.009 * x + hatch.mean() * 0.37


def test_sparse_probabilities_equal_full_tiles_on_odd_shape_and_seams() -> None:
    torch = pytest.importorskip("torch")
    height, width = 73, 101
    rgb = _marked_rgb(height, width)
    query_box = (7, 9, 23, 27)
    known = np.random.default_rng(11).random((height, width)) < 0.17
    known[0, 0] = True
    known[-1, -1] = True
    known[36, :] = True
    known[:, 37] = True

    full_model = NonconstantLogits(torch)
    full = adapter.tiled_probability(full_model, rgb, query_box, 64, 16, "cpu")
    sparse_model = NonconstantLogits(torch)
    sparse = sweep.score_known_probabilities(
        sparse_model, rgb, query_box, known, 64, 16, "cpu"
    )

    np.testing.assert_allclose(sparse, full[known], rtol=0.0, atol=1e-6)
    assert sparse.dtype == np.float32
    assert sparse_model.calls == full_model.calls


def test_sparse_sweep_skips_tiles_without_known_pixels() -> None:
    torch = pytest.importorskip("torch")
    height, width = 100, 100
    rgb = _marked_rgb(height, width)
    known = np.zeros((height, width), dtype=bool)
    known[10, 10] = True

    model = NonconstantLogits(torch)
    sparse = sweep.score_known_probabilities(model, rgb, (2, 2, 14, 14), known, 64, 16, "cpu")
    full_model = NonconstantLogits(torch)
    full = adapter.tiled_probability(full_model, rgb, (2, 2, 14, 14), 64, 16, "cpu")

    np.testing.assert_allclose(sparse, full[known], rtol=0.0, atol=1e-6)
    assert model.calls == 1
    assert full_model.calls == 4


def test_run_sweep_writes_incremental_records_and_probability_npz(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    height, width = 33, 41
    rgb = _marked_rgb(height, width)
    Image.fromarray(rgb, mode="RGB").save(tmp_path / "drawing.png")
    known = np.ones((height, width), dtype=np.uint8) * 255
    positive = np.zeros((height, width), dtype=np.uint8)
    positive[18:25, 18:25] = 255
    Image.fromarray(known, mode="L").save(tmp_path / "known.png")
    Image.fromarray(positive, mode="L").save(tmp_path / "positive.png")
    row = {
        "id": "query_01",
        "document_id": "document_01",
        "kind": "real",
        "image": "drawing.png",
        "width": width,
        "height": height,
        "query_box": [4, 4, 12, 12],
        "context_boxes": [],
        "labels": {"known_mask": "known.png", "positive_mask": "positive.png"},
    }

    output_dir = tmp_path / "sweep"
    records = sweep.run_sweep(
        [row], tmp_path, NonconstantLogits(torch), output_dir, 64, 16, "cpu", [0.5]
    )

    assert len(records) == 1
    logged = [json.loads(line) for line in (output_dir / "query-records.jsonl").read_text().splitlines()]
    assert logged[0]["id"] == "query_01"
    probability_path = output_dir / "probabilities" / "query_01.npz"
    assert probability_path.is_file()
    with np.load(probability_path) as data:
        assert data["probability"].dtype == np.float32
        assert data["positive"].dtype == np.bool_
        assert data["blank"].dtype == np.bool_
        assert data["id"].item() == "query_01"
        assert len(data["probability"]) == int(data["positive"].sum()) + int(
            (~data["positive"]).sum()
        )


def test_parser_defaults_include_pretrained_sweep_contract() -> None:
    args = sweep.parse_args(
        ["--manifest", "m", "--data-root", "d", "--model", "w", "--output-dir", "o"]
    )
    assert args.device == "mps"
    assert args.tile_size == 896
    assert args.overlap == 128
    assert sweep.DEFAULT_THRESHOLDS == (
        0.01,
        0.025,
        0.05,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        0.95,
        0.99,
    )


def _marked_rgb(height: int, width: int) -> np.ndarray:
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[:, :, 0] = np.arange(height, dtype=np.uint8)[:, None]
    rgb[:, :, 1] = np.arange(width, dtype=np.uint8)[None, :]
    rgb[:, :, 2] = (np.arange(height, dtype=np.uint16)[:, None] + np.arange(width, dtype=np.uint16)[None, :]).astype(np.uint8)
    return rgb
