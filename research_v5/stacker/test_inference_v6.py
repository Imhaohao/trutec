"""Unit tests for the native v6 inference helpers."""
from __future__ import annotations

import bootstrap  # noqa: F401

import json

import numpy as np

from features import cell_index
from inference_v6 import load_requests, native_cell_lookup


def test_native_lookup_matches_sparse_cell_index_on_odd_edges() -> None:
    rng = np.random.default_rng(7)
    height, width = 37, 53
    grid = rng.random((5, 7)).astype(np.float32)
    native = native_cell_lookup(grid, height, width)
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    iy, ix = cell_index(yy.ravel(), xx.ravel(), grid.shape[0], grid.shape[1])
    assert native.shape == (height, width)
    assert np.array_equal(native.ravel(), grid[iy, ix])


def test_load_requests_drops_every_label_field(tmp_path) -> None:
    row = {
        "id": "q-1", "image": "a.png", "width": 10, "height": 8, "query_box": [0, 0, 4, 4],
        "context_boxes": [], "document_id": "doc-1", "kind": "real",
        "labels": {"positive_mask": "p.png", "known_mask": "k.png"},
    }
    path = tmp_path / "inputs.json"
    path.write_text(json.dumps({"schema": "hatch-matching-challenge/v1", "examples": [row]}), encoding="utf-8")
    [request] = load_requests(path)
    assert set(request) == {"id", "image", "width", "height", "query_box", "context_boxes"}
