"""Unit tests for the v7 pipeline. The drawing test uses one development sheet."""
from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bootstrap  # noqa: E402,F401

import joblib
import numpy as np

from bootstrap import V6_STAGE2  # noqa: E402
from evaluate_dev import load_original  # noqa: E402
from features import cell_index  # noqa: E402
from inference_v6 import predict_drawing  # noqa: E402
from pipeline import gated_maps, identity_descriptor, native_cell_lookup  # noqa: E402


def test_native_lookup_matches_sparse_cell_index_on_odd_edges() -> None:
    rng = np.random.default_rng(7)
    height, width = 37, 53
    grid = rng.random((5, 7)).astype(np.float32)
    native = native_cell_lookup(grid, height, width)
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    iy, ix = cell_index(yy.ravel(), xx.ravel(), grid.shape[0], grid.shape[1])
    assert native.shape == (height, width)
    assert np.array_equal(native.ravel(), grid[iy, ix])


def test_load_requests_drops_every_label_field(tmp_path=None) -> None:
    from inference_v7 import load_requests

    directory = Path(tmp_path) if tmp_path is not None else Path(tempfile.mkdtemp())
    row = {
        "id": "q-1",
        "image": "a.png",
        "width": 10,
        "height": 8,
        "query_box": [0, 0, 4, 4],
        "context_boxes": [],
        "document_id": "doc-1",
        "kind": "real",
        "labels": {"positive_mask": "p.png", "known_mask": "k.png"},
    }
    path = directory / "inputs.json"
    path.write_text(json.dumps({"schema": "hatch-matching-challenge/v1", "examples": [row]}), encoding="utf-8")
    [request] = load_requests(path)
    assert set(request) == {"id", "image", "width", "height", "query_box", "context_boxes"}


def test_identity_cleaner_and_structure_model_match_published_cache() -> None:
    import devharness
    from bootstrap import STRUCTURE_GATED, STRUCTURE_MODEL

    requests = _smallest_requests()
    stage1, _metadata = load_original()
    stage2 = joblib.load(STRUCTURE_MODEL)["model"]
    maps = gated_maps(requests, stage1, stage2, identity_descriptor, devharness.DATA_ROOT)
    for request, native in zip(requests, maps):
        with np.load(devharness.DOMAIN_CACHE / f"{request['id']}.npz") as stored:
            y = np.array(stored["y"])
            x = np.array(stored["x"])
        expected = np.load(STRUCTURE_GATED / f"{request['id']}.npy")
        actual = np.asarray(native[y, x], dtype=np.float32)
        gap = float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64))))
        assert actual.shape == expected.shape
        assert gap == 0.0, f"{request['id']} structure gap {gap}"


def test_identity_cleaner_and_v6_stage2_match_v6_on_one_drawing() -> None:
    import devharness

    requests = _smallest_requests()
    stage1, _metadata = load_original()
    stage2 = joblib.load(V6_STAGE2)["model"]
    expected: dict[str, np.ndarray] = {}

    def emit(request, gated):
        expected[request["id"]] = gated

    predict_drawing(requests, stage1, stage2, devharness.DATA_ROOT, emit)
    maps = gated_maps(requests, stage1, stage2, identity_descriptor, devharness.DATA_ROOT)
    assert len(maps) == len(requests)
    for request, actual in zip(requests, maps):
        reference = expected[request["id"]]
        gap = float(np.max(np.abs(actual.astype(np.float64) - reference.astype(np.float64))))
        assert actual.shape == reference.shape
        assert gap == 0.0, f"{request['id']} max abs gap {gap}"


def _smallest_requests() -> list[dict]:
    import devharness

    rows = json.loads(devharness.DEVELOPMENT_MANIFEST.read_text(encoding="utf-8"))["examples"]
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["image"]] = counts.get(row["image"], 0) + 1
    singles = [row for row in rows if counts[row["image"]] == 1]
    image = min(singles, key=_area)["image"]
    return [devharness.request_view(row) for row in rows if row["image"] == image]


def _area(row: dict) -> int:
    return int(row["width"]) * int(row["height"])


if __name__ == "__main__":
    test_native_lookup_matches_sparse_cell_index_on_odd_edges()
    test_load_requests_drops_every_label_field()
    test_identity_cleaner_and_v6_stage2_match_v6_on_one_drawing()
    test_identity_cleaner_and_structure_model_match_published_cache()
    print("tests passed", flush=True)
