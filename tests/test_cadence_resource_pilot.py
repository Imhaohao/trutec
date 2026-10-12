from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate import EvaluationError
from hatch_cadence import LagState
from tools import cadence_resource_pilot as pilot
from tools.cadence_fit_support import support_gate


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows() -> list[dict[str, object]]:
    return [{"id": f"q-{i:02d}", "document_id": f"doc-{i:02d}", "image": "large.png",
             "width": 20, "height": 20, "query_box": [0, 0, 16, 16], "context_boxes": []}
            for i in range(30)]


def _support(rows: list[dict[str, object]], gray: np.ndarray, lag: LagState) -> dict[str, object]:
    crop = gray[:16, :16]
    records = [{"id": row["id"], "document_id": row["document_id"], "crop_shape": [16, 16],
                "crop_pixel_sha256": hashlib.sha256(crop.tobytes()).hexdigest(), "stable": True,
                "lag": json.loads(json.dumps(asdict(lag))), "dual_class": True,
                "positive_reviewed_pixels": 1, "negative_reviewed_pixels": 1} for row in rows]
    return {
        "schema": "cadence-fit-support/v1", "manifest_sha256": "fixture-manifest",
        "protocol_sha256": _sha(pilot.PROTOCOL),
        "source_sha256": {"hatch_cadence.py": _sha(pilot.ROOT / "hatch_cadence.py"),
                           "tools/cadence_fit_support.py": _sha(pilot.ROOT / "tools/cadence_fit_support.py")},
        "support_gate": {"passed": True, "stable_dual_queries": 30, "stable_dual_documents": 30},
        "queries": 30, "per_query": records,
    }


def test_support_gate_requires_stable_dual_rows_across_documents() -> None:
    records = [{"stable": True, "dual_class": True, "document_id": f"d{i}"} for i in range(30)]
    assert support_gate(records)["passed"] is True
    assert support_gate(records[:29])["passed"] is False
    same_document = [{**row, "document_id": "one"} for row in records]
    assert support_gate(same_document)["passed"] is False


def test_support_rejects_manifest_or_id_drift(tmp_path: Path) -> None:
    rows = _rows()
    gray = np.tile(np.arange(20, dtype=np.uint8), (20, 1))
    lag = LagState(8.0, 0.0, (None, None, None, None))
    report = _support(rows, gray, lag)
    with pytest.raises(EvaluationError, match="IDs"):
        pilot._verify_support(report, rows[:-1], "fixture-manifest")
    report["manifest_sha256"] = "wrong"
    with pytest.raises(EvaluationError, match="hash"):
        pilot._verify_support(report, rows, "fixture-manifest")


def test_selects_largest_drawing_with_stable_query() -> None:
    rows = [
        {"id": "small", "image": "small.png", "width": 10, "height": 10},
        {"id": "large-z", "image": "large.png", "width": 40, "height": 20},
        {"id": "large-a", "image": "large.png", "width": 40, "height": 20},
    ]
    reports = {"small": {"stable": True}, "large-z": {"stable": True}, "large-a": {"stable": True}}
    assert pilot._pick_row(rows, reports)["id"] == "large-a"
    reports["large-a"]["stable"] = False
    assert pilot._pick_row(rows, reports)["id"] == "large-z"


def test_reference_points_cover_tile_and_image_corners_deterministically() -> None:
    first = pilot.select_reference_points(400, 400)
    second = pilot.select_reference_points(400, 400)
    assert np.array_equal(first, second)
    assert len({tuple(point) for point in first.tolist()}) == len(first)
    expected = {(0, 0), (0, 399), (399, 0), (399, 399)}
    for y0 in (0, 168, 336):
        y1 = min(y0 + 168, 400) - 1
        for x0 in (0, 168, 336):
            x1 = min(x0 + 168, 400) - 1
            expected.update(((y0, x0), (y0, x1), (y1, x0), (y1, x1)))
    assert set(map(tuple, pilot._boundary_points(400, 400, 168).tolist())) == expected
    assert expected <= set(map(tuple, first.tolist()))
    assert set(map(tuple, [(0, 0), (0, 399), (399, 0), (399, 399)])) <= set(map(tuple, first.tolist()))


def test_small_mocked_pilot_records_hashes_grid_and_equivalence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    gray = np.tile(np.arange(20, dtype=np.uint8), (20, 1))
    lag = LagState(8.0, 0.0, (None, None, None, None))
    rows = _rows()
    monkeypatch.setattr(pilot, "_manifest_rows", lambda _path: (rows, "fixture-manifest"))
    monkeypatch.setattr(pilot, "read_drawing", lambda _root, _row: gray.copy())
    monkeypatch.setattr(pilot, "estimate_query_lag", lambda _crop: lag)
    monkeypatch.setattr(pilot, "_system_swap_snapshot", iter((
        {"captured_at_utc": "start", "page_size_bytes": 4096, "swapout_pages": 12},
        {"captured_at_utc": "finish", "page_size_bytes": 4096, "swapout_pages": 12},
    )).__next__)
    support_path = tmp_path / "support.json"
    support_path.write_text(json.dumps(_support(rows, gray, lag)), encoding="utf-8")
    output = tmp_path / "pilot.json"
    proof = pilot.run_pilot(tmp_path / "fit.json", tmp_path, support_path, output)
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert json.loads(json.dumps(proof)) == saved
    assert proof["status"] == "system_swap_passed_process_gate_pending"
    assert proof["drawing_pixel_sha256"] == hashlib.sha256(gray.tobytes()).hexdigest()
    assert proof["feature_shape"] == [5, 5, 5]
    assert proof["reference_points"] == 25
    assert proof["direct_max_abs_diff"] == 0
    assert proof["system_swap"]["swapout_delta_pages"] == 0
    assert proof["unavailable_fallback_verified"] is True


def test_unavailable_lag_uses_explicit_fallback() -> None:
    gray = np.zeros((17, 18), dtype=np.uint8)
    coords, values = pilot._feature_grid(gray, LagState(None, None, (None, None, None, None)))
    assert coords.shape == (25, 2)
    assert np.array_equal(values, np.broadcast_to(np.asarray([0.5, 0.5, 0.5, 0.5, 0.0], np.float32), (5, 5, 5)))


def test_vm_stat_parses_native_page_size_and_swapouts(monkeypatch: pytest.MonkeyPatch) -> None:
    output = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free: 20.\nSwapouts: 17251957.\n"
    monkeypatch.setattr(pilot.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=output))
    snapshot = pilot._system_swap_snapshot()
    assert snapshot["page_size_bytes"] == 16384
    assert snapshot["swapout_pages"] == 17251957
