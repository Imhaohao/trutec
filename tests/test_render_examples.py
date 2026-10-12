"""Focused checks for the native prediction review renderer."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate import EvaluationError
from render_examples import (
    _display_stride,
    annotation_layers,
    load_drawing,
    load_prediction,
    render_examples,
    select_rows,
)


PILOT_ID = "q-0b6fa1341f47a6fb"


def _pilot_paths() -> tuple[Path, Path, Path]:
    root = Path(__file__).resolve().parents[1]
    manifest = root / "artifacts" / "splits" / "development.json"
    data_root = root / "dataset"
    predictions = root / "artifacts" / "native-fusion-pilot" / "predictions"
    required = (manifest, data_root, predictions / f"{PILOT_ID}.png")
    if not all(path.exists() for path in required):
        pytest.skip("native pilot artifact or dataset is unavailable")
    return manifest, data_root, predictions


def test_selected_pilot_domains_respect_evaluator_exclusions() -> None:
    manifest, data_root, predictions = _pilot_paths()
    row = select_rows(manifest, [PILOT_ID])[0]
    drawing = load_drawing(row, data_root)
    assert drawing.shape == (row["height"], row["width"], 3)
    positive, negative, unscored, metadata = annotation_layers(row, data_root)
    x0, y0, x1, y1 = row["query_box"]
    query = np.zeros((row["height"], row["width"]), dtype=bool)
    query[y0:y1, x0:x1] = True

    assert bool(np.all(unscored[query]))
    assert not bool(np.any(positive & negative))
    assert not bool(np.any((positive | negative) & unscored))
    assert 0 <= int(metadata["excluded_known_pixels"]) <= int(query.sum())
    prediction = load_prediction(row, predictions)
    assert prediction.shape == (row["height"], row["width"])
    assert prediction.dtype == np.bool_


def test_annotation_layers_exclude_query_and_context_boxes(tmp_path: Path) -> None:
    width, height = 12, 10
    row = {
        "id": "context-example",
        "width": width,
        "height": height,
        "query_box": [1, 1, 3, 3],
        "context_boxes": [[5, 2, 7, 4]],
        "labels": {
            "known_boxes": [[0, 0, 12, 10]],
            "positive_boxes": [[8, 8, 10, 9]],
            "negative_boxes": [[3, 3, 4, 4]],
        },
    }

    positive, negative, unscored, metadata = annotation_layers(row, tmp_path)
    query = np.zeros((height, width), dtype=bool)
    query[1:3, 1:3] = True
    context = np.zeros_like(query)
    context[2:4, 5:7] = True

    assert bool(np.all(unscored[query]))
    assert bool(np.all(unscored[context]))
    assert not bool(np.any(positive & (query | context)))
    assert not bool(np.any(negative & (query | context)))
    assert not bool(np.any(positive & negative))
    assert metadata["excluded_known_pixels"] == int(query.sum() + context.sum())


def test_display_stride_bounds_native_panel_sampling() -> None:
    assert _display_stride(3400, 2200) == 3
    assert _display_stride(1600, 1600) == 1


def test_renderer_exports_pilot_png_at_300_dpi_and_pdf(tmp_path: Path) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    del matplotlib
    manifest, data_root, predictions = _pilot_paths()
    rendered = render_examples(manifest, data_root, predictions, tmp_path, [PILOT_ID])

    assert len(rendered) == 1
    result = rendered[0]
    assert result.png_path == tmp_path / f"{PILOT_ID}.png"
    assert result.pdf_path == tmp_path / f"{PILOT_ID}.pdf"
    assert result.pdf_path.read_bytes().startswith(b"%PDF-")
    with Image.open(result.png_path) as image:
        assert image.format == "PNG"
        assert image.info["dpi"][0] == pytest.approx(300.0, abs=1.0)
        assert image.info["dpi"][1] == pytest.approx(300.0, abs=1.0)
        assert image.width > image.height


def test_selection_rejects_duplicate_and_unknown_ids() -> None:
    manifest = Path(__file__).resolve().parents[1] / "artifacts" / "splits" / "development.json"
    if not manifest.exists():
        pytest.skip("development manifest is unavailable")
    with pytest.raises(EvaluationError, match="Duplicate"):
        select_rows(manifest, [PILOT_ID, PILOT_ID])
    with pytest.raises(EvaluationError, match="absent"):
        select_rows(manifest, ["missing-query-id"])
