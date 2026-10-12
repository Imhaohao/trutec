"""Tests for validation report integrity and sparse coordinate diagnostics."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate import EvaluationError, load_manifest, reviewed_domains
import analyze_validation as analysis


def _row(identifier: str, document: str = "doc-1") -> dict[str, object]:
    return {
        "id": identifier,
        "document_id": document,
        "kind": "real",
        "image": "drawing.png",
        "width": 7,
        "height": 5,
        "query_box": [0, 0, 2, 2],
        "context_boxes": [],
        "labels": {},
    }


def _report(rows: list[dict[str, object]]) -> dict[str, object]:
    examples = [
        {
            "id": row["id"],
            "document_id": row["document_id"],
            "metrics": {
                "reviewed_pixels": 3,
                "tp": 1,
                "fp": 1,
                "fn": 0,
                "tn": 1,
            },
        }
        for row in rows
    ]
    return {
        "schema": "hatch-matching-challenge-metrics/v1",
        "protocol": "Known pixels only",
        "whole_sheet_accuracy_certified": False,
        "examples": examples,
        "summary": {"examples": len(examples)},
    }


def test_official_report_rejects_missing_malformed_and_id_mismatch(tmp_path: Path) -> None:
    rows = [_row("q-a"), _row("q-b", "doc-2")]
    missing = tmp_path / "missing.json"
    with pytest.raises(EvaluationError, match="missing"):
        analysis.load_official_metrics(missing, rows, expected_count=2)

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    with pytest.raises(EvaluationError, match="valid JSON"):
        analysis.load_official_metrics(malformed, rows, expected_count=2)

    mismatch = tmp_path / "mismatch.json"
    payload = _report(rows)
    payload["examples"][1]["id"] = "q-other"  # type: ignore[index]
    mismatch.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(EvaluationError, match="IDs mismatch"):
        analysis.load_official_metrics(mismatch, rows, expected_count=2)


def test_sparse_outputs_preserve_odd_native_coordinates_and_labels(tmp_path: Path) -> None:
    row = _row("q-odd")
    coordinates = np.asarray([[0, 6], [2, 1], [4, 0], [4, 6]], dtype=np.int32)
    texture = np.asarray([0.1, 0.2, 0.8, 0.9], dtype=np.float32)
    cnn = np.asarray([0.9, 0.8, 0.7, 0.6], dtype=np.float32)
    positive = np.asarray([False, True, False, True])
    blank = np.asarray([True, False, True, False])
    metadata = {
        "original_known_pixels": 5,
        "excluded_known_pixels": 1,
        "unscored_pixels": 30,
    }

    path = analysis.save_sparse_outputs(
        tmp_path / "probabilities" / "q-odd.npz",
        row,
        coordinates,
        texture,
        cnn,
        positive,
        blank,
        metadata,
        official_reviewed_pixels=4,
    )
    saved = analysis.load_sparse_outputs(path)
    np.testing.assert_array_equal(saved["y"], coordinates[:, 0])
    np.testing.assert_array_equal(saved["x"], coordinates[:, 1])
    np.testing.assert_array_equal(saved["positive"], positive)
    np.testing.assert_array_equal(saved["blank"], blank)
    np.testing.assert_array_equal(saved["texture_probability"], texture)
    assert int(saved["official_reviewed_pixels"]) == 4


def test_frozen_comparison_reads_three_saved_native_styles_including_misses() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = root / "dataset" / "val.json"
    predictions = root / "artifacts" / "validation-v1" / "predictions"
    official_path = root / "artifacts" / "validation-v1" / "official-metrics.json"
    if not (manifest.is_file() and predictions.is_dir() and official_path.is_file()):
        pytest.skip("saved native validation artifacts are unavailable")

    rows = load_manifest(manifest)
    official = json.loads(official_path.read_text(encoding="utf-8"))
    chosen = {
        "q-054ce5dd82203aa7",  # high-recall small target
        "q-36a1f497a46cb9a3",  # known low-recall validation query
        "q-e3fa0e7c2a7ccbd9",  # known positive-region miss
    }
    selected_rows = [row for row in rows if row["id"] in chosen]
    sparse_rows: list[dict[str, object]] = []
    for row in selected_rows:
        positive, known, blank, metadata = reviewed_domains(row, root / "dataset")
        y, x = np.nonzero(known)
        coordinates = np.stack((y, x), axis=1).astype(np.int32)
        native = analysis.load_binary_png(
            predictions / f"{row['id']}.png", row["width"], row["height"]
        )
        sparse_rows.append(
            {
                "id": row["id"],
                "row": row,
                "coordinates": coordinates,
                "frozen_selected": native[y, x],
                "positive": positive[known],
                "blank": blank[known],
                "metadata": metadata,
            }
        )

    result = analysis.compare_frozen_candidate(
        sparse_rows, rows, official, predictions, require_all=False
    )
    assert result["verified"] is True
    assert result["summary_matches_official"] is True or result["summary_matches_official"] is None
    assert set(result["official_reviewed_pixels"]) == chosen


def test_input_alignment_rejects_different_native_request() -> None:
    row = _row("q-a")
    request = {key: row[key] for key in (
        "id", "image", "width", "height", "query_box", "context_boxes"
    )}
    request["query_box"] = [1, 0, 3, 2]
    with pytest.raises(EvaluationError, match="query_box"):
        analysis.validate_input_alignment([row], [request])


def test_selection_requires_recall_and_native_checks_are_candidate_specific() -> None:
    summaries = {
        "candidate-high-iou": {
            "summary": {"document_macro_iou": 0.90, "global_recall": 0.96, "fp": 100}
        },
        "candidate-recall": {
            "summary": {"document_macro_iou": 0.80, "global_recall": 0.97, "fp": 200}
        },
        "candidate-low-recall": {
            "summary": {"document_macro_iou": 0.99, "global_recall": 0.94, "fp": 1}
        },
    }
    assert analysis._select_recall_eligible_candidate(summaries) == "candidate-high-iou"
    assert analysis._select_recall_eligible_candidate({"x": {
        "summary": {"document_macro_iou": 1.0, "global_recall": 0.94, "fp": 0}
    }}) is None

    verification = {
        "native_final_masks_verified": True,
        "native_verified_candidate_id": analysis.FROZEN_CANDIDATE_ID,
        "official_evaluation_unchanged": True,
        "official_verified_candidate_id": analysis.FROZEN_CANDIDATE_ID,
    }
    frozen_status = analysis._gate_status(
        analysis.FROZEN_CANDIDATE_ID,
        {"summary": {"document_macro_iou": 0.76, "global_recall": 0.96}},
        verification,
    )
    other_status = analysis._gate_status(
        "and|q10_scale_0.7_offset_-0.1|cnn_0.3",
        {"summary": {"document_macro_iou": 0.76, "global_recall": 0.96}},
        verification,
    )
    assert frozen_status["gate_met"] is True
    assert other_status["native_final_masks_verified"] is False
    assert other_status["official_evaluation_unchanged"] is False
    assert other_status["gate_met"] is False

