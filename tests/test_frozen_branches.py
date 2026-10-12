"""Focused checks for frozen native-branch confirmation alignment."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tools.confirm_frozen_branches as confirm


def make_cache() -> tuple[dict[str, np.ndarray], dict[str, object], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    row: dict[str, object] = {"id": "q-one"}
    y = np.array([1, 2, 3], dtype=np.int32)
    x = np.array([4, 5, 6], dtype=np.int32)
    positive = np.array([True, False, True])
    blank = np.array([False, True, False])
    cache = {
        "id": np.asarray("q-one"),
        "y": y.copy(),
        "x": x.copy(),
        "positive": positive.copy(),
        "blank": blank.copy(),
        "cnn_probability": np.array([0.9, 0.3, 0.8], dtype=np.float32),
    }
    return cache, row, y, x, positive, blank


def test_cache_alignment_accepts_exact_coordinates_and_labels() -> None:
    cache, row, y, x, positive, blank = make_cache()
    observed = confirm.validate_cache_alignment(cache, row, y, x, positive, blank)
    assert observed.dtype == np.float32
    np.testing.assert_array_equal(observed, cache["cnn_probability"])


@pytest.mark.parametrize("field", ["id", "y", "x", "positive", "blank"])
def test_cache_alignment_rejects_mismatched_identity_or_reviewed_vectors(field: str) -> None:
    cache, row, y, x, positive, blank = make_cache()
    if field == "id":
        cache[field] = np.asarray("q-other")
    else:
        cache[field] = cache[field].copy()
        cache[field][0] = not bool(cache[field][0])
    with pytest.raises(ValueError, match="mismatch"):
        confirm.validate_cache_alignment(cache, row, y, x, positive, blank)


def test_threshold_uses_float32_probability_boundary() -> None:
    threshold = np.float32(0.15)
    probabilities = np.array(
        [np.nextafter(threshold, np.float32(0)), threshold,
         np.nextafter(threshold, np.float32(1))],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(confirm.threshold_mask(probabilities, 0.15), [False, True, True])


def test_reviewed_decision_comparison_reports_each_mismatch() -> None:
    texture = np.array([0.15, 0.149, 0.9], dtype=np.float32)
    cnn = np.array([0.25, 0.99, 0.24], dtype=np.float32)
    native = np.array([True, True, False])
    candidate, mismatch_count = confirm.compare_reviewed_decisions(texture, cnn, native)
    np.testing.assert_array_equal(candidate, [True, False, False])
    assert mismatch_count == 1


def test_freeze_requires_fixed_h5_candidate() -> None:
    freeze = {
        "schema": "hatch-matching-freeze/v3",
        "native_confirmation_pending": True,
        "public_validation_exposed_before_freeze": True,
        "selection": {
            "operator": "AND", "quantile": "q10", "scale": 0,
            "offset": 0.15, "clip": [0.01, 0.99], "cnn_threshold": 0.25,
        },
    }
    confirm.validate_freeze(freeze)
    freeze["selection"]["cnn_threshold"] = 0.4
    with pytest.raises(ValueError, match="selection"):
        confirm.validate_freeze(freeze)


def test_official_aggregate_must_match_exactly() -> None:
    summary = {"examples": 1, "tp": 3, "global_recall": 1.0}
    confirm._compare_official_summary(summary, {"summary": summary})
    with pytest.raises(ValueError, match="aggregate"):
        confirm._compare_official_summary(summary, {"summary": {**summary, "tp": 2}})
