"""Synthetic-only contract tests for the SegGPT resource probe."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import seggpt_resource_probe as probe
from seggpt_matching import SegGptError


def test_synthetic_case_is_deterministic_and_fits_prompt_canvas() -> None:
    target_a, query_a = probe.synthetic_case()
    target_b, query_b = probe.synthetic_case()

    assert target_a.shape == (448, 448)
    assert query_a.shape == (128, 160)
    assert target_a.dtype == np.uint8 and query_a.dtype == np.uint8
    assert np.array_equal(target_a, target_b)
    assert np.array_equal(query_a, query_b)


def test_probe_tensor_contract_requires_three_448_tensors() -> None:
    valid = {name: np.zeros((1, 3, 448, 448), dtype=np.float32) for name in (
        "pixel_values", "prompt_pixel_values", "prompt_masks"
    )}
    probe._validate_tensor_shapes(valid)

    invalid = dict(valid)
    invalid["pixel_values"] = np.zeros((1, 3, 447, 448), dtype=np.float32)
    with pytest.raises(SegGptError):
        probe._validate_tensor_shapes(invalid)


def test_probe_validates_default_mask_counts() -> None:
    mask = probe._mask_counts(np.concatenate((np.zeros(784, dtype=bool), np.ones(784, dtype=bool)))[None])

    assert mask == {
        "total": 1568,
        "prompt_visible": 784,
        "prompt_masked": 0,
        "target_masked": 784,
        "target_visible": 0,
    }
    probe._validate_mask_counts(mask)
    with pytest.raises(SegGptError):
        probe._validate_mask_counts({**mask, "prompt_masked": 1})


def test_probe_output_check_accepts_only_finite_bounded_448_scores() -> None:
    valid = np.full((448, 448), 0.5, dtype=np.float32)
    assert all(probe._output_checks(valid).values())
    assert not probe._output_checks(np.full((447, 448), 0.5, dtype=np.float32))["shape"]
    assert not probe._output_checks(np.full((448, 448), np.nan, dtype=np.float32))["finite"]
    assert not probe._output_checks(np.full((448, 448), 1.1, dtype=np.float32))["bounded"]


def test_local_loader_rejects_missing_snapshot_before_importing_model_stack(tmp_path: Path) -> None:
    with pytest.raises(SegGptError, match="model.safetensors"):
        from seggpt_matching import load_local_model

        load_local_model(tmp_path / "missing")
