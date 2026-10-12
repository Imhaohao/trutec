"""Synthetic sparse-tile tests for the frozen SegGPT development scorer."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evaluate_seggpt_matching_sweep as sweep


def test_sparse_scorer_gathers_row_major_scores_across_tile_and_image_edges() -> None:
    y_grid, x_grid = np.indices((500, 500))
    image = ((y_grid * 7 + x_grid * 11) % 251).astype(np.uint8)
    query = np.full((3, 5, 3), 93, dtype=np.uint8)
    y = np.array([0, 10, 447, 448, 449, 499], dtype=np.int32)
    x = np.array([1, 300, 447, 0, 449, 499], dtype=np.int32)
    seen: list[tuple[int, int]] = []

    def fake_predict(tile: np.ndarray, prompt: np.ndarray) -> np.ndarray:
        assert tile.ndim == 3 and tile.shape[2] == 3
        assert np.array_equal(prompt, query)
        seen.append(tile.shape[:2])
        return tile[:, :, 0].astype(np.float32) / np.float32(255)

    scores = sweep.score_reviewed_coordinates(image, query, y, x, fake_predict)

    expected = image[y, x].astype(np.float32) / np.float32(255)
    assert np.array_equal(scores, expected)
    assert seen == [(448, 448), (52, 448), (52, 52)]


def test_sparse_scorer_rejects_duplicate_and_non_row_major_coordinates() -> None:
    image = np.zeros((4, 5), dtype=np.uint8)
    prompt = np.zeros((1, 1, 3), dtype=np.uint8)
    predictor = lambda tile, query: np.zeros(tile.shape[:2], dtype=np.float32)

    with pytest.raises(ValueError, match="unique and row-major"):
        sweep.score_reviewed_coordinates(
            image, prompt, np.array([1, 1]), np.array([2, 2]), predictor
        )
    with pytest.raises(ValueError, match="unique and row-major"):
        sweep.score_reviewed_coordinates(
            image, prompt, np.array([2, 1]), np.array([0, 4]), predictor
        )


def test_sparse_scorer_checks_shape_bounds_and_finite_tile_scores() -> None:
    image = np.zeros((3, 3), dtype=np.uint8)
    prompt = np.zeros((1, 1, 3), dtype=np.uint8)
    y, x = np.array([0]), np.array([0])

    with pytest.raises(ValueError, match="exact native tile"):
        sweep.score_reviewed_coordinates(image, prompt, y, x, lambda tile, query: np.zeros((2, 2)))
    with pytest.raises(ValueError, match="finite float values"):
        sweep.score_reviewed_coordinates(
            image, prompt, y, x, lambda tile, query: np.full(tile.shape[:2], np.nan)
        )
    with pytest.raises(ValueError, match="outside"):
        sweep.score_reviewed_coordinates(image, prompt, np.array([3]), x, lambda tile, query: tile[:, :, 0])


def test_sparse_score_cache_keeps_legacy_key_and_explicit_score_semantics(tmp_path: Path) -> None:
    row = {"id": "query-1", "document_id": "doc-1", "kind": "real", "width": 2, "height": 1}
    score = np.array([0.25, 0.75], dtype=np.float32)
    path = sweep.save_known_scores(
        tmp_path, row, score, np.array([True, False]), np.array([False, True]),
        np.array([0, 0]), np.array([0, 1]), {"protocol_sha256": "abc"},
    )

    with np.load(path, allow_pickle=False) as archive:
        assert "probability" in archive.files
        assert "seggpt_score" in archive.files
        assert np.array_equal(archive["probability"], score)
        assert "not calibrated probability" in archive["score_semantics"].item()
        assert archive["protocol_sha256"].item() == "abc"


def test_threshold_grid_matches_frozen_float32_step() -> None:
    assert len(sweep.THRESHOLDS) == 101
    assert sweep.THRESHOLDS[0] == 0
    assert sweep.THRESHOLDS[-1] == 1
    assert np.allclose(np.diff(sweep.THRESHOLDS), 0.01, atol=1e-7)


def test_control_cache_hash_uses_frozen_canonical_directory_digest() -> None:
    assert sweep.cache_directory_sha256(
        Path("artifacts/texture-crop-development-v1/probabilities")
    ) == "3d3b60c5ab0d83c7b54453f655eb93669dd5694449cdd8b8bced201030fb2283"
    assert sweep.cache_directory_sha256(
        Path("artifacts/hatchfinder-adapt-development-v1/probabilities")
    ) == "cc5bc2b8fafae526db5007430d398530aef9994444563aa314f830ec5fe3fbe5"


def test_mutated_control_cache_fails_against_frozen_digest(tmp_path: Path) -> None:
    (tmp_path / "a.npz").write_bytes(b"alpha")
    (tmp_path / "z.npz").write_bytes(b"omega")
    expected = "e3b162fd84c3b1f48f3f6f56d13a5872e671938630695d7845bdcf14d936a3ce"

    assert sweep.cache_directory_sha256(tmp_path) == expected
    assert len(sweep._verify_control_cache(tmp_path, {"a", "z"}, expected, "fixture")) == 2
    (tmp_path / "a.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="directory hash"):
        sweep._verify_control_cache(tmp_path, {"a", "z"}, expected, "fixture")
