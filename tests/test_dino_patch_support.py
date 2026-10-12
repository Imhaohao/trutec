from __future__ import annotations

import numpy as np
import pytest

from tools.dino_patch_support import analyze_fit_rows, count_patch_support, verify_fit_manifest


def test_patch_support_counts_pure_mixed_and_empty_cells() -> None:
    positive = np.zeros((28, 42), dtype=bool)
    negative = np.zeros((28, 42), dtype=bool)
    positive[:5, :5] = True
    negative[:2, 14:18] = True
    positive[14:16, 14:17] = True
    negative[16:19, 14:17] = True

    counts = count_patch_support(positive, positive | negative)

    assert counts["pure_positive_patch_cells"] == 1
    assert counts["pure_negative_patch_cells"] == 1
    assert counts["mixed_label_patch_cells"] == 1
    assert counts["empty_patch_cells"] == 3
    assert counts["pure_positive_pixels"] == 25
    assert counts["pure_negative_pixels"] == 8
    assert counts["mixed_positive_pixels"] == 6
    assert counts["mixed_negative_pixels"] == 9


def test_reviewed_domain_excludes_query_and_context_from_patch_counts(tmp_path) -> None:
    row = {
        "id": "q1", "document_id": "d1", "image": "drawing.png", "width": 28, "height": 28,
        "query_box": [0, 0, 2, 2], "context_boxes": [[14, 0, 16, 2]],
        "labels": {
            "positive_boxes": [[0, 0, 14, 14], [14, 14, 17, 20]],
            "negative_boxes": [[14, 0, 28, 14], [17, 14, 28, 28]],
        },
    }

    report = analyze_fit_rows([row], tmp_path)
    query = report["per_query"][0]

    assert query["original_known_pixels"] == 564
    assert query["excluded_known_pixels"] == 8
    assert query["known_pixels"] == 556
    assert query["positive_pixels"] == 210
    assert query["negative_pixels"] == 346
    assert query["pure_positive_patch_cells"] == 1
    assert query["pure_negative_patch_cells"] == 1
    assert query["mixed_label_patch_cells"] == 1
    assert report["labels_only_no_image_or_model_features"] is True


def test_positive_only_fit_row_keeps_positive_support_without_negative() -> None:
    positive = np.zeros((14, 14), dtype=bool)
    positive[2:7, 3:8] = True

    counts = count_patch_support(positive, positive)

    assert counts["pure_positive_patch_cells"] == 1
    assert counts["pure_negative_patch_cells"] == 0
    assert counts["positive_pixels"] == 25
    assert counts["negative_pixels"] == 0


def test_positive_only_query_is_reported_without_fabricated_negative_support(tmp_path) -> None:
    row = {
        "id": "positive-only", "document_id": "d1", "image": "drawing.png", "width": 14, "height": 14,
        "query_box": [0, 0, 1, 1], "labels": {"positive_boxes": [[1, 1, 14, 14]]},
    }

    report = analyze_fit_rows([row], tmp_path)

    assert report["positive_only_queries"] == 1
    assert report["eligible_positive_queries"] == 1
    assert report["dual_pure_queries"] == 0
    assert report["totals"]["negative_pixels"] == 0


def test_wrong_manifest_hash_is_rejected_before_loading_labels(tmp_path) -> None:
    path = tmp_path / "other.json"
    path.write_text("not the pinned fit manifest", encoding="utf-8")

    with pytest.raises(ValueError, match="Fit manifest SHA-256 mismatch"):
        verify_fit_manifest(path)


def test_odd_edge_geometry_does_not_count_padding_as_label_support() -> None:
    positive = np.zeros((15, 29), dtype=bool)
    known = np.zeros((15, 29), dtype=bool)
    positive[14, 28] = True
    known[14, 28] = True

    counts = count_patch_support(positive, known)

    assert counts["patch_rows"] == 2
    assert counts["patch_columns"] == 3
    assert counts["known_pixels"] == 1
    assert counts["pure_positive_patch_cells"] == 1
    assert counts["pure_negative_patch_cells"] == 0
    assert counts["empty_patch_cells"] == 5
    assert counts["pure_positive_pixels"] == 1
    assert counts["pure_negative_pixels"] == 0
