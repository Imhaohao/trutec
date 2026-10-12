"""Unit checks for the reproducible H5 matched-control driver."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from evaluate import EvaluationError
from tools.train_faint_control import (
    assert_h2_samples_match,
    base_column_indices,
    _validate_cache_coverage,
    read_h2_cache,
    validate_fit_development_split,
    validate_h2_metadata,
    _validate_h2_coverage,
)
import hatch_texture
from texture_learning import pair_feature_names


def test_base_column_lookup_uses_names_not_prefix_order() -> None:
    faint_names = ["local_extra", "local_base", "query_base"]
    h2_names = ["query_base", "local_base"]

    assert base_column_indices(faint_names, h2_names) == [2, 1]


def test_sample_equality_checks_reconstructed_base_columns_labels_and_weights() -> None:
    faint_names = ["query_b", "local_b", "query_a", "local_a"]
    h2_names = ["local_a", "query_a"]
    faint_x = np.array([[7, 2, 8, 1], [3, 4, 5, 6]], dtype=np.float32)
    h2_x = np.array([[1, 8], [6, 5]], dtype=np.float32)
    labels = np.array([1, 0], dtype=np.int32)
    weights = np.array([0.5, 0.5], dtype=np.float32)

    proof = assert_h2_samples_match(
        faint_x, labels, weights, h2_x, labels.copy(), weights.copy(), faint_names, h2_names
    )

    assert proof["matches"] is True
    assert proof["base_columns"] == 2
    with pytest.raises(ValueError, match="base feature samples differ"):
        assert_h2_samples_match(
            faint_x, labels, weights, h2_x + 1, labels, weights, faint_names, h2_names
        )
    with pytest.raises(ValueError, match="labels differ"):
        assert_h2_samples_match(
            faint_x, labels, weights, h2_x, np.array([0, 1], dtype=np.int32), weights,
            faint_names, h2_names,
        )


def test_split_validation_uses_existing_validation_document_guard(tmp_path: Path) -> None:
    validation = tmp_path / "val.json"
    validation.write_text(
        json.dumps({"examples": [{"document_id": "val-doc", "labels": {"secret": "value"}}]}),
        encoding="utf-8",
    )
    fit_rows = [{"id": "fit-q", "document_id": "fit-doc"}]
    dev_rows = [{"id": "dev-q", "document_id": "dev-doc"}]

    documents = validate_fit_development_split(
        fit_rows, dev_rows, Path("fit.json"), Path("development.json"), tmp_path
    )

    assert documents == {"fit": ["fit-doc"], "development": ["dev-doc"]}
    with pytest.raises(EvaluationError, match="share document IDs"):
        validate_fit_development_split(
            fit_rows, [{"id": "dev-q", "document_id": "fit-doc"}],
            Path("fit.json"), Path("development.json"), tmp_path,
        )
    with pytest.raises(EvaluationError, match="overlaps public validation"):
        validate_fit_development_split(
            fit_rows, [{"id": "dev-q", "document_id": "val-doc"}],
            Path("fit.json"), Path("development.json"), tmp_path,
        )


def test_h2_cache_reads_actual_weight_key_and_types(tmp_path: Path) -> None:
    cache = tmp_path / "h2.npz"
    x = np.zeros((2, 3), dtype=np.float32)
    y = np.array([0, 1], dtype=np.int32)
    weights = np.array([0.5, 0.5], dtype=np.float32)
    np.savez_compressed(cache, x=x, y=y, weight=weights)

    loaded_x, loaded_y, loaded_weights = read_h2_cache(cache)

    np.testing.assert_array_equal(loaded_x, x)
    np.testing.assert_array_equal(loaded_y, y)
    np.testing.assert_array_equal(loaded_weights, weights)


def test_h2_metadata_must_match_locked_source_and_feature_contract(tmp_path: Path) -> None:
    fit_manifest = tmp_path / "fit.json"
    fit_manifest.write_text("[]", encoding="utf-8")
    metadata = {
        "query_descriptor_source": "distribution",
        "texture_feature_set": "base",
        "stride": 4,
        "seed": 42,
        "samples_per_class": 512,
        "params": {
            "learning_rate": 0.08,
            "max_leaf_nodes": 31,
            "l2_regularization": 1.0,
            "max_iter": 200,
            "early_stopping": False,
        },
        "source_manifest_sha256": hashlib.sha256(b"[]").hexdigest(),
        "pair_feature_names": pair_feature_names(hatch_texture.texture_feature_names("base"), "distribution"),
        "training_ids": ["fit-q"],
        "training_documents": ["fit-doc"],
    }

    names = validate_h2_metadata(metadata, fit_manifest)

    assert len(names) == 8 * len(hatch_texture.texture_feature_names("base"))
    _validate_h2_coverage(metadata, [{"id": "fit-q", "document_id": "fit-doc"}])
    with pytest.raises(ValueError, match="query IDs"):
        _validate_h2_coverage(metadata, [{"id": "other-q", "document_id": "fit-doc"}])
    metadata["seed"] = 13
    with pytest.raises(ValueError, match="stride and seed"):
        validate_h2_metadata(metadata, fit_manifest)


def test_h2_cache_dimensions_and_counts_must_match_metadata() -> None:
    x = np.zeros((3, 2), dtype=np.float32)
    y = np.array([1, 0, 1], dtype=np.int32)
    metadata = {
        "pair_feature_names": ["f0", "f1"],
        "training_coverage": {"total_samples": 3, "positive_samples": 2, "negative_samples": 1},
    }

    _validate_cache_coverage(metadata, x, y)
    with pytest.raises(ValueError, match="sample counts"):
        _validate_cache_coverage(
            {
                "pair_feature_names": ["f0", "f1"],
                "training_coverage": {
                    "total_samples": 3,
                    "positive_samples": 2,
                    "negative_samples": 2,
                },
            },
            x,
            y,
        )
