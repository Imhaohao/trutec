from __future__ import annotations

import json
import hashlib
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate import EvaluationError
import tools.analyze_research_pareto as analysis


def _row(identifier: str, document: str, positive_mask: str, known_mask: str, blank_mask: str) -> dict[str, object]:
    return {
        "id": identifier,
        "document_id": document,
        "kind": "real",
        "image": "input.png",
        "width": 4,
        "height": 2,
        "query_box": [0, 0, 1, 1],
        "context_boxes": [[3, 1, 4, 2]],
        "labels": {
            "positive_mask": positive_mask,
            "known_mask": known_mask,
            "blank_mask": blank_mask,
        },
    }


def _write_fixture(root: Path) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    masks = root / "masks"
    masks.mkdir(parents=True)
    positive = np.zeros((2, 4), dtype=np.uint8)
    positive[0, 1] = 255
    positive[1, 1] = 255
    known = np.zeros((2, 4), dtype=np.uint8)
    known[:, 1:3] = 255
    known[0, 0] = 255
    known[1, 3] = 255
    blank = np.zeros((2, 4), dtype=np.uint8)
    blank[:, 2] = 255
    for name, array in (("positive.png", positive), ("known.png", known), ("blank.png", blank)):
        Image.fromarray(array).save(masks / name)
    row_one = _row("q-one", "doc-one", "masks/positive.png", "masks/known.png", "masks/blank.png")
    row_two = _row("q-two", "doc-two", "masks/positive.png", "masks/known.png", "masks/blank.png")
    rows = [row_one, row_two]
    manifest = root / "development.json"
    manifest.write_text(json.dumps({"examples": rows}), encoding="utf-8")
    return rows, {"positive": positive != 0, "known": known != 0, "blank": blank != 0}


def _write_cache(
    cache_dir: Path,
    row: dict[str, object],
    masks: dict[str, np.ndarray],
    probabilities: np.ndarray,
    *,
    with_coordinates: bool = True,
    label_override: np.ndarray | None = None,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    known = masks["known"].copy()
    x0, y0, x1, y1 = row["query_box"]  # type: ignore[misc]
    known[y0:y1, x0:x1] = False
    for x0, y0, x1, y1 in row["context_boxes"]:  # type: ignore[assignment]
        known[y0:y1, x0:x1] = False
    y, x = np.nonzero(known)
    arrays: dict[str, np.ndarray] = {
        "probability": probabilities.astype(np.float32),
        "positive": (masks["positive"][known] if label_override is None else label_override),
        "blank": masks["blank"][known],
    }
    if with_coordinates:
        arrays.update(y=y.astype(np.int32), x=x.astype(np.int32))
    np.savez_compressed(cache_dir / f"{row['id']}.npz", **arrays)


def _records(rows: list[dict[str, object]], probability: np.ndarray) -> list[dict[str, object]]:
    return [
        {
            "id": row["id"],
            "document_id": row["document_id"],
            "probability": probability.copy(),
            "positive": np.asarray([True, False]),
            "blank": np.asarray([False, True]),
        }
        for row in rows
    ]


def test_full_threshold_grid_uses_known_row_major_pixels_and_expected_counts(tmp_path: Path) -> None:
    rows, masks = _write_fixture(tmp_path)
    crop = tmp_path / "crop"
    distribution = tmp_path / "distribution"
    cnn = tmp_path / "cnn"
    probabilities = np.asarray([0.8, 0.2, 0.8, 0.2], dtype=np.float32)
    for row in rows:
        _write_cache(crop, row, masks, probabilities)
        _write_cache(distribution, row, masks, probabilities)
        _write_cache(cnn, row, masks, np.asarray([0.9, 0.1, 0.9, 0.1], dtype=np.float32))

    loaded = analysis.load_model_cache(rows, tmp_path, crop, "texture_probability", "crop")
    cnn_loaded = analysis.load_model_cache(rows, tmp_path, cnn, "cnn_probability", "CNN")
    result = analysis.analyze_model(rows, loaded, cnn_loaded)

    assert result["and"]["document_macro_iou"][50][50] == 1.0
    assert result["and"]["global_recall"][50][50] == 1.0
    assert result["and"]["global_precision"][50][50] == 1.0
    assert len(result["and"]["document_macro_iou"]) == 101
    assert all(len(line) == 101 for line in result["and"]["global_recall"])
    assert len(result["texture_only"]["global_recall"]) == 101
    assert result["and"]["selections"]["joint_gate_count"] > 0
    assert result["and"]["selections"]["best_joint_gate"] is not None
    best_fdr = result["and"]["selections"]["macro_iou_best_at_fdr_le_0_05"]
    assert best_fdr is not None
    assert best_fdr["false_discovery_rate"] <= 0.05
    assert best_fdr["false_positive_rate"] == pytest.approx(
        best_fdr["fp"] / (best_fdr["fp"] + best_fdr["tn"])
    )
    assert best_fdr["tn"] >= 0


def test_cache_ids_coordinates_and_masks_must_match_manifest(tmp_path: Path) -> None:
    rows, masks = _write_fixture(tmp_path)
    cache = tmp_path / "cache"
    probabilities = np.asarray([0.8, 0.2, 0.8, 0.2], dtype=np.float32)
    for row in rows:
        _write_cache(cache, row, masks, probabilities)
    (cache / "q-extra.npz").write_bytes((cache / "q-one.npz").read_bytes())
    with pytest.raises(EvaluationError, match="IDs mismatch"):
        analysis.load_model_cache(rows, tmp_path, cache, "texture_probability", "crop")

    (cache / "q-extra.npz").unlink()
    first_path = cache / "q-one.npz"
    data = np.load(first_path, allow_pickle=False)
    arrays = {key: np.asarray(data[key]) for key in data.files}
    data.close()
    arrays["x"] = arrays["x"][::-1].copy()
    np.savez_compressed(first_path, **arrays)
    with pytest.raises(EvaluationError, match="coordinates"):
        analysis.load_model_cache(rows, tmp_path, cache, "texture_probability", "crop")


def test_unknown_pixels_and_label_order_are_rejected(tmp_path: Path) -> None:
    rows, masks = _write_fixture(tmp_path)
    cache = tmp_path / "cache"
    wrong_labels = np.asarray([False, True, False, True])
    for row in rows:
        _write_cache(cache, row, masks, np.asarray([0.8, 0.2, 0.8, 0.2]), label_override=wrong_labels)
    with pytest.raises(EvaluationError, match="labels do not match reviewed"):
        analysis.load_model_cache(rows, tmp_path, cache, "texture_probability", "crop")

    _write_cache(cache, rows[0], masks, np.asarray([0.8, 0.2, 0.1]))
    with pytest.raises(EvaluationError, match="length does not match known"):
        analysis.load_model_cache(rows, tmp_path, cache, "texture_probability", "crop")


def test_training_metadata_must_be_subset_of_fit_and_disjoint_from_dev(tmp_path: Path) -> None:
    dev_rows = [
        {"id": "q-dev", "document_id": "doc-dev", "kind": "real", "image": "x.png", "width": 1, "height": 1,
         "query_box": [0, 0, 1, 1], "context_boxes": [], "labels": {"positive_boxes": [[0, 0, 1, 1]]}},
    ]
    fit_rows = [
        {"id": "q-fit", "document_id": "doc-fit", "kind": "real", "image": "x.png", "width": 1, "height": 1,
         "query_box": [0, 0, 1, 1], "context_boxes": [], "labels": {"positive_boxes": [[0, 0, 1, 1]]}},
    ]
    fit_path = tmp_path / "fit.json"
    fit_path.write_text(json.dumps({"examples": fit_rows}), encoding="utf-8")
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps({"training_ids": ["q-dev"], "training_documents": ["doc-dev"]}), encoding="utf-8")
    with pytest.raises(EvaluationError, match="outside the fit manifest"):
        analysis._validate_fit_metadata(fit_path, dev_rows, {"crop": metadata_path}, tmp_path)


def _metadata_manifests(tmp_path: Path) -> tuple[Path, list[dict[str, object]], list[dict[str, object]]]:
    dev_rows = [
        {"id": "q-dev", "document_id": "doc-dev", "kind": "real", "image": "x.png", "width": 1, "height": 1,
         "query_box": [0, 0, 1, 1], "context_boxes": [], "labels": {"positive_boxes": [[0, 0, 1, 1]]}},
    ]
    fit_rows = [
        {"id": "q-fit-one", "document_id": "doc-fit", "kind": "real", "image": "x.png", "width": 1, "height": 1,
         "query_box": [0, 0, 1, 1], "context_boxes": [], "labels": {"positive_boxes": [[0, 0, 1, 1]]}},
        {"id": "q-fit-two", "document_id": "doc-fit", "kind": "real", "image": "x.png", "width": 1, "height": 1,
         "query_box": [0, 0, 1, 1], "context_boxes": [], "labels": {"positive_boxes": [[0, 0, 1, 1]]}},
    ]
    fit_path = tmp_path / "fit.json"
    fit_path.write_text(json.dumps({"examples": fit_rows}), encoding="utf-8")
    return fit_path, fit_rows, dev_rows


def _trainer_metadata(fit_path: Path, *, query_ids: list[str] | None = None,
                      document_ids: list[str] | None = None) -> dict[str, object]:
    digest = hashlib.sha256(fit_path.read_bytes()).hexdigest()
    return {
        "manifest_sha256": digest,
        "fit_document_ids": ["doc-fit"],
        "selected_query_ids": query_ids if query_ids is not None else ["q-fit-one", "q-fit-one", "q-fit-two"],
        "selected_document_ids": document_ids if document_ids is not None else ["doc-fit", "doc-fit", "doc-fit"],
        "optimizer_updates": 2,
    }


def _write_metadata(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_trainer_metadata_accepts_aligned_repeated_pair_ids(tmp_path: Path) -> None:
    fit_path, _, dev_rows = _metadata_manifests(tmp_path)
    metadata_path = _write_metadata(tmp_path, _trainer_metadata(fit_path))

    analysis._validate_fit_metadata(fit_path, dev_rows, {"P1": metadata_path}, tmp_path)


@pytest.mark.parametrize(
    ("query_ids", "document_ids"),
    [(["q-outside"], ["doc-fit"]), (["q-fit-one"], ["doc-outside"])],
)
def test_trainer_metadata_rejects_out_of_fit_query_or_document(
    tmp_path: Path, query_ids: list[str], document_ids: list[str],
) -> None:
    fit_path, _, dev_rows = _metadata_manifests(tmp_path)
    metadata_path = _write_metadata(
        tmp_path, _trainer_metadata(fit_path, query_ids=query_ids, document_ids=document_ids),
    )

    with pytest.raises(EvaluationError, match="outside the fit manifest"):
        analysis._validate_fit_metadata(fit_path, dev_rows, {"P0": metadata_path}, tmp_path)


def test_trainer_metadata_rejects_misaligned_fit_pair(tmp_path: Path) -> None:
    fit_path, _, dev_rows = _metadata_manifests(tmp_path)
    metadata_path = _write_metadata(
        tmp_path,
        _trainer_metadata(fit_path, query_ids=["q-fit-one"], document_ids=["doc-fit-other"]),
    )

    with pytest.raises(EvaluationError, match="pair is outside the fit manifest"):
        analysis._validate_fit_metadata(fit_path, dev_rows, {"P0": metadata_path}, tmp_path)


def test_trainer_metadata_rejects_wrong_fit_manifest_hash(tmp_path: Path) -> None:
    fit_path, _, dev_rows = _metadata_manifests(tmp_path)
    payload = _trainer_metadata(fit_path)
    payload["manifest_sha256"] = "0" * 64
    metadata_path = _write_metadata(tmp_path, payload)

    with pytest.raises(EvaluationError, match="manifest_sha256 does not match"):
        analysis._validate_fit_metadata(fit_path, dev_rows, {"P0": metadata_path}, tmp_path)


def test_trainer_metadata_rejects_partial_schema(tmp_path: Path) -> None:
    fit_path, _, dev_rows = _metadata_manifests(tmp_path)
    payload = _trainer_metadata(fit_path)
    del payload["fit_document_ids"]
    metadata_path = _write_metadata(tmp_path, payload)

    with pytest.raises(EvaluationError, match="trainer metadata is missing fields"):
        analysis._validate_fit_metadata(fit_path, dev_rows, {"P0": metadata_path}, tmp_path)


def test_trainer_metadata_rejects_unequal_selected_pair_arrays(tmp_path: Path) -> None:
    fit_path, _, dev_rows = _metadata_manifests(tmp_path)
    metadata_path = _write_metadata(
        tmp_path,
        _trainer_metadata(fit_path, query_ids=["q-fit-one", "q-fit-two"], document_ids=["doc-fit"]),
    )

    with pytest.raises(EvaluationError, match="must be nonempty and aligned"):
        analysis._validate_fit_metadata(fit_path, dev_rows, {"P0": metadata_path}, tmp_path)


@pytest.mark.parametrize("query_ids", [["q-fit-one", 2], ["q-fit-one", ""]])
def test_trainer_metadata_rejects_malformed_selected_ids(
    tmp_path: Path, query_ids: list[object],
) -> None:
    fit_path, _, dev_rows = _metadata_manifests(tmp_path)
    payload = _trainer_metadata(fit_path)
    payload["selected_query_ids"] = query_ids
    metadata_path = _write_metadata(tmp_path, payload)

    with pytest.raises(EvaluationError, match="selected_query_ids must be a list of nonempty strings"):
        analysis._validate_fit_metadata(fit_path, dev_rows, {"P0": metadata_path}, tmp_path)


@pytest.mark.parametrize(
    ("training_ids", "error"),
    [(["q-fit-one", "q-fit-one"], "duplicates"), (["q-fit-one", 3], "list of strings")],
)
def test_canonical_training_metadata_still_rejects_malformed_ids(
    tmp_path: Path, training_ids: list[object], error: str,
) -> None:
    fit_path, _, dev_rows = _metadata_manifests(tmp_path)
    metadata_path = _write_metadata(
        tmp_path,
        {"training_ids": training_ids, "training_documents": ["doc-fit"]},
    )

    with pytest.raises(EvaluationError, match=error):
        analysis._validate_fit_metadata(fit_path, dev_rows, {"P0": metadata_path}, tmp_path)


def test_canonical_training_metadata_accepts_ancillary_manifest_hash(tmp_path: Path) -> None:
    fit_path, _, dev_rows = _metadata_manifests(tmp_path)
    metadata_path = _write_metadata(
        tmp_path,
        {"training_ids": ["q-fit-one"], "training_documents": ["doc-fit"],
         "manifest_sha256": hashlib.sha256(fit_path.read_bytes()).hexdigest()},
    )

    analysis._validate_fit_metadata(fit_path, dev_rows, {"P0": metadata_path}, tmp_path)


def test_mixed_canonical_and_trainer_schemas_are_rejected(tmp_path: Path) -> None:
    fit_path, _, dev_rows = _metadata_manifests(tmp_path)
    payload = _trainer_metadata(fit_path)
    payload.update({"training_ids": ["q-fit-one", "q-fit-one"], "training_documents": ["doc-fit"]})
    metadata_path = _write_metadata(tmp_path, payload)

    with pytest.raises(EvaluationError, match="mixes canonical and trainer schemas"):
        analysis._validate_fit_metadata(fit_path, dev_rows, {"P0": metadata_path}, tmp_path)


def test_trainer_metadata_requires_exact_fit_documents(tmp_path: Path) -> None:
    fit_path, _, dev_rows = _metadata_manifests(tmp_path)
    payload = _trainer_metadata(fit_path)
    payload["fit_document_ids"] = ["doc-fit", "doc-extra"]
    metadata_path = _write_metadata(tmp_path, payload)

    with pytest.raises(EvaluationError, match="fit_document_ids do not match"):
        analysis._validate_fit_metadata(fit_path, dev_rows, {"P0": metadata_path}, tmp_path)


def test_grids_match_brute_force_for_float32_thresholds_and_empty_targets() -> None:
    rows = [
        {"id": "q-a1", "document_id": "doc-a"},
        {"id": "q-a2", "document_id": "doc-a"},
        {"id": "q-b", "document_id": "doc-b"},
    ]
    texture = [
        {"probability": np.asarray([0.29, 0.38, 0.71, 0.99], np.float32),
         "positive": np.asarray([True, False, False, True]), "blank": np.asarray([False, False, True, False])},
        {"probability": np.asarray([0.0, 0.51], np.float32),
         "positive": np.asarray([False, False]), "blank": np.asarray([True, False])},
        {"probability": np.asarray([0.30, 0.38, 0.70], np.float32),
         "positive": np.asarray([True, True, True]), "blank": np.asarray([False, False, False])},
    ]
    cnn = [
        {"probability": np.asarray([0.30, 0.38, 0.70, 0.99], np.float32)},
        {"probability": np.asarray([0.0, 0.50], np.float32)},
        {"probability": np.asarray([0.31, 0.38, 0.69], np.float32)},
    ]
    texture_records = [
        {**record, "id": row["id"], "document_id": row["document_id"]}
        for row, record in zip(rows, texture)
    ]
    cnn_records = [
        {**record, "id": row["id"], "document_id": row["document_id"],
         "positive": truth["positive"], "blank": truth["blank"]}
        for row, record, truth in zip(rows, cnn, texture)
    ]
    result = analysis.analyze_model(rows, texture_records, cnn_records)

    for texture_index, texture_grid_value in enumerate(analysis.THRESHOLDS):
        texture_threshold = float(texture_grid_value)
        for cnn_index, cnn_grid_value in enumerate(analysis.THRESHOLDS):
            cnn_threshold = float(cnn_grid_value)
            expected = []
            for row, t_record, c_record in zip(rows, texture_records, cnn_records):
                positive = t_record["positive"]
                selected = (
                    (t_record["probability"] >= texture_threshold)
                    & (c_record["probability"] >= cnn_threshold)
                )
                tp = int((selected & positive).sum())
                fp = int((selected & ~positive).sum())
                fn = int((~selected & positive).sum())
                tn = int((~selected & ~positive).sum())
                blank_fp = int((selected & t_record["blank"]).sum())
                expected.append({
                    "id": row["id"], "document_id": row["document_id"],
                    "metrics": analysis.metrics_from_counts(
                        tp, fp, fn, tn, blank_fp, int(t_record["blank"].sum())
                    ),
                })
            expected_summary = analysis.aggregate(expected)
            actual = result["and"]
            for key, matrix_key in (
                ("document_macro_iou", "document_macro_iou"),
                ("global_recall", "global_recall"),
                ("global_precision", "global_precision"),
            ):
                expected_value = expected_summary[key]
                actual_value = actual[matrix_key][texture_index][cnn_index]
                if expected_value is None:
                    assert actual_value is None
                else:
                    assert actual_value == pytest.approx(expected_value)

    no_predictions = analysis._candidate(
        rows, texture_records,
        np.zeros((3, 101, 101), dtype=np.int64),
        np.zeros((3, 101, 101), dtype=np.int64),
        np.zeros((3, 101, 101), dtype=np.int64), 100, 100,
    )
    assert no_predictions["false_discovery_rate"] is None
    assert no_predictions["false_positive_rate"] == 0.0
    assert no_predictions["tn"] == sum(len(item["positive"]) - int(item["positive"].sum()) for item in texture_records)


def test_training_metadata_without_fit_manifest_raises_evaluation_error(tmp_path: Path) -> None:
    with pytest.raises(EvaluationError, match="--fit-manifest is required"):
        analysis._validate_fit_metadata(None, [], {"crop": tmp_path / "metadata.json"}, tmp_path)


def test_aggregate_selection_counts_positive_only_from_reviewed_pixels(tmp_path: Path) -> None:
    rows = [
        {"id": "q-positive-only", "document_id": "doc-one", "kind": "real",
         "width": 3, "height": 1, "query_box": [0, 0, 1, 1], "context_boxes": [],
         "labels": {"positive_boxes": [[0, 0, 3, 1]]}},
        {"id": "q-mixed", "document_id": "doc-two", "kind": "real",
         "width": 3, "height": 1, "query_box": [0, 0, 1, 1], "context_boxes": [],
         "labels": {"positive_boxes": [[1, 0, 2, 1]], "negative_boxes": [[2, 0, 3, 1]]}},
    ]
    records = []
    expected_flags = []
    for row in rows:
        positive, known, blank, _ = analysis.reviewed_domains(row, tmp_path)
        positive_vector = positive[known]
        records.append({
            "id": row["id"], "document_id": row["document_id"],
            "positive": positive_vector, "blank": blank[known],
        })
        expected_flags.append(bool(positive_vector.any() and positive_vector.all()))

    summary = analysis._aggregate_selection(
        rows, records, np.asarray([2, 1]), np.asarray([0, 0]), np.asarray([0, 0])
    )
    assert expected_flags == [True, False]
    assert summary["positive_only_examples"] == sum(expected_flags) == 1

