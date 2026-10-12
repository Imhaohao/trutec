from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate import EvaluationError
from tools.verify_dino_fit import _pairs, _runtime, _verify_audit, _verify_model


def test_runtime_reports_only_process_attributed_swap_count(tmp_path: Path) -> None:
    log = tmp_path / "time.log"
    log.write_text(
        "2.50 real 1.00 user 0.25 sys\n 100 maximum resident set size\n 200 peak memory footprint\n 0 swaps\n",
        encoding="utf-8",
    )
    proof = _runtime(log)
    assert proof["process_swaps"] == 0
    assert proof["global_swap_claim"] is None
    assert proof["process_attributed"] is True


@pytest.mark.parametrize("content", [
    "2.0 real 1.0 user 0.1 sys\n 100 maximum resident set size\n 0 swaps\n 0 swaps\n",
    "2.0 real 1.0 user 0.1 sys\n 100 maximum resident set size\n",
    "global swapouts: 0\n",
    "1.0 real 0.5 user 0.1 sys\n 10 maximum resident set size\n 0 swaps\n2.0 real 1.0 user 0.1 sys\n 10 maximum resident set size\n 0 swaps\n",
])
def test_runtime_rejects_missing_or_ambiguous_process_measurements(tmp_path: Path, content: str) -> None:
    log = tmp_path / "time.log"
    log.write_text(content, encoding="utf-8")
    with pytest.raises(EvaluationError):
        _runtime(log)


@pytest.mark.parametrize("footprint,swaps", [(200, 1), (16 * 1024**3, 0), (0, 0)])
def test_runtime_rejects_failed_fit_resource_gates(tmp_path: Path, footprint: int, swaps: int) -> None:
    log = tmp_path / "time.log"
    log.write_text(
        f"2.5 real 1.0 user 0.2 sys\n 100 maximum resident set size\n"
        f" {footprint} peak memory footprint\n {swaps} swaps\n",
        encoding="utf-8",
    )
    with pytest.raises(EvaluationError, match="resource gate"):
        _runtime(log)


def test_coordinate_parser_rejects_float_and_out_of_bounds_coordinates() -> None:
    row = {"id": "q", "height": 20, "width": 30}
    assert np.array_equal(_pairs([[2, 3]], "positive", row), np.asarray([[2, 3]], np.int32))
    with pytest.raises(EvaluationError):
        _pairs([[2.0, 3.0]], "positive", row)
    with pytest.raises(EvaluationError):
        _pairs([[20, 3]], "positive", row)


def test_audit_recomputes_samples_and_enforces_coverage(tmp_path: Path) -> None:
    from dino_matching import sample_pure_patch_pixels

    positive = np.zeros((28, 28), dtype=bool)
    known = np.ones_like(positive)
    positive[:14, :14] = True
    row = {"id": "q", "document_id": "d", "height": 28, "width": 28,
           "labels": {}}
    pos, neg = sample_pure_patch_pixels(positive, known, 45, 32)
    p_idx, n_idx = pos // 14, neg // 14
    audit = [{
            "query_id": "q", "document_id": "d", "seed": 45,
            "positive_native_pixels": pos.tolist(), "negative_native_pixels": neg.tolist(),
            "positive_patch_indices": p_idx.tolist(), "negative_patch_indices": n_idx.tolist(),
            "class_counts": {"positive": len(pos), "negative": len(neg)},
            "class_weights": {"positive": 0.5 / len(pos), "negative": 0.5 / len(neg)},
            "positive_patch_centers_yx": [[min(int(y) * 14 + 7, 27), min(int(x) * 14 + 7, 27)] for y, x in p_idx],
            "negative_patch_centers_yx": [[min(int(y) * 14 + 7, 27), min(int(x) * 14 + 7, 27)] for y, x in n_idx],
        }]
    from train_dino_matching import build_metadata
    coverage = {
        "sampling_audit": audit, "sampled_query_ids": ["q"], "dual_class_query_ids": ["q"],
        "positive_only_query_ids": [], "skipped_query_ids": [], "positive_samples": len(pos),
        "negative_samples": len(neg), "samples": len(pos) + len(neg), "queries": 1, "documents": 1,
    }
    metadata = build_metadata([row], "test", {}, coverage, np.concatenate((np.ones(len(pos)), np.zeros(len(neg)))), 0.1)
    # Reviewed-domain loading is the trust boundary, so supply one minimal box-only query
    # through a monkeypatch rather than creating any unrelated dataset files.
    from tools import verify_dino_fit
    original = verify_dino_fit.reviewed_domains
    verify_dino_fit.reviewed_domains = lambda _row, _root: (positive, known, np.zeros_like(known), {})
    try:
        assert _verify_audit([row], tmp_path, metadata) == {"positive": len(pos), "negative": len(neg)}
        metadata["sampling_audit"][0]["negative_native_pixels"][0] = [1, 1]
        with pytest.raises(EvaluationError, match="Recomputed"):
            _verify_audit([row], tmp_path, metadata)
    finally:
        verify_dino_fit.reviewed_domains = original


def test_positive_only_audit_accepts_empty_negative_vectors(tmp_path: Path) -> None:
    from dino_matching import sample_pure_patch_pixels
    from tools import verify_dino_fit

    positive = np.zeros((14, 14), dtype=bool)
    positive[:4, :4] = True
    known = positive.copy()
    pos, neg = sample_pure_patch_pixels(positive, known, 45, 32)
    indices = pos // 14
    record = {
        "query_id": "q", "document_id": "d", "seed": 45,
        "positive_native_pixels": pos.tolist(), "negative_native_pixels": [],
        "positive_patch_indices": indices.tolist(), "negative_patch_indices": [],
        "class_counts": {"positive": len(pos), "negative": 0},
        "class_weights": {"positive": 1.0 / len(pos), "negative": 0.0},
        "positive_patch_centers_yx": [[7, 7]], "negative_patch_centers_yx": [],
    }
    row = {"id": "q", "document_id": "d", "height": 14, "width": 14, "labels": {}}
    coverage = {"sampling_audit": [record], "sampled_query_ids": ["q"], "dual_class_query_ids": [],
                "positive_only_query_ids": ["q"], "skipped_query_ids": [], "positive_samples": len(pos),
                "negative_samples": 0, "samples": len(pos), "queries": 1, "documents": 1}
    original = verify_dino_fit.reviewed_domains
    verify_dino_fit.reviewed_domains = lambda _row, _root: (positive, known, np.zeros_like(known), {})
    try:
        assert len(neg) == 0
        from train_dino_matching import build_metadata
        metadata = build_metadata([row], "test", {}, coverage, np.ones(len(pos)), 0.1)
        assert _verify_audit([row], tmp_path, metadata) == {
            "positive": len(pos), "negative": 0}
    finally:
        verify_dino_fit.reviewed_domains = original


def test_verify_model_accepts_saved_small_hist_gradient_boosting(tmp_path: Path) -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from dino_matching import FEATURE_SIGNATURE, MODEL_PARAMS, save_dino_model
    from tools.dino_patch_support import FIT_MANIFEST_SHA256

    rng = np.random.default_rng(7)
    x = rng.normal(size=(20, 769)).astype(np.float32)
    y = np.asarray([0, 1] * 10)
    classifier = HistGradientBoostingClassifier(**MODEL_PARAMS).fit(x, y)
    path = tmp_path / "model.joblib"
    save_dino_model(classifier, {"feature_signature": FEATURE_SIGNATURE,
                                 "classifier_params": MODEL_PARAMS,
                                 "fit_manifest_sha256": FIT_MANIFEST_SHA256}, path)
    model, metadata = _verify_model(path)
    assert model.n_features_in_ == 769
    assert metadata["model_signature"]
