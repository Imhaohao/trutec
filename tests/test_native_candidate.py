from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate import EvaluationError, evaluate
from tools.verify_native_candidate import verify_candidate


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(root: Path) -> tuple[Path, ...]:
    ident = "q-test"
    request = {"id": ident, "image": "drawing.png", "width": 2, "height": 2,
               "query_box": [0, 0, 1, 1], "context_boxes": []}
    inputs = root / "inputs.json"
    public = root / "val.json"
    _write_json(inputs, {"schema": "hatch-matching-inputs/v1", "examples": [request]})
    public_row = {
        **request, "document_id": "doc-test", "kind": "real",
        "labels": {"positive_boxes": [[1, 0, 2, 1]], "negative_boxes": [[0, 1, 2, 2]]},
    }
    _write_json(public, {"examples": [public_row]})

    texture_model = root / "models" / "texture.joblib"
    cnn_model = root / "models" / "cnn.pt"
    texture_model.parent.mkdir(parents=True)
    texture_model.write_bytes(b"texture")
    cnn_model.write_bytes(b"cnn")
    metadata = texture_model.with_suffix(".json")
    metadata.write_text("{}", encoding="utf-8")
    source = root / "frozen.py"
    source.write_text("frozen", encoding="utf-8")
    freeze = root / "freeze.json"
    _write_json(freeze, {
        "source_sha256": {"frozen.py": _sha(source)},
        "models": [
            {"path": "models/texture.joblib", "bytes": texture_model.stat().st_size,
             "sha256": _sha(texture_model)},
            {"path": "models/cnn.pt", "bytes": cnn_model.stat().st_size,
             "sha256": _sha(cnn_model)},
        ],
        "model_metadata_sha256": _sha(metadata),
        "validation_inputs_sha256": _sha(inputs),
        "selection": {"operator": "AND", "quantile": "q10", "scale": 0.0,
                      "offset": 0.15, "clip": [0.01, 0.99], "cnn_threshold": 0.25, "tile_size": 896,
                      "overlap": 128, "device": "mps"},
    })
    prediction_dir = root / "predictions"
    prediction_dir.mkdir()
    Image.fromarray(np.asarray([[0, 255], [0, 0]], dtype=np.uint8)).save(prediction_dir / f"{ident}.png")
    _write_json(prediction_dir / "runtime.json", {
        "method": "learned-texture-cnn-fusion-v1", "operator": "AND", "elapsed_seconds": 3.0,
        "device": "mps", "inputs": str(inputs), "selection": {"quantile": "q10", "scale": 0.0,
        "offset": 0.15, "cnn_threshold": 0.25}, "tile": {"tile_size": 896, "overlap": 128},
        "processed_query_count": 1, "texture_model": "models/texture.joblib",
        "texture_model_sha256": _sha(texture_model), "cnn_model": "models/cnn.pt",
        "cnn_model_sha256": _sha(cnn_model), "examples": [{
            "id": ident, "selected_pixels": 1, "quantile": "q10", "scale": 0.0,
            "offset": 0.15, "cnn_threshold": 0.25, "texture_threshold": 0.15,
        }],
    })
    log = root / "inference.log"
    log.write_text("3.00 real\n 100 maximum resident set size\n 200 peak memory footprint\n 0 swaps\n")
    prior = root / "v2.json"
    _write_json(prior, {"summary": {
        "document_macro_iou": 0.5, "global_precision": 0.5, "global_recall": 0.5,
        "tp": 1, "fp": 1, "tn": 1, "fn": 1,
    }})
    data_root = root / "dataset"
    data_root.mkdir()
    official = root / "official-metrics.json"
    official_result = evaluate(public, data_root, prediction_dir)
    official.write_text(
        json.dumps(official_result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return root, freeze, inputs, public, data_root, prediction_dir, log, official, prior


def test_verifier_checks_frozen_run_and_emits_official_summary(tmp_path: Path) -> None:
    root, freeze, inputs, public, data_root, predictions, log, official, prior = _fixture(tmp_path)
    output = tmp_path / "proof" / "integrity.json"
    official_bytes = official.read_bytes()
    proof = verify_candidate(
        root, freeze, inputs, public, data_root, predictions, log, official, prior, output,
    )
    assert proof["all_request_prediction_report_ids_match"]
    assert proof["official_metrics"]["tp"] == 1
    assert proof["false_discovery_rate"] == 0
    assert proof["false_positive_rate"] == 0
    assert proof["all_strict_gates_met"]
    assert proof["runtime"]["swaps"] == 0
    assert official.read_bytes() == official_bytes
    assert proof["official_metrics_sha256"] == hashlib.sha256(official_bytes).hexdigest()


def test_verifier_rejects_nonbinary_native_png(tmp_path: Path) -> None:
    root, freeze, inputs, public, data_root, predictions, log, official, prior = _fixture(tmp_path)
    Image.fromarray(np.asarray([[0, 1], [0, 0]], dtype=np.uint8)).save(predictions / "q-test.png")
    with pytest.raises(EvaluationError, match="0/255"):
        verify_candidate(
            root, freeze, inputs, public, data_root, predictions, log, official, prior,
            tmp_path / "proof.json",
        )


def test_verifier_rejects_input_hash_or_runtime_configuration_drift(tmp_path: Path) -> None:
    root, freeze, inputs, public, data_root, predictions, log, official, prior = _fixture(tmp_path)
    inputs.write_text(inputs.read_text() + " ", encoding="utf-8")
    with pytest.raises(EvaluationError, match="input hash"):
        verify_candidate(
            root, freeze, inputs, public, data_root, predictions, log, official, prior,
            tmp_path / "proof.json",
        )


def test_verifier_rejects_mutated_official_metrics_without_rewriting_them(tmp_path: Path) -> None:
    root, freeze, inputs, public, data_root, predictions, log, official, prior = _fixture(tmp_path)
    payload = json.loads(official.read_text(encoding="utf-8"))
    payload["summary"]["tp"] += 1
    official.write_text(json.dumps(payload), encoding="utf-8")
    mutated_bytes = official.read_bytes()
    with pytest.raises(EvaluationError, match="differ from unchanged evaluator"):
        verify_candidate(
            root, freeze, inputs, public, data_root, predictions, log, official, prior,
            tmp_path / "proof.json",
        )
    assert official.read_bytes() == mutated_bytes


def test_optional_cnn_cache_uses_public_schema_and_rejects_gate_mismatch(tmp_path: Path) -> None:
    root, freeze, inputs, public, data_root, predictions, log, official, prior = _fixture(tmp_path)
    cache = root / "cnn-cache"
    cache.mkdir()
    np.savez_compressed(
        cache / "q-test.npz", cnn_probability=np.asarray([0.5, 0.1, 0.4], np.float32),
        y=np.asarray([0, 1, 1], np.int32), x=np.asarray([1, 0, 1], np.int32),
    )
    proof = verify_candidate(
        root, freeze, inputs, public, data_root, predictions, log, official, prior,
        tmp_path / "proof.json", cache,
    )
    assert proof["cnn_gate_subset_check"]["reviewed_cache_coordinates"] == 3
    np.savez_compressed(
        cache / "q-test.npz", cnn_probability=np.asarray([0.1, 0.1, 0.4], np.float32),
        y=np.asarray([0, 1, 1], np.int32), x=np.asarray([1, 0, 1], np.int32),
    )
    with pytest.raises(EvaluationError, match="violate CNN threshold gate"):
        verify_candidate(
            root, freeze, inputs, public, data_root, predictions, log, official, prior,
            tmp_path / "proof.json", cache,
        )
