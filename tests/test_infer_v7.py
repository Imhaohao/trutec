"""CLI and path checks for the frozen v7 entry point."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

INFER = ROOT / "infer_v7.py"
MODEL_NAMES = (
    "texture-hgb-crop-v1.joblib",
    "texture-hgb-crop-v1.json",
    "stage2-structure-hgb-v7.joblib",
    "stage2-structure-hgb-v7.json",
)
MISSING_ARGUMENTS = (
    [],
    ["--data-root", "dataset", "--output-dir", "predictions"],
    ["--inputs", "requests.json", "--output-dir", "predictions"],
    ["--inputs", "requests.json", "--data-root", "dataset"],
)


@pytest.mark.parametrize("argv", MISSING_ARGUMENTS)
def test_cli_rejects_missing_arguments(argv: list[str]) -> None:
    import infer_v7

    with pytest.raises(SystemExit) as caught:
        infer_v7.parse_args(argv)
    assert caught.value.code != 0
    completed = subprocess.run(
        [sys.executable, str(INFER), *argv],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2


def test_requests_are_stripped_of_label_fields(tmp_path: Path) -> None:
    import infer_v7

    path = tmp_path / "inputs.json"
    path.write_text(json.dumps({"examples": [_labeled_request()]}), encoding="utf-8")
    loaded = infer_v7.load_requests(path)
    assert loaded == [_label_free_request()]
    assert "labels" not in loaded[0]
    assert "document_id" not in loaded[0]


def test_models_load_from_model_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import numpy as np

    import infer_v7

    model_dir = _copy_models(tmp_path)
    seen: dict[str, object] = {}
    monkeypatch.setattr(infer_v7, "load_stage2", _capture_stage2(infer_v7.load_stage2, seen))
    monkeypatch.setattr(infer_v7, "predict_drawing", _fake_predict(seen, np.zeros))
    output = tmp_path / "out"
    infer_v7.main(
        [
            "--inputs",
            str(_write_inputs(tmp_path)),
            "--data-root",
            str(tmp_path),
            "--output-dir",
            str(output),
            "--model-dir",
            str(model_dir),
        ]
    )
    assert seen["stage1_path"] == (model_dir / "texture-hgb-crop-v1.joblib").resolve()
    assert seen["stage2_path"] == (model_dir / "stage2-structure-hgb-v7.joblib").resolve()
    assert seen["cleaner"] == "g2"
    assert seen["stage1"] is not None
    assert seen["stage2"] is not None
    assert (output / "q-test.png").is_file()
    runtime = json.loads((output / "runtime.json").read_text(encoding="utf-8"))
    assert runtime["threshold"] == 0.25
    assert runtime["cleaner"] == "g2"
    assert Path(runtime["model_dir"]) == model_dir.resolve()
    assert Path(runtime["stage1"]) == (model_dir / "texture-hgb-crop-v1.joblib").resolve()


def _labeled_request() -> dict:
    request = _label_free_request()
    request["document_id"] = "doc-secret"
    request["kind"] = "real"
    request["labels"] = {"positive_boxes": [[0, 0, 1, 1]], "negative_boxes": [[1, 0, 2, 2]]}
    request["expected_count"] = 4
    return request


def _label_free_request() -> dict:
    return {
        "id": "q-test",
        "image": "assets/drawing.png",
        "width": 2,
        "height": 2,
        "query_box": [0, 0, 1, 1],
        "context_boxes": [],
    }


def _write_inputs(tmp_path: Path) -> Path:
    path = tmp_path / "inputs.json"
    path.write_text(json.dumps({"examples": [_labeled_request()]}), encoding="utf-8")
    return path


def _copy_models(tmp_path: Path) -> Path:
    source = ROOT / "models" / "v7"
    dest = tmp_path / "models"
    dest.mkdir()
    for name in MODEL_NAMES:
        shutil.copyfile(source / name, dest / name)
    return dest


def _capture_stage2(real_load, seen: dict[str, object]):
    def load(path: Path):
        seen["stage2_path"] = Path(path).resolve()
        seen["stage2"] = real_load(path)
        return seen["stage2"]

    return load


def _fake_predict(seen: dict[str, object], zeros):
    def predict(requests, stage1, stage2, cleaner, data_root, emit) -> None:
        import infer_v7

        seen["stage1_path"] = Path(infer_v7.evaluate_dev.MODEL_PATH).resolve()
        seen["stage1"] = stage1
        seen["cleaner"] = cleaner
        seen["data_root"] = Path(data_root).resolve()
        for request in requests:
            height = int(request["height"])
            width = int(request["width"])
            emit(request, zeros((height, width), dtype="float32"))

    return predict
