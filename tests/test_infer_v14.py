"""CLI and path checks for the frozen v14 entry point."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

INFER = ROOT / "infer_v14.py"
MODEL_NAMES = (
    "stage1-alldata-v12.joblib",
    "stage1-alldata-v12.json",
    "stage2-structure-hgb-v14.joblib",
    "stage2-structure-hgb-v14.json",
)
MISSING_ARGUMENTS = (
    [],
    ["--data-root", "dataset", "--output-dir", "predictions"],
    ["--inputs", "requests.json", "--output-dir", "predictions"],
    ["--inputs", "requests.json", "--data-root", "dataset"],
)


@pytest.mark.parametrize("argv", MISSING_ARGUMENTS)
def test_cli_rejects_missing_arguments(argv: list[str]) -> None:
    import infer_v14

    with pytest.raises(SystemExit) as caught:
        infer_v14.parse_args(argv)
    assert caught.value.code != 0
    completed = subprocess.run([sys.executable, str(INFER), *argv], capture_output=True, text=True, check=False)
    assert completed.returncode == 2


def test_requests_are_stripped_of_label_fields(tmp_path: Path) -> None:
    import infer_v14

    path = tmp_path / "inputs.json"
    path.write_text(json.dumps({"examples": [_labeled_request()]}), encoding="utf-8")
    loaded = infer_v14.load_requests(path)
    assert loaded == [_label_free_request()]


def test_missing_model_files_are_reported(tmp_path: Path) -> None:
    import infer_v14

    with pytest.raises(FileNotFoundError):
        infer_v14.model_paths(tmp_path)


def test_models_load_from_model_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import numpy as np

    import infer_v14

    model_dir = _copy_models(tmp_path)
    seen: dict[str, object] = {}
    monkeypatch.setattr(infer_v14, "predict_drawing", _fake_predict(seen, np.zeros))
    output = tmp_path / "out"
    infer_v14.main(
        ["--inputs", str(_write_inputs(tmp_path)), "--data-root", str(tmp_path), "--output-dir", str(output), "--model-dir", str(model_dir)]
    )
    assert int(seen["stage1"].n_features_in_) == 184
    assert int(seen["stage2"].n_features_in_) == 144
    assert int(seen["stage2"].max_iter) == 600 and int(seen["stage2"].max_leaf_nodes) == 63
    assert seen["cleaner"] == "g2"
    assert (output / "q-test.png").is_file()
    runtime = json.loads((output / "runtime.json").read_text(encoding="utf-8"))
    assert runtime["threshold"] == 0.25
    assert Path(runtime["stage2"]) == (model_dir / "stage2-structure-hgb-v14.joblib").resolve()


def _labeled_request() -> dict:
    request = _label_free_request()
    request["document_id"] = "doc-secret"
    request["kind"] = "real"
    request["labels"] = {"positive_boxes": [[0, 0, 1, 1]], "negative_boxes": [[1, 0, 2, 2]]}
    return request


def _label_free_request() -> dict:
    return {"id": "q-test", "image": "assets/drawing.png", "width": 2, "height": 2, "query_box": [0, 0, 1, 1], "context_boxes": []}


def _write_inputs(tmp_path: Path) -> Path:
    path = tmp_path / "inputs.json"
    path.write_text(json.dumps({"examples": [_labeled_request()]}), encoding="utf-8")
    return path


def _copy_models(tmp_path: Path) -> Path:
    source = ROOT / "models" / "v14"
    dest = tmp_path / "models"
    dest.mkdir()
    for name in MODEL_NAMES:
        shutil.copyfile(source / name, dest / name)
    return dest


def _fake_predict(seen: dict[str, object], zeros):
    def predict(requests, stage1, stage2, cleaner, data_root, emit) -> None:
        seen["stage1"] = stage1
        seen["stage2"] = stage2
        seen["cleaner"] = cleaner
        for request in requests:
            emit(request, zeros((int(request["height"]), int(request["width"])), dtype="float32"))

    return predict
