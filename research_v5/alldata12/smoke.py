"""Run v12 inference on one development drawing."""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys

import runtime  # noqa: F401

import numpy as np
from PIL import Image

from runtime import DEVELOPMENT, HERE, ROOT, STAGE1_PATH, STAGE2_PATH, inside, log, require

Image.MAX_IMAGE_PIXELS = None
DATA_ROOT = ROOT / "dataset"
SMOKE_INPUTS = HERE / "smoke-inputs.json"
SMOKE_DIR = HERE / "masks-smoke"
THRESHOLD = "0.25"


def smoke_one_drawing() -> dict:
    image, requests = _smallest_drawing()
    _write_inputs(requests)
    invoked = _invoke()
    masks = _mask_stats(SMOKE_DIR)
    payload = {
        "image": image,
        "queries": [row["id"] for row in requests],
        "threshold": float(THRESHOLD),
        "cleaner": "g2",
        "invoke": invoked,
        "masks": masks,
    }
    inside(HERE / "smoke.json")
    (HERE / "smoke.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log(f"smoke {image} pixels={masks['pixels']} positive={masks['positive_pixels']}")
    return payload


def _smallest_drawing() -> tuple[str, list[dict]]:
    payload = json.loads(DEVELOPMENT.read_text(encoding="utf-8"))
    rows = payload["examples"] if isinstance(payload, dict) else payload
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["image"], []).append(_label_free(row))
    image = min(grouped, key=lambda name: _drawing_key(name, grouped))
    return image, grouped[image]


def _drawing_key(image: str, grouped: dict[str, list[dict]]) -> tuple:
    row = grouped[image][0]
    return (int(row["width"]) * int(row["height"]), len(grouped[image]), image)


def _label_free(row: dict) -> dict:
    keys = ("id", "image", "width", "height", "query_box", "context_boxes")
    return {key: row[key] for key in keys if key in row}


def _write_inputs(requests: list[dict]) -> None:
    inside(SMOKE_INPUTS)
    body = {"schema": "hatch-matching-challenge/v1", "examples": requests}
    SMOKE_INPUTS.write_text(json.dumps(body, indent=2), encoding="utf-8")


def _invoke() -> dict:
    inside(SMOKE_DIR)
    command = [
        sys.executable,
        str(HERE / "inference_v12.py"),
        "--inputs",
        str(SMOKE_INPUTS),
        "--data-root",
        str(DATA_ROOT),
        "--output-dir",
        str(SMOKE_DIR),
        "--threshold",
        THRESHOLD,
        "--cleaner",
        "g2",
        "--stage1",
        str(STAGE1_PATH),
        "--stage2",
        str(STAGE2_PATH),
    ]
    log(" ".join(shlex.quote(part) for part in command))
    subprocess.run(command, check=True, cwd=ROOT, env=_env())
    runtime_payload = json.loads((SMOKE_DIR / "runtime.json").read_text(encoding="utf-8"))
    return {"command": command, "runtime": runtime_payload}


def _env() -> dict:
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[name] = "4"
    return env


def _mask_stats(directory) -> dict:
    names = sorted(path.name for path in directory.glob("*.png"))
    require(names, "v12 wrote no masks")
    total = positive = 0
    for name in names:
        total, positive = _one_mask(directory / name, total, positive)
    return {"files": names, "pixels": total, "positive_pixels": positive}


def _one_mask(path, total: int, positive: int) -> tuple[int, int]:
    mask = np.asarray(Image.open(path))
    require(mask.ndim == 2, f"mask {path.name}")
    return total + int(mask.size), positive + int(np.count_nonzero(mask))
