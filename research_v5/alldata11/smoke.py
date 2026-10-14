"""Run v11 and v7 on one development drawing and count mask pixels that differ."""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import runtime  # noqa: F401

import numpy as np
from PIL import Image

from runtime import DEVELOPMENT_MANIFEST, HERE, MODEL_PATH, ROOT, log, require

Image.MAX_IMAGE_PIXELS = None
DATA_ROOT = ROOT / "dataset"
SMOKE_INPUTS = HERE / "smoke-inputs.json"
V11_DIR = HERE / "masks-v11"
V7_DIR = HERE / "masks-v7"
THRESHOLD = "0.25"


def compare_one_drawing() -> dict:
    image, requests = _smallest_drawing()
    _write_inputs(requests)
    v11 = _invoke(HERE / "inference_v11.py", V11_DIR, ["--stage2", str(MODEL_PATH)])
    v7 = _invoke(ROOT / "research_v5" / "combo" / "inference_v7.py", V7_DIR, [])
    diff = _differing_pixels(V11_DIR, V7_DIR)
    payload = {"image": image, "queries": [row["id"] for row in requests], "threshold": float(THRESHOLD), "v11": v11, "v7": v7, "diff": diff}
    (HERE / "smoke-diff.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log(f"smoke {image} pixels={diff['pixels']} differing={diff['differing_pixels']}")
    return payload


def _smallest_drawing() -> tuple[str, list[dict]]:
    payload = json.loads(DEVELOPMENT_MANIFEST.read_text(encoding="utf-8"))
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
    body = {"schema": "hatch-matching-challenge/v1", "examples": requests}
    SMOKE_INPUTS.write_text(json.dumps(body, indent=2), encoding="utf-8")


def _invoke(script, output: Path, extra: list[str]) -> dict:
    command = [
        sys.executable,
        str(script),
        "--inputs",
        str(SMOKE_INPUTS),
        "--data-root",
        str(DATA_ROOT),
        "--output-dir",
        str(output),
        "--threshold",
        THRESHOLD,
        "--cleaner",
        "g2",
        *extra,
    ]
    log(" ".join(shlex.quote(part) for part in command))
    subprocess.run(command, check=True, cwd=ROOT, env=_env())
    runtime = json.loads((output / "runtime.json").read_text(encoding="utf-8"))
    return {"command": command, "runtime": runtime}


def _env() -> dict:
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[name] = "4"
    return env


def _differing_pixels(left_dir, right_dir) -> dict:
    names = sorted(path.name for path in left_dir.glob("*.png"))
    require(names, "v11 wrote no masks")
    require(names == sorted(path.name for path in right_dir.glob("*.png")), "mask names differ")
    total = differ = 0
    for name in names:
        total, differ = _one_mask(left_dir / name, right_dir / name, total, differ)
    return {"files": names, "pixels": total, "differing_pixels": differ}


def _one_mask(left_path, right_path, total: int, differ: int) -> tuple[int, int]:
    left = np.asarray(Image.open(left_path))
    right = np.asarray(Image.open(right_path))
    require(left.shape == right.shape, f"shape {left_path.name}")
    return total + int(left.size), differ + int(np.count_nonzero(left != right))
