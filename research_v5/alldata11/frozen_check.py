"""Re-hash every path recorded in frozen-pipeline-v7.json."""
from __future__ import annotations

import hashlib
import json

import runtime  # noqa: F401

from runtime import FROZEN_MANIFEST, ORIGINAL_STAGE2, ROOT, require


MODEL_FILES = {
    "stage1": ROOT / "models" / "models" / "texture-hgb-crop-v1.joblib",
    "stage2_structure": ORIGINAL_STAGE2,
}


def verify_frozen() -> dict:
    payload = json.loads(FROZEN_MANIFEST.read_text(encoding="utf-8"))
    sources = _check_sources(payload["sources"])
    models = _check_models(payload["models"])
    mismatches = sources["mismatches"] + models["mismatches"]
    require(not mismatches, "frozen hash mismatch: " + "; ".join(mismatches))
    return {
        "sources_checked": sources["checked"],
        "models_checked": models["checked"],
        "mismatches": mismatches,
    }


def _check_sources(recorded: dict) -> dict:
    mismatches = []
    for relative, expected in recorded.items():
        actual = _sha256(ROOT / relative)
        if actual != expected:
            mismatches.append(f"{relative} {actual}")
    return {"checked": len(recorded), "mismatches": mismatches}


def _check_models(recorded: dict) -> dict:
    mismatches = []
    require(set(recorded) == set(MODEL_FILES), "frozen model keys")
    for name, path in MODEL_FILES.items():
        actual = _sha256(path)
        if actual != recorded[name]:
            mismatches.append(f"{name} {actual}")
    return {"checked": len(recorded), "mismatches": mismatches}


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
