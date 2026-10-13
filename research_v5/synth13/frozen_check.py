"""Re-hash every path recorded in frozen-pipeline-v12.json."""
from __future__ import annotations

import json

import runtime  # noqa: F401

from runtime import FROZEN_MANIFEST, FROZEN_STAGE1, ORIGINAL_STAGE2, ROOT, require, sha256_file

MODEL_FILES = {
    "stage1_alldata": FROZEN_STAGE1,
    "stage2_alldata": ORIGINAL_STAGE2,
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
        "records": sources["records"] + models["records"],
    }


def _check_sources(recorded: dict) -> dict:
    mismatches = []
    records = []
    for relative, expected in recorded.items():
        actual = sha256_file(ROOT / relative)
        records.append({"path": relative, "sha256": actual})
        if actual != expected:
            mismatches.append(f"{relative} {actual}")
    return {"checked": len(recorded), "mismatches": mismatches, "records": records}


def _check_models(recorded: dict) -> dict:
    mismatches = []
    records = []
    require(set(recorded) == set(MODEL_FILES), "frozen model keys")
    for name, path in MODEL_FILES.items():
        actual = sha256_file(path)
        records.append({"path": name, "sha256": actual})
        if actual != recorded[name]:
            mismatches.append(f"{name} {actual}")
    return {"checked": len(recorded), "mismatches": mismatches, "records": records}
