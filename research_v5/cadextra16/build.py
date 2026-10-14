"""Build v12-recipe stage-2 cells for the 36 development generated-CAD queries v12 never used.

These 9 cad- documents are in artifacts/splits/development.json but not development-real.json,
so v12's stage 1 never trained on them; each document is assigned round-robin (sorted id) to a
fold and scored with that fold's v12 stage-1 model. Cells, banks and folds are written under
research_v5/cadextra16 only; nothing in research_v5/alldata12 is modified.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "alldata12"))

import runtime  # noqa: E402,F401

import json  # noqa: E402

import build_cells  # noqa: E402
from runtime import ROOT, log, require  # noqa: E402

CELLS = HERE / "cells"
BANKS = HERE / "banks"


def cad_rows() -> list[dict]:
    examples = json.loads((ROOT / "artifacts" / "splits" / "development.json").read_text(encoding="utf-8"))["examples"]
    real = {row["id"] for row in json.loads((ROOT / "artifacts" / "splits" / "development-real.json").read_text(encoding="utf-8"))["examples"]}
    rows = [row for row in examples if row["document_id"].startswith("cad-") and row["id"] not in real]
    require(len(rows) == 36, f"cad rows {len(rows)}")
    return rows


def assign_folds(rows: list[dict]) -> dict[str, int]:
    documents = sorted({row["document_id"] for row in rows})
    return {document: index % 4 for index, document in enumerate(documents)}


def inside_own_tree(path: Path) -> None:
    require(HERE in Path(path).resolve().parents, f"write outside cadextra16: {path}")


def main() -> None:
    CELLS.mkdir(exist_ok=True)
    BANKS.mkdir(exist_ok=True)
    build_cells.CELL_DEV = CELLS
    build_cells.BANK_DIR = BANKS
    build_cells.inside = inside_own_tree
    rows = cad_rows()
    folds = assign_folds(rows)
    (HERE / "folds.json").write_text(json.dumps(folds, indent=2), encoding="utf-8")
    summary = build_cells.build_cells(rows, folds)
    summary.pop("images", None)
    (HERE / "cells_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"cadextra cells {summary}")


if __name__ == "__main__":
    main()
