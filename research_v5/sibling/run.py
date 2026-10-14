"""Run the sibling-competition experiment."""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research_v5"))

from grids import build_grids  # noqa: E402
from rules import _self_check  # noqa: E402
from score import evaluate  # noqa: E402


def main() -> None:
    _self_check()
    payload = build_grids()
    evaluate(payload)


if __name__ == "__main__":
    main()
