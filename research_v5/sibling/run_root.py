"""Root-supervisor driver: run the sibling experiment with a recorded identity tolerance.

The cached HGB development scores were produced on another host with older code; the
rebuilt grids differ by up to about 0.075 (mean about 0.0017). Every comparison below
uses the rebuilt grids as R0, and the gap per query is recorded in findings.txt.
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "4"

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for path in (HERE, ROOT, ROOT / "research_v5"):
    sys.path.insert(0, str(path))

import grids  # noqa: E402
from rules import _self_check  # noqa: E402
from score import evaluate  # noqa: E402

RECORDED_IDENTITY_TOLERANCE = 0.1


def main() -> None:
    _self_check()
    grids.IDENTITY_LIMIT = RECORDED_IDENTITY_TOLERANCE
    payload = grids.build_grids()
    evaluate(payload)


if __name__ == "__main__":
    main()
