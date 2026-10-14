"""Repeat the stage-2 capacity CV on v12's own cells (v12 stage-1 folds, no synthetic rows).

Same rule as cv_capacity.py: a configuration replaces the recipe only if its real-document
macro IoU at threshold 0.25 is at least 0.01 above the recipe's; the largest passing macro IoU
wins. Public validation is never read.
"""
from __future__ import annotations

import cv

import json

import cv_capacity
from runtime import ALLDATA12, log

cv.CELL_FIT = ALLDATA12 / "cache" / "cells" / "fit"
cv.CELL_DEV = ALLDATA12 / "cache" / "cells" / "development"
cv.CELL_SYN = ALLDATA12 / "cache" / "cells" / "no-synthetic"
OUT = cv.HERE / "cv_capacity12.json"


def main() -> None:
    table = cv.load()
    weights = cv.training_weights(table, 0.0)
    results = [cv_capacity.run_config(table, weights, item) for item in cv_capacity.configurations()]
    baseline = results[0]["macro_iou"]
    winners = [item for item in results[1:] if item["macro_iou"] >= baseline + cv_capacity.MARGIN]
    chosen = max(winners, key=lambda item: item["macro_iou"]) if winners else results[0]
    OUT.write_text(json.dumps({"results": results, "chosen": chosen["config"]["name"], "rule": __doc__}, indent=2), encoding="utf-8")
    log(f"chosen {chosen['config']['name']}")


if __name__ == "__main__":
    main()
