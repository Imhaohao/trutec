"""Attach spectral features to every v12 stage-2 cell (fit and development).

Cell positions are recomputed with the same sampler and stored sample_index that built
research_v5/alldata12/cache/cells; the recomputed labels must equal the stored labels.
Output: research_v5/spectral15/cache/<query id>.npz with S (cells x features) and y.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "alldata12"))
sys.path.insert(1, str(HERE))

import runtime  # noqa: E402,F401

from collections import defaultdict  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import spectral  # noqa: E402
from banks import load_rgb  # noqa: E402
from constants import CELLS_PER_CLASS, DATA_ROOT, GRID_STRIDE  # noqa: E402
from dataset import _sample_cells  # noqa: E402
from evaluate import reviewed_domains  # noqa: E402
from manifests import development_rows, fit_ids, fit_rows  # noqa: E402
from runtime import CELL_DEV, CELL_FIT, log  # noqa: E402
from texture_learning import grid_centers  # noqa: E402

CACHE = HERE / "cache"


def cell_file(row: dict, owned: set[str]) -> Path:
    return (CELL_FIT if row["id"] in owned else CELL_DEV) / f"{row['id']}.npz"


def positions(row: dict, sample_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positive, known, _blank, _meta = reviewed_domains(row, DATA_ROOT)
    centers_y, centers_x = grid_centers(int(row["height"]), int(row["width"]), GRID_STRIDE)
    pos_idx, neg_idx, _n_pos, _n_neg = _sample_cells(positive, known, centers_y, centers_x, CELLS_PER_CLASS, sample_index)
    grid = np.concatenate([pos_idx, neg_idx]) if len(pos_idx) + len(neg_idx) else np.zeros((0, 2), int)
    labels = np.concatenate([np.ones(len(pos_idx), np.int8), np.zeros(len(neg_idx), np.int8)])
    return centers_y[grid[:, 0]], centers_x[grid[:, 1]], labels


def build_query(row: dict, gray: np.ndarray, owned: set[str]) -> int:
    target = CACHE / f"{row['id']}.npz"
    if target.is_file():
        return 0
    with np.load(cell_file(row, owned)) as stored:
        stored_y = np.array(stored["y"], dtype=np.int8)
        sample_index = int(stored["sample_index"])
    ys, xs, labels = positions(row, sample_index)
    assert np.array_equal(labels, stored_y), f"label drift {row['id']}"
    features = spectral.cell_features(gray, row["query_box"], ys, xs)
    np.savez(target, S=features, y=labels)
    return len(labels)


def main() -> None:
    CACHE.mkdir(exist_ok=True)
    owned = fit_ids()
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in fit_rows() + development_rows():
        grouped[row["image"]].append(row)
    total = 0
    for image, rows in grouped.items():
        gray = cv2.cvtColor(load_rgb(image), cv2.COLOR_RGB2GRAY)
        total += sum(build_query(row, gray, owned) for row in rows)
        log(f"spectral {image} queries={len(rows)} cells_so_far={total}")
    log(f"spectral done cells={total}")


if __name__ == "__main__":
    main()
