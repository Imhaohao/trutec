"""Render the real-drawing training queries with the most out-of-fold false-positive mass (v14 CV scores).

For each, writes a PNG: downsampled drawing with true positives green, false positives red,
missed positives blue and the query box yellow, plus native crops of the query, the
highest-scoring false positive and the highest-scoring true positive. Fit and
development documents only.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "alldata12"))
sys.path.insert(1, str(HERE))

import runtime  # noqa: E402,F401

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from banks import load_rgb  # noqa: E402
from build import cell_file, positions  # noqa: E402
from manifests import development_rows, fit_ids, fit_rows  # noqa: E402
from runtime import CELL_DEV, CELL_FIT  # noqa: E402

SCORES = HERE.parent / "synthw14" / "oof_scores_v14.npy"
OUT = HERE / "figures"
TOP = 4
CROP = 160


def table_offsets() -> dict[str, tuple[int, int]]:
    offsets, start = {}, 0
    for directory in (CELL_FIT, CELL_DEV):
        for path in sorted(directory.glob("*.npz")):
            with np.load(path) as stored:
                count = len(stored["y"])
            if count:
                offsets[path.stem] = (start, start + count)
                start += count
    return offsets


def worst_queries(rows: dict[str, dict], offsets, scores) -> list[str]:
    owned = fit_ids()
    mass = []
    for query_id, (start, stop) in offsets.items():
        if not rows[query_id]["document_id"].startswith("doc-"):
            continue
        with np.load(cell_file(rows[query_id], owned)) as stored:
            y = stored["y"]
            share = float(stored["n_neg_domain"]) / max(int((y == 0).sum()), 1)
        mass.append((float(((scores[start:stop] >= 0.25) & (y == 0)).sum()) * share, query_id))
    return [query_id for _mass, query_id in sorted(mass, reverse=True)[:TOP]]


def crop(rgb, y: int, x: int) -> np.ndarray:
    half = CROP // 2
    padded = np.pad(rgb, ((half, half), (half, half), (0, 0)), constant_values=255)
    return padded[y : y + CROP, x : x + CROP]


def render(row: dict, scores: np.ndarray, sample_index: int) -> np.ndarray:
    rgb = load_rgb(row["image"])
    ys, xs, labels = positions(row, sample_index)
    predicted = scores >= 0.25
    scale = 900 / max(rgb.shape[:2])
    small = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA).copy()
    colours = {(1, True): (0, 170, 0), (0, True): (220, 0, 0), (1, False): (0, 80, 255)}
    for y, x, label, hit in zip(ys, xs, labels, predicted):
        colour = colours.get((int(label), bool(hit)))
        if colour:
            cv2.circle(small, (int(x * scale), int(y * scale)), 2, colour, -1)
    x0, y0, x1, y1 = (int(value) for value in row["query_box"])
    cv2.rectangle(small, (int(x0 * scale), int(y0 * scale)), (int(x1 * scale), int(y1 * scale)), (255, 200, 0), 2)
    panels = [crop(rgb, (y0 + y1) // 2, (x0 + x1) // 2)]
    for label in (0, 1):
        chosen = np.flatnonzero(predicted & (labels == label))
        best = chosen[np.argmax(scores[chosen])] if len(chosen) else None
        panels.append(crop(rgb, int(ys[best]), int(xs[best])) if best is not None else np.full((CROP, CROP, 3), 255, np.uint8))
    strip = np.concatenate(panels, axis=1)
    width = max(small.shape[1], strip.shape[1])
    pad = lambda image: np.pad(image, ((0, 0), (0, width - image.shape[1]), (0, 0)), constant_values=255)
    return np.concatenate([pad(small), pad(strip)], axis=0)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    rows = {row["id"]: row for row in fit_rows() + development_rows()}
    offsets = table_offsets()
    scores = np.load(SCORES)
    owned = fit_ids()
    for rank, query_id in enumerate(worst_queries(rows, offsets, scores)):
        start, stop = offsets[query_id]
        with np.load(cell_file(rows[query_id], owned)) as stored:
            sample_index = int(stored["sample_index"])
        figure = render(rows[query_id], scores[start:stop], sample_index)
        path = OUT / f"fp{rank}_{query_id}.png"
        cv2.imwrite(str(path), cv2.cvtColor(figure, cv2.COLOR_RGB2BGR))
        print(path, rows[query_id]["document_id"])


if __name__ == "__main__":
    main()
