"""Render report figures for public validation examples with Pillow only.

Each figure has three panels cropped around the reviewed area: the original drawing with the
query box, the prediction, and the scored overlay. Overlay colours: green = predicted and
positive (TP), red = predicted but reviewed negative (FP), blue = positive but not predicted
(FN), grey = reviewed negative correctly left out (TN); unscored pixels keep the drawing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluate import reviewed_domains  # noqa: E402

PANEL = 900
MARGIN = 160
COLOURS = {"tp": (40, 170, 70), "fp": (220, 40, 40), "fn": (40, 90, 230), "tn": (150, 150, 150)}
QUERY = (255, 170, 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--ids", required=True, nargs="+")
    return parser.parse_args()


def support_mask(row: dict, shape) -> np.ndarray:
    """Query and context rectangles; the official evaluator leaves these unscored."""
    mask = np.zeros(shape, bool)
    for x0, y0, x1, y1 in [row["query_box"], *row.get("context_boxes", [])]:
        mask[int(y0) : int(y1), int(x0) : int(x1)] = True
    return mask


def crop_box(known: np.ndarray, query_box, shape) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(known)
    x0, y0, x1, y1 = (int(v) for v in query_box)
    left, top = min(xs.min(), x0) - MARGIN, min(ys.min(), y0) - MARGIN
    right, bottom = max(xs.max(), x1) + MARGIN, max(ys.max(), y1) + MARGIN
    return max(left, 0), max(top, 0), min(right, shape[1]), min(bottom, shape[0])


def overlay(rgb: np.ndarray, predicted: np.ndarray, positive: np.ndarray, known: np.ndarray) -> np.ndarray:
    out = rgb.astype(np.float32)
    regions = {
        "tp": predicted & positive,
        "fp": predicted & known & ~positive,
        "fn": ~predicted & positive,
        "tn": ~predicted & known & ~positive,
    }
    for name, mask in regions.items():
        out[mask] = 0.45 * out[mask] + 0.55 * np.array(COLOURS[name], np.float32)
    return out.astype(np.uint8)


def panel(image: np.ndarray, box) -> Image.Image:
    left, top, right, bottom = box
    tile = Image.fromarray(image[top:bottom, left:right])
    tile.thumbnail((PANEL, PANEL))
    return tile


def render(row: dict, data_root: Path, predictions: Path, output_dir: Path) -> Path:
    rgb = np.asarray(Image.open(data_root / row["image"]).convert("RGB"))
    predicted = np.asarray(Image.open(predictions / f"{row['id']}.png")) > 0
    positive, known, _blank, _meta = reviewed_domains(row, data_root)
    support = support_mask(row, known.shape)
    positive, known = positive & ~support, known & ~support
    box = crop_box(known | positive, row["query_box"], rgb.shape)
    marked = Image.fromarray(rgb.copy())
    ImageDraw.Draw(marked).rectangle(tuple(int(v) for v in row["query_box"]), outline=QUERY, width=8)
    prediction = np.where(predicted[..., None], np.array([30, 30, 30], np.uint8), np.array([255, 255, 255], np.uint8))
    tiles = [panel(np.asarray(marked), box), panel(prediction, box), panel(overlay(rgb, predicted, positive, known), box)]
    sheet = Image.new("RGB", (sum(t.width for t in tiles) + 40, max(t.height for t in tiles)), "white")
    x = 0
    for tile in tiles:
        sheet.paste(tile, (x, 0))
        x += tile.width + 20
    path = output_dir / f"{row['id']}.png"
    sheet.save(path)
    return path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = {row["id"]: row for row in json.loads(args.manifest.read_text(encoding="utf-8"))["examples"]}
    for query_id in args.ids:
        print(render(rows[query_id], args.data_root, args.predictions, args.output_dir))


if __name__ == "__main__":
    main()
