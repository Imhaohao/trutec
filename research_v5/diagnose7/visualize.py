"""Generate visualization figures for top false positive queries.

Saves PNG figures with color-coded reviewed overlays and side-by-side
original vs cleaned query crops.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bootstrap  # noqa: E402,F401

from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from analysis import QueryStat  # noqa: E402
from bootstrap import DATA_ROOT, FIGURES_DIR, log  # noqa: E402
from guards import clean_crop_g2  # noqa: E402

COLOR_TP = np.array([0, 220, 0], dtype=np.float32)       # Green
COLOR_FP = np.array([235, 30, 30], dtype=np.float32)     # Red
COLOR_FN = np.array([255, 215, 0], dtype=np.float32)     # Yellow
COLOR_TN = np.array([45, 105, 225], dtype=np.float32)    # Blue


def render_top_figures(query_stats: list[QueryStat], output_dir: Path = FIGURES_DIR, top_k: int = 6) -> list[Path]:
    """Generate and save PNG figures for top FP queries."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for rank, stat in enumerate(query_stats[:top_k]):
        out_path = output_dir / f"{stat.id}.png"
        render_single_figure(stat, rank + 1, out_path)
        paths.append(out_path)
        log(f"saved figure {rank+1}/{top_k}: {out_path.name}")
    return paths


def render_single_figure(stat: QueryStat, rank: int, output_path: Path, data_root: Path = DATA_ROOT) -> None:
    """Render drawing overlay, query crops, legend, and metric banner."""
    rgb = _load_rgb_image(data_root / stat.image)
    overlay, y0, x0 = _build_overlay_crop(stat, rgb)
    orig_crop, clean_crop = _extract_query_crops(stat, rgb)

    panel = _compose_side_panel(stat, orig_crop, clean_crop)
    canvas = _assemble_canvas(stat, rank, overlay, panel)
    canvas.save(output_path, format="PNG")


def _load_rgb_image(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"))


def _build_overlay_crop(stat: QueryStat, rgb: np.ndarray) -> tuple[np.ndarray, int, int]:
    margin = 40
    y0 = max(0, int(stat.y.min()) - margin)
    y1 = min(rgb.shape[0], int(stat.y.max()) + margin)
    x0 = max(0, int(stat.x.min()) - margin)
    x1 = min(rgb.shape[1], int(stat.x.max()) + margin)

    crop = rgb[y0:y1, x0:x1].copy().astype(np.float32)
    cy = stat.y - y0
    cx = stat.x - x0

    tp = stat.decision & stat.positive
    fp = stat.decision & ~stat.positive
    fn = ~stat.decision & stat.positive
    tn = ~stat.decision & ~stat.positive

    # Blend overlays
    crop[cy[tn], cx[tn]] = crop[cy[tn], cx[tn]] * 0.55 + COLOR_TN * 0.45
    crop[cy[tp], cx[tp]] = crop[cy[tp], cx[tp]] * 0.25 + COLOR_TP * 0.75
    crop[cy[fp], cx[fp]] = crop[cy[fp], cx[fp]] * 0.25 + COLOR_FP * 0.75
    crop[cy[fn], cx[fn]] = crop[cy[fn], cx[fn]] * 0.25 + COLOR_FN * 0.75

    return np.clip(crop, 0, 255).astype(np.uint8), y0, x0


def _extract_query_crops(stat: QueryStat, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x0, y0, x1, y1 = stat.query_box
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    orig_gray = np.ascontiguousarray(gray[y0:y1, x0:x1])
    clean_gray = clean_crop_g2(orig_gray)

    orig_rgb = cv2.cvtColor(orig_gray, cv2.COLOR_GRAY2RGB)
    clean_rgb = cv2.cvtColor(clean_gray, cv2.COLOR_GRAY2RGB)
    return orig_rgb, clean_rgb


def _compose_side_panel(stat: QueryStat, orig_rgb: np.ndarray, clean_rgb: np.ndarray) -> Image.Image:
    panel_w = 400
    panel_h = 700
    img = Image.new("RGB", (panel_w, panel_h), (245, 245, 248))
    draw = ImageDraw.Draw(img)

    # Title
    draw.text((20, 20), "Query Crops", fill=(20, 20, 20))

    # Original Crop
    orig_pil = _resize_crop(orig_rgb, 180, 180)
    img.paste(orig_pil, (20, 60))
    draw.rectangle([19, 59, 20 + orig_pil.width, 59 + orig_pil.height], outline=(150, 150, 150), width=1)
    draw.text((20, 250), f"Original: {stat.query_box[2]-stat.query_box[0]}x{stat.query_box[3]-stat.query_box[1]}px", fill=(50, 50, 50))

    # Cleaned Crop
    clean_pil = _resize_crop(clean_rgb, 180, 180)
    img.paste(clean_pil, (210, 60))
    draw.rectangle([209, 59, 210 + clean_pil.width, 59 + clean_pil.height], outline=(150, 150, 150), width=1)
    draw.text((210, 250), "Cleaned (G2)", fill=(50, 50, 50))

    # Legend section
    _draw_legend(draw, 20, 310)

    # Metrics section
    _draw_metrics(draw, stat, 20, 480)
    return img


def _resize_crop(crop_rgb: np.ndarray, max_w: int, max_h: int) -> Image.Image:
    pil_crop = Image.fromarray(crop_rgb)
    scale = min(max_w / max(pil_crop.width, 1), max_h / max(pil_crop.height, 1), 3.0)
    nw = max(1, int(pil_crop.width * scale))
    nh = max(1, int(pil_crop.height * scale))
    return pil_crop.resize((nw, nh), resample=Image.Resampling.NEAREST)


def _draw_legend(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    draw.text((x, y), "Color Legend", fill=(20, 20, 20))
    items = [
        ("True Positive (TP)", (0, 190, 0)),
        ("False Positive (FP)", (220, 30, 30)),
        ("False Negative (FN)", (210, 170, 0)),
        ("True Negative (TN)", (45, 105, 225)),
    ]
    for idx, (label, color) in enumerate(items):
        iy = y + 30 + idx * 32
        draw.rectangle([x, iy, x + 24, iy + 24], fill=color, outline=(0, 0, 0))
        draw.text((x + 35, iy + 4), label, fill=(30, 30, 30))


def _draw_metrics(draw: ImageDraw.ImageDraw, stat: QueryStat, x: int, y: int) -> None:
    draw.text((x, y), "Evaluation Metrics", fill=(20, 20, 20))
    lines = [
        f"IoU:       {stat.iou:0.4f}",
        f"Precision: {stat.precision:0.4f}",
        f"Recall:    {stat.recall:0.4f}",
        f"TP Count:  {stat.tp:,}",
        f"FP Count:  {stat.fp:,}",
        f"FN Count:  {stat.fn:,}",
        f"TN Count:  {stat.tn:,}",
    ]
    for idx, line in enumerate(lines):
        draw.text((x, y + 25 + idx * 24), line, fill=(40, 40, 40))


def _assemble_canvas(stat: QueryStat, rank: int, overlay_rgb: np.ndarray, panel_img: Image.Image) -> Image.Image:
    header_h = 70
    overlay_pil = Image.fromarray(overlay_rgb)

    # Scale overlay if too large
    max_overlay_w = 2400
    if overlay_pil.width > max_overlay_w:
        scale = max_overlay_w / overlay_pil.width
        overlay_pil = overlay_pil.resize(
            (max_overlay_w, int(overlay_pil.height * scale)), resample=Image.Resampling.BILINEAR
        )

    content_h = max(overlay_pil.height, panel_img.height)
    canvas_w = overlay_pil.width + panel_img.width + 40
    canvas_h = content_h + header_h + 30

    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Top banner
    draw.rectangle([0, 0, canvas_w, header_h], fill=(30, 35, 45))
    title_text = f"Rank {rank} FP Query: {stat.id}   |   Document: {stat.document_id}"
    draw.text((20, 24), title_text, fill=(255, 255, 255))

    # Paste main overlay and side panel
    canvas.paste(overlay_pil, (20, header_h + 20))
    canvas.paste(panel_img, (overlay_pil.width + 30, header_h + 20))

    return canvas
