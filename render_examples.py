"""Render native predictions beside their query and reviewed annotation domains."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from evaluate import EvaluationError, data_path, load_binary_png, load_manifest, reviewed_domains


POSITIVE_COLOR = "#009E73"
NEGATIVE_COLOR = "#D55E00"
UNSCORED_COLOR = "#BDBDBD"
PREDICTION_COLOR = "#0072B2"
QUERY_COLOR = "#E69F00"
MAX_DISPLAY_DIMENSION = 1600


@dataclass(frozen=True)
class RenderedExample:
    """Paths produced for one selected query example."""

    query_id: str
    png_path: Path
    pdf_path: Path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--ids", required=True, nargs="+", help="Selected query IDs")
    return parser.parse_args(None if argv is None else list(argv))


def select_rows(manifest: Path, query_ids: Sequence[str]) -> list[dict[str, Any]]:
    """Load labeled rows and return the requested IDs in caller-provided order."""
    rows = load_manifest(manifest, require_labels=True)
    by_id = {str(row["id"]): row for row in rows}
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for query_id in query_ids:
        if query_id in seen:
            raise EvaluationError(f"Duplicate selected query ID: {query_id}")
        seen.add(query_id)
        if query_id not in by_id:
            raise EvaluationError(f"Selected query ID is absent from manifest: {query_id}")
        selected.append(by_id[query_id])
    if not selected:
        raise EvaluationError("At least one query ID must be selected")
    return selected


def annotation_layers(
    row: dict[str, Any], data_root: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Return positive, reviewed-negative, and unscored layers from evaluator domains."""
    positive, known, _blank, metadata = reviewed_domains(row, data_root)
    negative = known & ~positive
    unscored = ~known
    if bool(np.any(positive & negative)) or bool(np.any((positive | negative) & unscored)):
        raise EvaluationError(f"Annotation layers overlap for {row['id']}")
    return positive, negative, unscored, metadata


def load_drawing(row: dict[str, Any], data_root: Path) -> np.ndarray:
    """Load the declared drawing at its native dimensions as uint8 RGB."""
    path = data_path(data_root, row["image"])
    try:
        with Image.open(path) as image:
            if image.size != (row["width"], row["height"]):
                raise EvaluationError(
                    f"Drawing dimensions do not match row {row['id']}: {path}"
                )
            return np.asarray(image.convert("RGB"), dtype=np.uint8)
    except (OSError, ValueError) as exc:
        raise EvaluationError(f"Cannot read drawing for {row['id']}: {path}") from exc


def load_prediction(row: dict[str, Any], predictions: Path) -> np.ndarray:
    """Load and validate the native binary prediction through the evaluator helper."""
    return load_binary_png(
        predictions / f"{row['id']}.png", row["width"], row["height"]
    )


def render_examples(
    manifest: Path,
    data_root: Path,
    predictions: Path,
    output_dir: Path,
    query_ids: Sequence[str],
) -> list[RenderedExample]:
    """Render each selected manifest row as a native drawing review figure."""
    rows = select_rows(manifest, query_ids)
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[RenderedExample] = []
    for row in rows:
        rendered.append(_render_one(row, data_root, predictions, output_dir))
    return rendered


def _render_one(
    row: dict[str, Any], data_root: Path, predictions: Path, output_dir: Path
) -> RenderedExample:
    plt, Patch, Rectangle = _matplotlib()
    drawing = load_drawing(row, data_root)
    prediction = load_prediction(row, predictions)
    positive, negative, unscored, metadata = annotation_layers(row, data_root)
    x0, y0, x1, y1 = (int(value) for value in row["query_box"])
    crop = drawing[y0:y1, x0:x1]
    if crop.shape != (y1 - y0, x1 - x0, 3):
        raise EvaluationError(f"Query crop is invalid for {row['id']}")
    display_stride = _display_stride(row["width"], row["height"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=False)
    fig.suptitle(
        f"{row['id']} | Native size {row['width']} × {row['height']} pixels",
        fontsize=15,
        fontweight="bold",
    )
    _draw_overview(axes[0, 0], drawing, row, Rectangle, display_stride)
    _draw_crop(axes[0, 1], crop, row)
    _draw_prediction(axes[1, 0], drawing, prediction, row, Rectangle, display_stride)
    _draw_annotations(
        axes[1, 1],
        drawing,
        positive,
        negative,
        unscored,
        row,
        Patch,
        Rectangle,
        display_stride,
    )
    _style_axes(axes)

    exclusion_note = (
        "Query/context boxes are excluded; unknown and unscored pixels are never negatives.\n"
        f"Full-size masks are validated; display panels sample every {display_stride}th native pixel."
    )
    fig.text(0.02, 0.012, exclusion_note, fontsize=8, color="#444444", va="bottom")
    fig.text(
        0.98,
        0.015,
        f"Reviewed pixels: {int(metadata['original_known_pixels']) - int(metadata['excluded_known_pixels']):,}",
        ha="right",
        fontsize=9,
        color="#444444",
    )
    fig.subplots_adjust(top=0.91, bottom=0.08, wspace=0.12, hspace=0.24)

    png_path = output_dir / f"{row['id']}.png"
    pdf_path = output_dir / f"{row['id']}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return RenderedExample(str(row["id"]), png_path, pdf_path)


def _draw_overview(
    axes: Any,
    drawing: np.ndarray,
    row: dict[str, Any],
    rectangle: Any,
    display_stride: int,
) -> None:
    _show_display_image(axes, drawing, row, display_stride)
    _draw_boxes(axes, row, rectangle)
    axes.set_title(f"Original RGB drawing (display stride {display_stride})")
    axes.set_xlabel("x (native pixels)")
    axes.set_ylabel("y (native pixels)")


def _draw_crop(axes: Any, crop: np.ndarray, row: dict[str, Any]) -> None:
    x0, y0, x1, y1 = (int(value) for value in row["query_box"])
    axes.imshow(
        crop,
        extent=(x0 - 0.5, x1 - 0.5, y1 - 0.5, y0 - 0.5),
        origin="upper",
        interpolation="nearest",
    )
    axes.set_title(f"Exact RGB query crop ({x1 - x0} × {y1 - y0} native pixels)")
    axes.set_xlabel("x (native pixels)")
    axes.set_ylabel("y (native pixels)")


def _draw_prediction(
    axes: Any,
    drawing: np.ndarray,
    prediction: np.ndarray,
    row: dict[str, Any],
    rectangle: Any,
    display_stride: int,
) -> None:
    _show_display_image(axes, drawing, row, display_stride)
    overlay = _solid_overlay(prediction, PREDICTION_COLOR, alpha=0.72, display_stride=display_stride)
    _show_display_overlay(axes, overlay, row)
    _draw_boxes(axes, row, rectangle)
    axes.set_title(f"Actual native prediction (display stride {display_stride})")
    axes.set_xlabel("x (native pixels)")
    axes.set_ylabel("y (native pixels)")


def _draw_annotations(
    axes: Any,
    drawing: np.ndarray,
    positive: np.ndarray,
    negative: np.ndarray,
    unscored: np.ndarray,
    row: dict[str, Any],
    patch: Any,
    rectangle: Any,
    display_stride: int,
) -> None:
    _show_display_image(axes, drawing, row, display_stride)
    overlay = _annotation_overlay(
        positive, negative, unscored, display_stride
    )
    _show_display_overlay(axes, overlay, row)
    axes.set_title(f"Reviewed annotation domains (display stride {display_stride})")
    axes.set_xlabel("x (native pixels)")
    axes.set_ylabel("y (native pixels)")
    axes.legend(
        handles=[
            patch(
                facecolor=PREDICTION_COLOR,
                edgecolor="none",
                label="Predicted selected pixels",
            ),
            patch(facecolor=POSITIVE_COLOR, edgecolor="none", label="Reviewed positive"),
            patch(
                facecolor=NEGATIVE_COLOR,
                edgecolor="none",
                label="Reviewed negative (known minus positive)",
            ),
            patch(
                facecolor=UNSCORED_COLOR,
                edgecolor="none",
                label="Unscored: unknown or excluded; never negative",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        frameon=False,
        fontsize=8,
    )
    _draw_boxes(axes, row, rectangle)


def _draw_boxes(axes: Any, row: dict[str, Any], rectangle: Any) -> None:
    x0, y0, x1, y1 = (int(value) for value in row["query_box"])
    axes.add_patch(
        rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            fill=False,
            edgecolor=QUERY_COLOR,
            linewidth=1.6,
            label="Query rectangle",
        )
    )
    for box in row.get("context_boxes", []):
        cx0, cy0, cx1, cy1 = (int(value) for value in box)
        axes.add_patch(
            rectangle(
                (cx0, cy0),
                cx1 - cx0,
                cy1 - cy0,
                fill=False,
                edgecolor=QUERY_COLOR,
                linestyle="--",
                linewidth=1.2,
            )
        )


def _solid_overlay(
    mask: np.ndarray, color: str, alpha: float, display_stride: int = 1
) -> np.ndarray:
    sampled_mask = mask[::display_stride, ::display_stride]
    overlay = np.zeros((*sampled_mask.shape, 4), dtype=np.float32)
    _apply_overlay(overlay, sampled_mask, color, alpha)
    return overlay


def _annotation_overlay(
    positive: np.ndarray,
    negative: np.ndarray,
    unscored: np.ndarray,
    display_stride: int,
) -> np.ndarray:
    sampled_unscored = unscored[::display_stride, ::display_stride]
    overlay = np.zeros((*sampled_unscored.shape, 4), dtype=np.float32)
    _apply_overlay(overlay, sampled_unscored, UNSCORED_COLOR, 0.42)
    _apply_overlay(
        overlay,
        negative[::display_stride, ::display_stride],
        NEGATIVE_COLOR,
        0.62,
    )
    _apply_overlay(
        overlay,
        positive[::display_stride, ::display_stride],
        POSITIVE_COLOR,
        0.78,
    )
    return overlay


def _apply_overlay(overlay: np.ndarray, mask: np.ndarray, color: str, alpha: float) -> None:
    rgb = _hex_to_rgb(color)
    overlay[mask, :3] = rgb
    overlay[mask, 3] = alpha


def _hex_to_rgb(color: str) -> tuple[float, float, float]:
    value = color.lstrip("#")
    if len(value) != 6:
        raise ValueError(f"Expected six-digit hex color: {color}")
    return (
        int(value[0:2], 16) / 255.0,
        int(value[2:4], 16) / 255.0,
        int(value[4:6], 16) / 255.0,
    )


def _display_stride(width: int, height: int) -> int:
    """Choose a common integer sampling stride for native display panels."""
    largest_dimension = max(int(width), int(height))
    return max(1, math.ceil(largest_dimension / MAX_DISPLAY_DIMENSION))


def _show_display_image(
    axes: Any, drawing: np.ndarray, row: dict[str, Any], display_stride: int
) -> None:
    axes.imshow(
        drawing[::display_stride, ::display_stride],
        extent=(0, row["width"], row["height"], 0),
        origin="upper",
        interpolation="nearest",
    )


def _show_display_overlay(axes: Any, overlay: np.ndarray, row: dict[str, Any]) -> None:
    axes.imshow(
        overlay,
        extent=(0, row["width"], row["height"], 0),
        origin="upper",
        interpolation="nearest",
    )


def _style_axes(axes: np.ndarray) -> None:
    for axis in axes.flat:
        axis.set_aspect("equal", adjustable="box")
        axis.tick_params(labelsize=8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)


def _matplotlib() -> tuple[Any, Any, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 9,
            "axes.titlesize": 12,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 120,
            "savefig.dpi": 300,
        }
    )
    return plt, Patch, Rectangle


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        rendered = render_examples(
            args.manifest,
            args.data_root,
            args.predictions,
            args.output_dir,
            args.ids,
        )
    except (EvaluationError, OSError, ValueError) as exc:
        raise SystemExit(f"Rendering failed: {exc}") from exc
    for example in rendered:
        print(f"{example.query_id}: {example.png_path} and {example.pdf_path}")


if __name__ == "__main__":
    main()
