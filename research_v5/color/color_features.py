"""Color feature extraction and soft gating for CAD hatch matching.

Computes label-free neighborhood chroma and hue statistics from full RGB
drawings and compares them with robust query crop signatures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class QueryColorSignature:
    """Color signature extracted from a query crop."""

    is_chromatic: bool
    chroma_fraction: float
    hue_histogram: np.ndarray


@dataclass(frozen=True)
class WindowColorMaps:
    """Local neighborhood color statistics for a specific window size."""

    ink_count: np.ndarray
    chroma_count: np.ndarray
    hue_histogram_stack: np.ndarray


def extract_ink_and_chroma(
    rgb: np.ndarray,
    value_threshold: float = 235.0,
    chroma_threshold: float = 25.0,
    saturation_threshold: float = 25.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Identify ink pixels and chromatic ink pixels in an RGB image.

    Returns:
        is_ink: float32 mask (1.0 for ink, 0.0 for paper)
        is_chroma_ink: float32 mask (1.0 for chromatic ink, 0.0 otherwise)
        hue: uint8 hue channel in [0, 179]
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    red = rgb[:, :, 0].astype(np.float32)
    green = rgb[:, :, 1].astype(np.float32)
    blue = rgb[:, :, 2].astype(np.float32)

    max_channel = np.maximum(np.maximum(red, green), blue)
    min_channel = np.minimum(np.minimum(red, green), blue)
    chroma = max_channel - min_channel

    is_ink_bool = (value < value_threshold) | (chroma > 20.0)
    is_chroma_bool = is_ink_bool & (chroma > chroma_threshold) & (saturation > saturation_threshold)

    return (
        is_ink_bool.astype(np.float32),
        is_chroma_bool.astype(np.float32),
        hue,
    )


def compute_window_maps(
    is_ink: np.ndarray,
    is_chroma_ink: np.ndarray,
    hue: np.ndarray,
    window_size: int,
    num_hue_bins: int = 6,
) -> WindowColorMaps:
    """Compute local neighborhood color statistics via box filters."""
    ksize = (window_size, window_size)
    ink_cnt = cv2.boxFilter(
        is_ink,
        ddepth=-1,
        ksize=ksize,
        normalize=False,
        borderType=cv2.BORDER_REFLECT_101,
    )
    chr_cnt = cv2.boxFilter(
        is_chroma_ink,
        ddepth=-1,
        ksize=ksize,
        normalize=False,
        borderType=cv2.BORDER_REFLECT_101,
    )

    bin_width = 180.0 / float(num_hue_bins)
    hue_bins = np.clip((hue / bin_width).astype(int), 0, num_hue_bins - 1)

    hist_layers = []
    for bin_idx in range(num_hue_bins):
        bin_mask = (is_chroma_ink > 0.0) & (hue_bins == bin_idx)
        bin_layer = cv2.boxFilter(
            bin_mask.astype(np.float32),
            ddepth=-1,
            ksize=ksize,
            normalize=False,
            borderType=cv2.BORDER_REFLECT_101,
        )
        hist_layers.append(bin_layer)

    hist_stack = np.stack(hist_layers, axis=-1)
    return WindowColorMaps(
        ink_count=ink_cnt,
        chroma_count=chr_cnt,
        hue_histogram_stack=hist_stack,
    )


def build_image_color_cache(
    rgb: np.ndarray,
    window_sizes: Tuple[int, ...] = (16, 32),
    num_hue_bins: int = 6,
) -> Dict[int, WindowColorMaps]:
    """Precompute neighborhood color maps for multiple window sizes."""
    is_ink, is_chroma_ink, hue = extract_ink_and_chroma(rgb)
    return {
        w: compute_window_maps(is_ink, is_chroma_ink, hue, w, num_hue_bins)
        for w in window_sizes
    }


def extract_robust_query_signature(
    rgb: np.ndarray,
    query_box: Tuple[int, int, int, int],
    num_hue_bins: int = 6,
    chroma_ratio_threshold: float = 0.20,
    min_chromatic_pixels: int = 40,
) -> QueryColorSignature:
    """Extract robust color signature from the exact query crop."""
    x0, y0, x1, y1 = query_box
    crop = rgb[y0:y1, x0:x1]

    is_ink, is_chroma_ink, hue = extract_ink_and_chroma(crop)
    total_ink = float(np.sum(is_ink))
    chroma_ink = float(np.sum(is_chroma_ink))

    frac = (chroma_ink / total_ink) if total_ink > 0.0 else 0.0
    bin_width = 180.0 / float(num_hue_bins)

    hist = np.zeros(num_hue_bins, dtype=np.float32)
    chroma_mask = is_chroma_ink > 0.0
    if np.any(chroma_mask):
        chroma_hues = hue[chroma_mask]
        bins = np.clip((chroma_hues / bin_width).astype(int), 0, num_hue_bins - 1)
        for bin_idx in range(num_hue_bins):
            hist[bin_idx] = float(np.sum(bins == bin_idx))

        # Robust noise trimming: zero bins with less than 10% of chromatic ink
        trim_threshold = 0.10 * chroma_ink
        hist[hist < trim_threshold] = 0.0
        hist_sum = float(np.sum(hist))
        if hist_sum > 0.0:
            hist = hist / hist_sum

    is_chromatic = (frac >= chroma_ratio_threshold) and (chroma_ink >= min_chromatic_pixels)
    return QueryColorSignature(
        is_chromatic=is_chromatic,
        chroma_fraction=frac,
        hue_histogram=hist,
    )


def compute_sample_mismatch(
    query_sig: QueryColorSignature,
    color_maps: WindowColorMaps,
    sample_y: np.ndarray,
    sample_x: np.ndarray,
    min_ink_threshold: float = 2.0,
) -> np.ndarray:
    """Compute color mismatch in [0, 1] at reviewed sample coordinates."""
    ink_cnt = color_maps.ink_count[sample_y, sample_x]
    chr_cnt = color_maps.chroma_count[sample_y, sample_x]
    hist_stack = color_maps.hue_histogram_stack[sample_y, sample_x]

    has_ink = ink_cnt >= min_ink_threshold
    local_chroma_fraction = np.zeros_like(ink_cnt)
    local_chroma_fraction[has_ink] = chr_cnt[has_ink] / ink_cnt[has_ink]

    if not query_sig.is_chromatic:
        # Achromatic query: mismatch equals local chromatic fraction
        return np.where(has_ink, local_chroma_fraction, 0.0)

    # Chromatic query: evaluate hue similarity on chromatic ink
    has_chroma = chr_cnt >= 1.0
    hue_similarity = np.zeros_like(ink_cnt)
    if np.any(has_chroma):
        norm_hist = hist_stack[has_chroma] / chr_cnt[has_chroma, None]
        q_hist = query_sig.hue_histogram[None, :]
        intersection = np.sum(np.minimum(norm_hist, q_hist), axis=1)
        hue_similarity[has_chroma] = intersection

    # Total mismatch: 1.0 - (fraction_chromatic * hue_similarity)
    mismatch = 1.0 - (local_chroma_fraction * hue_similarity)
    return np.where(has_ink, mismatch, 0.0)


def apply_gate(
    base_score: np.ndarray,
    mismatch: np.ndarray,
    gate_type: str = "linear",
    strength: float = 0.6,
) -> np.ndarray:
    """Apply soft color gate to baseline prediction scores."""
    if gate_type == "linear":
        gate = np.maximum(0.0, 1.0 - strength * mismatch)
    elif gate_type == "exponential":
        gate = np.exp(-strength * mismatch)
    elif gate_type == "power":
        clamped = np.clip(1.0 - mismatch, 0.0, 1.0)
        gate = np.power(clamped, strength)
    else:
        raise ValueError(f"Unknown gate_type: {gate_type}")

    return np.asarray(base_score * gate, dtype=np.float32)
