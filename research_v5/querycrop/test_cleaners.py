"""Unit tests for query cleaning rules."""
from __future__ import annotations

import bootstrap  # noqa: F401

import numpy as np

from cleaners import (
    clean_crop_components,
    clean_descriptor_autocorr,
    clean_descriptor_combined,
    clean_descriptor_components,
    clean_descriptor_iqr,
)


def test_rule_a_iqr_outlier_filtering():
    """Test that IQR outlier cell rejection drops noisy cells and handles edge cases."""
    # 1. Uniform background with periodic hatch
    hatch = np.full((64, 64), 255, dtype=np.uint8)
    hatch[::8, :] = 0  # horizontal hatch lines

    # Pure descriptor
    desc_clean = clean_descriptor_iqr(hatch, stride=4)
    assert desc_clean.shape == (46,)
    assert np.all(np.isfinite(desc_clean))

    # Add a huge solid black patch in one quadrant
    noisy_hatch = hatch.copy()
    noisy_hatch[:16, :16] = 0

    desc_noisy = clean_descriptor_iqr(noisy_hatch, stride=4)
    assert desc_noisy.shape == (46,)
    assert np.all(np.isfinite(desc_noisy))

    # Edge case: tiny crop (8x8 -> 4 cells at stride 4)
    tiny_crop = np.full((8, 8), 200, dtype=np.uint8)
    desc_tiny = clean_descriptor_iqr(tiny_crop, stride=4)
    assert desc_tiny.shape == (46,)


def test_rule_b_autocorr_exemplar():
    """Test that autocorrelation sub-window selection works on periodic patterns."""
    # Create periodic stripe pattern
    gray = np.full((128, 128), 255, dtype=np.uint8)
    gray[::8, :] = 50  # 8-pixel period stripes
    rgb = np.repeat(gray[:, :, None], 3, axis=2)

    box = (0, 0, 128, 128)
    desc = clean_descriptor_autocorr(rgb, gray, box, stride=4)
    assert desc.shape == (46,)
    assert np.all(np.isfinite(desc))

    # Test plain non-periodic image fallback
    plain_gray = np.full((64, 64), 240, dtype=np.uint8)
    plain_rgb = np.repeat(plain_gray[:, :, None], 3, axis=2)
    plain_desc = clean_descriptor_autocorr(plain_rgb, plain_gray, (0, 0, 64, 64), stride=4)
    assert plain_desc.shape == (46,)
    assert np.all(np.isfinite(plain_desc))


def test_rule_c_connected_components():
    """Test that text glyphs and long border lines are cleanly removed."""
    # Pattern: small dots/dashes
    crop = np.full((100, 100), 255, dtype=np.uint8)
    for y in range(10, 90, 15):
        for x in range(10, 90, 15):
            crop[y : y + 3, x : x + 3] = 0  # 3x3 dot (area 9)

    # Overlaid text letter: solid 16x16 block (area 256)
    crop[40:56, 40:56] = 0

    # Overlaid long boundary line: 1x70 line (width 70, height 1)
    crop[2, 10:80] = 0

    cleaned = clean_crop_components(crop, paper_level=210)

    # Letter and long line should be erased to white background
    assert np.all(cleaned[40:56, 40:56] >= 200)
    assert np.all(cleaned[2, 10:80] >= 200)

    # Dots outside the letter should still be present
    assert np.any(cleaned[10:13, 10:13] == 0)

    # Descriptor computation
    desc = clean_descriptor_components(crop, paper_level=210, stride=4)
    assert desc.shape == (46,)
    assert np.all(np.isfinite(desc))


def test_combined_cleaning_rule():
    """Test combined cleaning pipeline."""
    gray = np.full((96, 96), 255, dtype=np.uint8)
    gray[::6, :] = 0
    gray[30:50, 30:50] = 0  # text block
    rgb = np.repeat(gray[:, :, None], 3, axis=2)

    desc = clean_descriptor_combined(rgb, gray, (0, 0, 96, 96), stride=4)
    assert desc.shape == (46,)
    assert np.all(np.isfinite(desc))
