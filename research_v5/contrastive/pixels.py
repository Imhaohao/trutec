"""RGB plus luminance packing. No labels."""

from __future__ import annotations

import numpy as np

from constants import INPUT_CHANNELS


def pack_channels(rgb: np.ndarray) -> np.ndarray:
    """Return float32 CHW with channels (R, G, B, luminance), values in [0, 1]."""
    array = np.asarray(rgb, dtype=np.float32)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("expected an HxWx3 RGB image")
    if float(np.max(array)) > 1.5:
        array = array / 255.0
    luma = 0.2126 * array[..., 0] + 0.7152 * array[..., 1] + 0.0722 * array[..., 2]
    packed = np.concatenate((array, luma[..., None]), axis=-1)
    if packed.shape[2] != INPUT_CHANNELS:
        raise ValueError("packed image does not have four channels")
    return np.ascontiguousarray(packed.transpose(2, 0, 1))
