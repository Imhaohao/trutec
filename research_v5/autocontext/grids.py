"""Build a stage-3 feature volume from one stage-2 context volume.

The stage-2 probability grid is the frozen scorer's full-lattice prediction.
Stage-1 probability, signed distance, and the w32 color mismatch are channels
the stacker already computed from the drawing and the query box.
"""
from __future__ import annotations

import limits  # noqa: F401

import numpy as np

from context_features import context_volume
from features import feature_names as stacker_feature_names
from inference_v6 import stage2_grid
from stage3_score import score_grid


def feature_volume_from_stage2(volume: np.ndarray, stage2) -> np.ndarray:
    score = stage2_grid(stage2, volume)
    names = stacker_feature_names()
    stage1 = volume[:, :, names.index("cell_score")].copy()
    distance = volume[:, :, names.index("signed_distance_over_256")].copy()
    mismatch = volume[:, :, names.index("color_mismatch_w32")].copy()
    return context_volume(score, stage1, distance, mismatch)


def stage3_score_grid(volume: np.ndarray, stage2, stage3) -> np.ndarray:
    features = feature_volume_from_stage2(volume, stage2)
    grid = score_grid(stage3, features)
    del features
    return grid
