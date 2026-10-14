"""Fixed experiment settings. Chosen before any development score is read."""
from __future__ import annotations

from bootstrap import ROOT

STACKER = ROOT / "research_v5" / "stacker"
CACHE = STACKER / "cache"
DATA_ROOT = ROOT / "dataset"
MODEL_PATH = ROOT / "models" / "models" / "texture-hgb-crop-v1.joblib"
FIT_MANIFEST = ROOT / "artifacts" / "splits" / "fit.json"

MODEL_STRIDE = 4
GRID_STRIDE = 8
QUERY_SOURCE = "crop"
FEATURE_SET = "base"
STAGE1_SEED = 42
SAMPLES_PER_CLASS = 512
STAGE1_MAX_ITER = 200
FOLD_COUNT = 4
RADII = (16, 32, 64, 128, 256)
SCORE_STATS = ("mean", "max", "p75", "p90", "frac_gt_0_5")
COLOR_WINDOWS = (32, 128)
HISTOGRAM_BINS = 32
DISTANCE_SCALE = 256.0
CELLS_PER_CLASS = 2000
CV_PER_CLASS = 400
STAGE2_SEED = 42
PREDICT_BATCH = 65536
FEATURE_VERSION = 1

EXPECTED_STAGE1_SAMPLES = 331422
EXPECTED_STAGE1_POSITIVE = 163578
EXPECTED_STAGE1_NEGATIVE = 167844

CANDIDATES = (
    {
        "name": "shallow",
        "learning_rate": 0.1,
        "max_leaf_nodes": 15,
        "l2_regularization": 1.0,
        "max_iter": 150,
        "min_samples_leaf": 40,
    },
    {
        "name": "recipe",
        "learning_rate": 0.08,
        "max_leaf_nodes": 31,
        "l2_regularization": 1.0,
        "max_iter": 200,
        "min_samples_leaf": 20,
    },
    {
        "name": "deep",
        "learning_rate": 0.06,
        "max_leaf_nodes": 63,
        "l2_regularization": 0.1,
        "max_iter": 250,
        "min_samples_leaf": 20,
    },
)


def log(message: str) -> None:
    print(message, flush=True)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def settings_snapshot() -> dict:
    return {
        "model_path": str(MODEL_PATH),
        "model_stride": MODEL_STRIDE,
        "grid_stride": GRID_STRIDE,
        "query_descriptor_source": QUERY_SOURCE,
        "texture_feature_set": FEATURE_SET,
        "stage1_seed": STAGE1_SEED,
        "samples_per_class": SAMPLES_PER_CLASS,
        "stage1_learning_rate": 0.08,
        "stage1_max_leaf_nodes": 31,
        "stage1_l2_regularization": 1.0,
        "stage1_max_iter": STAGE1_MAX_ITER,
        "stage1_early_stopping": False,
        "stage1_seed_rule": (
            "seed = 42 + query index in train_texture.py grouped-image order "
            "on the full fit manifest"
        ),
        "fold_count": FOLD_COUNT,
        "fold_rule": "greedy lightest-fold by query count; document_id breaks ties",
        "radii_px": list(RADII),
        "window_rule": (
            "score windows use Chebyshev half-width radius//8 cells; "
            "ink and chroma use native squares of half-width radius "
            "(side 2*radius+1), clipped to the image and divided by the clipped area"
        ),
        "score_stats": list(SCORE_STATS),
        "histogram_bins": HISTOGRAM_BINS,
        "color_windows": list(COLOR_WINDOWS),
        "distance_scale_px": DISTANCE_SCALE,
        "cells_per_class": CELLS_PER_CLASS,
        "cv_per_class": CV_PER_CLASS,
        "stage2_candidates": list(CANDIDATES),
        "stage2_seed": STAGE2_SEED,
        "expected_stage1_samples": EXPECTED_STAGE1_SAMPLES,
        "feature_version": FEATURE_VERSION,
    }
