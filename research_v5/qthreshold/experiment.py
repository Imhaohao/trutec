"""Choices fixed before any development score is read."""
from __future__ import annotations

QUANTILES = (0.01, 0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
OTSU_BINS = 256
TARGET_STEPS = 1001
RECALL_FLOOR = 0.98
MEDIAN_FRACTIONS = (0.25, 0.5, 0.75, 1.0)
CLIP_LOW = 0.05
CLIP_HIGH = 0.9
COLOR_WINDOW = 16
COLOR_STRENGTH = 0.6
COLOR_GATE = "linear"
RIDGE_ALPHAS = (0.1, 1.0, 10.0)
TREE_SPECS = ((7, 80), (15, 120))
GLOBAL_V6_THRESHOLD = 0.44


def experiment_settings() -> dict:
    return {
        "score_map": "v6 gated native map: stride-8 stage-2 cell lookup, then w16 linear color gate strength 0.6",
        "fit_models": "stage-2 fold model whose held-out fold contains the query document, with that fold's stage-1 model",
        "dev_models": "frozen stacker stage2_model.joblib and the original crop HGB stage-1",
        "features": "full-map quantiles, fractions strictly above 0.1..0.9, query-box median, 256-bin Otsu, fraction of pixels >= Otsu, box width/height/area",
        "targets": "fit reviewed pixels only; 1001 thresholds from 0 to 1; IoU-maximizing cutoff (largest on ties); largest cutoff with recall >= 0.98",
        "model_selection": "document-grouped OOF document-macro IoU on fit, raw predicted cutoff; tie break higher pooled precision then name",
        "sample_weight": "each training document has equal total weight",
        "median_fractions": list(MEDIAN_FRACTIONS),
        "fraction_selection": "full fit document-macro IoU; tie break higher pooled precision then larger fraction",
        "clip": [CLIP_LOW, CLIP_HIGH],
        "clip_role": "reported blend, not the model-selection objective",
        "threads": 4,
        "memory_limit_gb": 10,
        "color_window": COLOR_WINDOW,
        "color_strength": COLOR_STRENGTH,
        "color_gate": COLOR_GATE,
    }
