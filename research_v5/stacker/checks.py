"""Synthetic checks for window math, weights, and a tiny texture forward pass."""
from __future__ import annotations

import bootstrap  # noqa: F401

import numpy as np

from constants import CANDIDATES, GRID_STRIDE, MODEL_PATH, log, require
from features import (
    assemble,
    cell_index,
    feature_names,
    fraction_maps,
    gather,
    integral_image,
    mismatch_grid,
    signed_distance,
    window_statistics,
    window_sum,
)
from folds import assign_folds
from model import _classifier, document_balanced_weights


def run() -> None:
    _test_integral()
    _test_constant_window()
    _test_hole_context()
    _test_distance()
    _test_lookup()
    _test_weights()
    _test_folds()
    _test_feature_volume()
    _test_classifier_api()
    _test_texture_smoke()
    log("checks passed")


def _test_integral() -> None:
    values = np.array([[1.0, 2.0], [3.0, 4.0]])
    sat = integral_image(values)
    require(float(sat[2, 2]) == 10.0, "integral total")
    require(float(window_sum(values, 0)[1, 1]) == 4.0, "half-zero cell")
    require(float(window_sum(values, 1)[0, 0]) == 10.0, "full window")


def _test_constant_window() -> None:
    plane = np.full((6, 6), 0.25, dtype=np.float32)
    mean, maximum, p75, p90, frac, above = window_statistics(plane, 1, 0.25)
    require(np.allclose(mean, 0.25), "constant mean")
    require(np.allclose(maximum, 0.25), "constant max")
    require(np.allclose(frac, 0.0), "constant fraction")
    require(np.allclose(above, 0.0), "constant above median")
    center = (min(int(0.25 * 32), 31) + 0.5) / 32
    require(np.allclose(p75, center), "p75 bin")
    require(np.allclose(p90, center), "p90 bin")


def _test_hole_context() -> None:
    plane = np.full((9, 9), 0.1, dtype=np.float32)
    plane[2:7, 2:7] = 0.95
    plane[4, 4] = 0.02
    _mean, _maximum, _p75, p90, _frac, _above = window_statistics(plane, 2, 0.95)
    require(float(p90[4, 4]) > 0.8, "neighborhood p90 stays high on a hole")
    require(float(plane[4, 4]) < 0.1, "center itself stays low")


def _test_distance() -> None:
    centers_y = np.array([4, 12])
    centers_x = np.array([4, 12])
    distance = signed_distance(centers_y, centers_x, (8, 0, 16, 24))
    require(np.isclose(distance[0, 0], 4 / 256), "outside distance")
    require(np.isclose(distance[1, 1], -3 / 256), "inside distance")


def _test_lookup() -> None:
    grid = np.arange(4, dtype=np.float32).reshape(2, 2)
    y = np.array([0, 0, 8, 15])
    x = np.array([0, 8, 0, 15])
    found = gather(grid, y, x)
    require(np.allclose(found, [0, 1, 2, 3]), "cell lookup")
    iy, ix = cell_index(np.array([100]), np.array([100]), 2, 2)
    require(int(iy[0]) == 1 and int(ix[0]) == 1, "lookup clamps")
    require(GRID_STRIDE == 8, "stride")


def _test_weights() -> None:
    documents = np.array([0, 0, 0, 1, 1])
    labels = np.array([1, 0, 0, 1, 0])
    weights = document_balanced_weights(documents, labels)
    require(np.isclose(weights[(documents == 0) & (labels == 1)].sum(), 0.5), "doc0 pos")
    require(np.isclose(weights[(documents == 0) & (labels == 0)].sum(), 0.5), "doc0 neg")
    require(np.isclose(weights[documents == 1].sum(), 1.0), "doc1")


def _test_folds() -> None:
    rows = []
    for index, count in enumerate((5, 1, 4, 2, 3, 6)):
        for query in range(count):
            rows.append({"document_id": f"d{index}", "id": f"d{index}-{query}", "image": "a"})
    assigned = assign_folds(rows)
    require(set(assigned) == {f"d{index}" for index in range(6)}, "all documents")
    require(set(assigned.values()) <= {0, 1, 2, 3}, "fold ids")


def _test_feature_volume() -> None:
    from color.color_features import (
        compute_window_maps,
        extract_ink_and_chroma,
        extract_robust_query_signature,
    )
    from texture_learning import grid_centers

    rgb = np.random.default_rng(0).integers(0, 255, size=(32, 32, 3), dtype=np.uint8)
    prob = np.full((4, 4), 0.4, dtype=np.float32)
    ink, chroma, hue = extract_ink_and_chroma(rgb)
    centers_y, centers_x = grid_centers(32, 32, GRID_STRIDE)
    signature = extract_robust_query_signature(rgb, (4, 4, 20, 20))
    mismatches = {
        window: mismatch_grid(
            signature, compute_window_maps(ink, chroma, hue, window), centers_y, centers_x
        )
        for window in (32, 128)
    }
    volume = assemble(
        prob,
        (4, 4, 20, 20),
        32,
        32,
        fraction_maps(ink, centers_y, centers_x),
        fraction_maps(chroma, centers_y, centers_x),
        mismatches,
    )
    require(len(feature_names()) == 75, "feature count")
    require(volume.shape == (4, 4, 75), f"volume {volume.shape}")
    require(bool(np.isfinite(volume).all()), "finite volume")


def _test_classifier_api() -> None:
    model = _classifier(CANDIDATES[1])
    model.set_params(max_iter=5, min_samples_leaf=1)
    features = np.array([[0.0], [1.0], [0.2], [0.8]], dtype=np.float32)
    labels = np.array([0, 1, 0, 1])
    model.fit(features, labels, sample_weight=np.ones(4, dtype=np.float32))
    probabilities = model.predict_proba(features)
    require(probabilities.shape == (4, 2), "proba shape")


def _test_texture_smoke() -> None:
    import hatch_texture
    from drawing import _probability
    from inference_texture import verify_model_metadata
    from texture_learning import load_texture_model

    model, metadata = load_texture_model(MODEL_PATH)
    verify_model_metadata(metadata)
    gray = np.random.default_rng(1).integers(0, 255, size=(64, 64), dtype=np.uint8)
    features = hatch_texture.texture_features(gray, stride=GRID_STRIDE)
    probability = _probability(model, gray, features, (8, 8, 40, 40))
    require(probability.shape == (8, 8), f"smoke shape {probability.shape}")
    require(bool(np.isfinite(probability).all()), "smoke finite")
    require(float(probability.min()) >= 0.0, "smoke low")
    require(float(probability.max()) <= 1.0, "smoke high")


if __name__ == "__main__":
    run()
