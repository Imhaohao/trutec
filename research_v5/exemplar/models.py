"""Positive-unlabeled heads for one query, then a [0, 1] score on the full grid.

Training rows are descriptor cells. The decision at every grid cell is a linear
function of that cell alone, so sampling a subset of pixels cannot change it.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import map_coordinates, maximum_filter, minimum_filter
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from descriptors import STRIDE, Channel

UNLABELED_LIMIT = 6000
KNN_CAP = 96
KNN_NEIGHBORS = 5
PCA_DIMS = 16


def kept_columns(
    table: tuple[Channel, ...],
    support: int,
    families: frozenset[str] | None = None,
) -> np.ndarray:
    """Columns whose window fits inside the query, optionally limited by family."""
    chosen = [
        index
        for index, channel in enumerate(table)
        if channel.window <= support and (families is None or channel.kind in families)
    ]
    if len(chosen) < 4:
        raise RuntimeError("descriptor mask kept fewer than 4 channels")
    return np.asarray(chosen, dtype=np.int32)


def unlabeled_rows(
    grid_y: np.ndarray,
    grid_x: np.ndarray,
    box: tuple[int, int, int, int],
    limit: int = UNLABELED_LIMIT,
) -> np.ndarray:
    """Spatially strided grid cells whose centers lie outside the query box."""
    x0, y0, x1, y1 = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
    inside_y = (grid_y >= y0) & (grid_y < y1)
    inside_x = (grid_x >= x0) & (grid_x < x1)
    eligible = np.flatnonzero(~(inside_y[:, None] & inside_x[None, :]).ravel())
    if eligible.size == 0:
        return np.zeros(1, dtype=np.int64)
    if eligible.size <= limit:
        return eligible.astype(np.int64)
    step = max(1, int(eligible.size // limit))
    return eligible[::step][:limit].astype(np.int64)


def pack_training(
    grid: np.ndarray,
    positives: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat = np.ascontiguousarray(grid.reshape(-1, grid.shape[-1])[:, columns], dtype=np.float32)
    pos = np.ascontiguousarray(positives[:, columns], dtype=np.float32)
    unlabeled = np.ascontiguousarray(flat[rows], dtype=np.float32)
    return flat, pos, unlabeled


def sigmoid_anchor(scores: np.ndarray, positive_scores: np.ndarray, quantile: float) -> np.ndarray:
    """Map scores so the given percentile of query cells sits at 0.5."""
    anchor = float(np.percentile(positive_scores, quantile))
    iqr = float(np.percentile(positive_scores, 80) - np.percentile(positive_scores, 20))
    spread = max(iqr, float(np.std(positive_scores)), 1e-3)
    shaped = (scores - anchor) / spread
    curved = 1.0 / (1.0 + np.exp(-np.clip(shaped, -12.0, 12.0)))
    return curved.astype(np.float32)


def blur_score(score: np.ndarray, radius: int) -> np.ndarray:
    kernel = radius * 2 + 1
    blurred = cv2.blur(np.ascontiguousarray(score, dtype=np.float32), (kernel, kernel))
    return blurred.astype(np.float32)


def sample_bilinear(
    score: np.ndarray,
    grid_y: np.ndarray,
    grid_x: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    """Bilinear lookup on the score grid. Same value a full-image map would give."""
    stride = int(grid_y[1] - grid_y[0]) if grid_y.size > 1 else STRIDE
    fy = (y.astype(np.float64) - float(grid_y[0])) / stride
    fx = (x.astype(np.float64) - float(grid_x[0])) / stride
    clean = np.nan_to_num(score, nan=0.0, posinf=1.0, neginf=0.0)
    sampled = map_coordinates(clean, np.vstack([fy, fx]), order=1, mode="nearest", prefilter=False)
    return np.clip(sampled, 0.0, 1.0).astype(np.float32)


def logreg_scores(
    flat: np.ndarray,
    positives: np.ndarray,
    unlabeled: np.ndarray,
    height: int,
    width: int,
    cost: float,
    balanced: bool,
    quantile: float,
) -> np.ndarray:
    anchored = logreg_quantiles(flat, positives, unlabeled, height, width, cost, balanced, (quantile,))
    return anchored[quantile]


def logreg_quantiles(
    flat: np.ndarray,
    positives: np.ndarray,
    unlabeled: np.ndarray,
    height: int,
    width: int,
    cost: float,
    balanced: bool,
    quantiles: tuple[float, ...],
) -> dict[float, np.ndarray]:
    decision, positive_decision = _linear_decision(
        flat, positives, unlabeled, height, width, cost, balanced, "logreg"
    )
    return {
        quantile: sigmoid_anchor(decision, positive_decision, quantile) for quantile in quantiles
    }


def trimmed_logreg_scores(
    flat: np.ndarray,
    positives: np.ndarray,
    unlabeled: np.ndarray,
    height: int,
    width: int,
    cost: float,
    quantile: float,
) -> np.ndarray:
    kept = _trim_positives(positives, 0.7)
    return logreg_scores(flat, kept, unlabeled, height, width, cost, True, quantile)


def svm_scores(
    flat: np.ndarray,
    positives: np.ndarray,
    unlabeled: np.ndarray,
    height: int,
    width: int,
    cost: float,
    quantile: float,
) -> np.ndarray:
    decision, positive_decision = _linear_decision(
        flat, positives, unlabeled, height, width, cost, True, "svm"
    )
    return sigmoid_anchor(decision, positive_decision, quantile)


def prototype_ratio(
    flat: np.ndarray,
    positives: np.ndarray,
    unlabeled: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    """Diagonal Mahalanobis: distance to the unlabeled mean versus the positive median."""
    mu, scale = _whiten_fit(unlabeled)
    proto = np.median(_whiten(positives, mu, scale), axis=0)
    return _distance_ratio(_whiten(flat, mu, scale), proto, height, width)


def dilate_score(score: np.ndarray, size: int) -> np.ndarray:
    """Max in a square neighborhood, so paper between strokes inherits a nearby score."""
    filled = maximum_filter(_as_float_grid(score), size=size, mode="nearest")
    return filled.astype(np.float32)


def close_score(score: np.ndarray, size: int) -> np.ndarray:
    """Grayscale closing: fill holes inside a high region without growing its outline."""
    filled = maximum_filter(_as_float_grid(score), size=size, mode="nearest")
    closed = minimum_filter(filled, size=size, mode="nearest")
    return closed.astype(np.float32)


def _as_float_grid(score: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(score, dtype=np.float32)


def logreg_grid_scale(
    flat: np.ndarray,
    positives: np.ndarray,
    unlabeled: np.ndarray,
    height: int,
    width: int,
    cost: float = 0.3,
) -> np.ndarray:
    """Sigmoid whose width is the whole-sheet spread, anchored at the weakest query cell."""
    decision, positive_decision = _linear_decision(
        flat, positives, unlabeled, height, width, cost, True, "logreg"
    )
    anchor = float(np.min(positive_decision))
    spread = max(float(np.std(decision)), 1e-3)
    shaped = (decision - anchor) / spread
    curved = 1.0 / (1.0 + np.exp(-np.clip(shaped, -12.0, 12.0)))
    return curved.astype(np.float32)


def trimmed_knn_ratio(
    flat: np.ndarray,
    positives: np.ndarray,
    unlabeled: np.ndarray,
    height: int,
    width: int,
    neighbors: int = 1,
) -> np.ndarray:
    kept = _trim_positives(positives, 0.7)
    return knn_ratio(flat, kept, unlabeled, height, width, neighbors)


def knn_ratio(
    flat: np.ndarray,
    positives: np.ndarray,
    unlabeled: np.ndarray,
    height: int,
    width: int,
    neighbors: int = KNN_NEIGHBORS,
) -> np.ndarray:
    """Mean distance to nearby query cells versus nearby unlabeled cells."""
    mu, scale = _whiten_fit(unlabeled)
    pos = _row_stride(_whiten(positives, mu, scale), KNN_CAP)
    background = _row_stride(_whiten(unlabeled, mu, scale), KNN_CAP)
    z = _whiten(flat, mu, scale)
    near_pos = _mean_neighbor_distance(z, pos, neighbors)
    near_bg = _mean_neighbor_distance(z, background, neighbors)
    ratio = near_bg / (near_pos + near_bg + 1e-6)
    return ratio.reshape(height, width).astype(np.float32)


def mahalanobis_ratio(
    flat: np.ndarray,
    positives: np.ndarray,
    unlabeled: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    """Mahalanobis distance in a 16-D PCA of the unlabeled cloud."""
    mu, scale = _whiten_fit(unlabeled)
    z_unl = _whiten(unlabeled, mu, scale)
    dims = min(PCA_DIMS, z_unl.shape[0] - 1, z_unl.shape[1])
    pca = PCA(n_components=dims, random_state=0)
    background = pca.fit_transform(z_unl)
    proto = np.median(pca.transform(_whiten(positives, mu, scale)), axis=0)
    center = background.mean(axis=0)
    cov = np.cov(background, rowvar=False) + np.eye(dims) * 1e-2
    embedded = pca.transform(_whiten(flat, mu, scale))
    d_pos = _mahalanobis(embedded, proto, cov)
    d_bg = _mahalanobis(embedded, center, cov)
    ratio = d_bg / (d_pos + d_bg + 1e-6)
    return ratio.reshape(height, width).astype(np.float32)


def _linear_decision(flat, positives, unlabeled, height, width, cost, balanced, kind):
    train = np.concatenate([positives, unlabeled], axis=0)
    labels = np.zeros(train.shape[0], dtype=np.int8)
    labels[: positives.shape[0]] = 1
    scaler = StandardScaler()
    scaled = scaler.fit_transform(train)
    model = _linear_model(kind, cost, balanced)
    model.fit(scaled, labels)
    grid_decision = model.decision_function(scaler.transform(flat)).reshape(height, width)
    positive_decision = model.decision_function(scaler.transform(positives))
    return grid_decision.astype(np.float32), positive_decision.astype(np.float32)


def _linear_model(kind: str, cost: float, balanced: bool):
    weight = "balanced" if balanced else None
    if kind == "svm":
        return LinearSVC(C=cost, class_weight=weight, dual=False, max_iter=4000, tol=1e-3)
    return LogisticRegression(
        C=cost, class_weight=weight, solver="lbfgs", max_iter=400, tol=1e-3
    )


def _trim_positives(positives: np.ndarray, keep: float) -> np.ndarray:
    center = np.median(positives, axis=0)
    distance = np.linalg.norm(positives - center, axis=1)
    kept = positives[distance <= np.quantile(distance, keep)]
    if kept.shape[0] < 8:
        return positives
    return kept


def _whiten_fit(unlabeled: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = unlabeled.mean(axis=0).astype(np.float32)
    scale = np.maximum(unlabeled.std(axis=0), 1e-3).astype(np.float32)
    return mu, scale


def _whiten(values: np.ndarray, mu: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((values - mu) / scale).astype(np.float32)


def _distance_ratio(embedded: np.ndarray, proto: np.ndarray, height: int, width: int) -> np.ndarray:
    toward_pos = np.linalg.norm(embedded - proto, axis=1)
    toward_bg = np.linalg.norm(embedded, axis=1)
    ratio = toward_bg / (toward_pos + toward_bg + 1e-6)
    return ratio.reshape(height, width).astype(np.float32)


def _row_stride(values: np.ndarray, limit: int) -> np.ndarray:
    if values.shape[0] <= limit:
        return values
    step = max(1, values.shape[0] // limit)
    return np.ascontiguousarray(values[::step][:limit])


def _mean_neighbor_distance(values: np.ndarray, proto: np.ndarray, neighbors: int) -> np.ndarray:
    k_use = min(neighbors, proto.shape[0])
    squared_x = np.sum(values * values, axis=1, dtype=np.float32)
    squared_p = np.sum(proto * proto, axis=1, dtype=np.float32)
    means = np.empty(values.shape[0], dtype=np.float32)
    proto_t = np.ascontiguousarray(proto.T)
    for start in range(0, values.shape[0], 20000):
        block = values[start : start + 20000]
        dist = squared_x[start : start + block.shape[0], None] + squared_p[None, :]
        dist -= 2.0 * (block @ proto_t)
        np.maximum(dist, 0.0, out=dist)
        nearest = np.partition(dist, k_use - 1, axis=1)[:, :k_use]
        means[start : start + block.shape[0]] = np.mean(np.sqrt(nearest), axis=1)
    return means


def _mahalanobis(embedded: np.ndarray, center: np.ndarray, cov: np.ndarray) -> np.ndarray:
    delta = embedded - center
    solved = np.linalg.solve(cov, delta.T).T
    return np.sqrt(np.maximum(np.sum(delta * solved, axis=1), 0.0))
