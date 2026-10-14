"""Drawing-level pattern competition matcher for label-free hatch matching.

Discovers distinct texture patterns across an architectural sheet using
unsupervised prototype clustering, then scores pixels by their margin of
similarity between the query prototype and competing drawing prototypes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans

_INK_WINDOWS = (8, 16, 32, 64, 128)
_CHROMA_WINDOWS = (16, 32, 64)
_GRAD_WINDOWS = (16, 32)
_ST_WINDOWS = (12, 24, 48)
_GABOR_WAVELENGTHS = (4, 6, 8, 12, 16, 24, 32)
_GABOR_ANGLES = (0, 30, 60, 90, 120, 150)
_GRAD_ANGLES = (0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5)


@dataclass(frozen=True)
class MatcherConfig:
    stride: int = 8
    n_clusters: int = 20
    competitor_margin_threshold: float = 0.12
    min_similarity_same: float = 0.72
    blank_ink_threshold: float = 0.003
    spatial_blur_radius: int = 9
    spatial_blur_sigma: float = 2.2
    score_mapping_mode: str = "linear"  # "linear" or "sigmoid"
    score_mapping_scale: float = 1.0
    score_mapping_offset: float = 0.5
    sigmoid_gain: float = 8.0
    sigmoid_bias: float = 0.05
    include_blank_prototype: bool = True
    use_softmax_margin: bool = False
    softmax_temperature: float = 5.0
    use_bilateral_smoothing: bool = False
    query_trim_percentile: float = 10.0


def _compute_ink_channels(
    ink: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
) -> list[np.ndarray]:
    """Compute multi-scale luminance ink density and variance."""
    channels: list[np.ndarray] = []
    for w in _INK_WINDOWS:
        m_ink = cv2.boxFilter(ink, cv2.CV_32F, (w, w))
        v_ink = np.maximum(0.0, cv2.boxFilter(ink * ink, cv2.CV_32F, (w, w)) - m_ink * m_ink)
        channels.extend([
            m_ink[np.ix_(ys, xs)],
            v_ink[np.ix_(ys, xs)],
        ])
    return channels


def _compute_chroma_channels(
    rgb: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
) -> list[np.ndarray]:
    """Compute multi-scale chroma and color deviation channels."""
    rgb_f = rgb.astype(np.float32) / 255.0
    rg_diff = np.abs(rgb_f[:, :, 0] - rgb_f[:, :, 1])
    gb_diff = np.abs(rgb_f[:, :, 1] - rgb_f[:, :, 2])
    br_diff = np.abs(rgb_f[:, :, 2] - rgb_f[:, :, 0])
    color_dev = (rg_diff + gb_diff + br_diff) * 0.5

    channels: list[np.ndarray] = []
    for w in _CHROMA_WINDOWS:
        m_chr = cv2.boxFilter(color_dev, cv2.CV_32F, (w, w))
        v_chr = np.maximum(0.0, cv2.boxFilter(color_dev * color_dev, cv2.CV_32F, (w, w)) - m_chr * m_chr)
        channels.extend([
            m_chr[np.ix_(ys, xs)],
            v_chr[np.ix_(ys, xs)],
        ])
    return channels


def _compute_structure_tensor_channels(
    dx: np.ndarray,
    dy: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
) -> list[np.ndarray]:
    """Compute structure tensor orientation and coherence channels at multiple scales."""
    channels: list[np.ndarray] = []
    for w in _ST_WINDOWS:
        jxx = cv2.boxFilter(dx * dx, cv2.CV_32F, (w, w))
        jyy = cv2.boxFilter(dy * dy, cv2.CV_32F, (w, w))
        jxy = cv2.boxFilter(dx * dy, cv2.CV_32F, (w, w))
        diff = jxx - jyy
        hyp = np.sqrt(diff * diff + 4.0 * jxy * jxy)
        tr = jxx + jyy
        coh = np.where(tr > 1e-5, hyp / (tr + 1e-5), 0.0)
        c2t = np.where(hyp > 1e-5, diff / (hyp + 1e-5), 0.0) * coh
        s2t = np.where(hyp > 1e-5, 2.0 * jxy / (hyp + 1e-5), 0.0) * coh
        channels.extend([
            coh[np.ix_(ys, xs)],
            c2t[np.ix_(ys, xs)],
            s2t[np.ix_(ys, xs)],
            tr[np.ix_(ys, xs)],
        ])
    return channels


def _compute_gradient_histograms(
    dx: np.ndarray,
    dy: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
) -> list[np.ndarray]:
    """Compute gradient energy along 8 orientation directions."""
    channels: list[np.ndarray] = []
    for ang in _GRAD_ANGLES:
        rad = float(np.deg2rad(ang))
        proj = dx * np.cos(rad) + dy * np.sin(rad)
        p2 = proj * proj
        for w in _GRAD_WINDOWS:
            pooled = cv2.boxFilter(p2, cv2.CV_32F, (w, w))
            channels.append(pooled[np.ix_(ys, xs)])
    return channels


def _compute_gabor_channels(
    gray: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
) -> list[np.ndarray]:
    """Compute multi-scale Gabor orientation and period energy."""
    height, width = gray.shape
    half_h = max(1, height // 2)
    half_w = max(1, width // 2)
    half_gray = cv2.resize(gray, (half_w, half_h), interpolation=cv2.INTER_AREA)
    half_ys = np.minimum(ys // 2, half_h - 1)
    half_xs = np.minimum(xs // 2, half_w - 1)

    channels: list[np.ndarray] = []
    for wl in _GABOR_WAVELENGTHS:
        for ang in _GABOR_ANGLES:
            rad = float(np.deg2rad(ang))
            ksize = int(max(7, min(25, 3 * wl + 1)))
            if ksize % 2 == 0:
                ksize += 1
            cos_k = cv2.getGaborKernel((ksize, ksize), wl * 0.5, rad, wl, 0.5, 0, ktype=cv2.CV_32F)
            sin_k = cv2.getGaborKernel(
                (ksize, ksize), wl * 0.5, rad, wl, 0.5, np.pi / 2, ktype=cv2.CV_32F
            )
            r_resp = cv2.filter2D(half_gray, cv2.CV_32F, cos_k)
            i_resp = cv2.filter2D(half_gray, cv2.CV_32F, sin_k)
            energy = np.sqrt(r_resp * r_resp + i_resp * i_resp)
            e_pooled = cv2.boxFilter(energy, cv2.CV_32F, (16, 16))
            channels.append(e_pooled[np.ix_(half_ys, half_xs)])
    return channels


def extract_texture_grid(
    rgb: np.ndarray,
    stride: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute multi-scale local texture descriptors on a stride grid.

    Returns:
        features: float32 array [grid_h, grid_w, channels]
        ink_grid: float32 array [grid_h, grid_w] of local ink density
        ys: native row coordinate centers
        xs: native col coordinate centers
    """
    height, width, _ = rgb.shape
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    ink = 1.0 - gray

    grid_h = math.ceil(height / stride)
    grid_w = math.ceil(width / stride)
    ys = np.minimum(np.arange(grid_h, dtype=np.int32) * stride + stride // 2, height - 1)
    xs = np.minimum(np.arange(grid_w, dtype=np.int32) * stride + stride // 2, width - 1)

    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    channels: list[np.ndarray] = []
    channels.extend(_compute_ink_channels(ink, ys, xs))
    channels.extend(_compute_chroma_channels(rgb, ys, xs))
    channels.extend(_compute_structure_tensor_channels(dx, dy, ys, xs))
    channels.extend(_compute_gradient_histograms(dx, dy, ys, xs))
    channels.extend(_compute_gabor_channels(gray, ys, xs))

    features = np.stack(channels, axis=-1).astype(np.float32)
    ink_grid = channels[2]
    return features, ink_grid, ys, xs


def _normalize_vectors(
    vectors: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Standardize feature vectors and normalize to unit L2 length."""
    scaled = (vectors - mean) / std
    norm = np.linalg.norm(scaled, axis=1, keepdims=True) + 1e-6
    return scaled / norm


def discover_prototypes(
    flat_features: np.ndarray,
    flat_ink: np.ndarray,
    config: MatcherConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cluster non-blank cells to discover drawing pattern prototypes."""
    non_blank = flat_ink > config.blank_ink_threshold
    min_cells = max(32, config.n_clusters * 4)
    if np.count_nonzero(non_blank) < min_cells:
        non_blank = np.ones(flat_ink.shape, dtype=bool)

    ink_features = flat_features[non_blank]
    mean = np.mean(ink_features, axis=0, keepdims=True)
    std = np.std(ink_features, axis=0, keepdims=True) + 1e-4

    normalized_ink = _normalize_vectors(ink_features, mean, std)

    clusterer = MiniBatchKMeans(
        n_clusters=config.n_clusters,
        batch_size=4096,
        random_state=42,
        n_init=3,
        max_iter=50,
    )
    clusterer.fit(normalized_ink)
    centers = clusterer.cluster_centers_
    centers = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-6)

    prototypes_list = [centers]
    if config.include_blank_prototype and np.any(~non_blank):
        blank_features = flat_features[~non_blank]
        blank_mean = np.mean(blank_features, axis=0, keepdims=True)
        blank_norm = _normalize_vectors(blank_mean, mean, std)
        prototypes_list.append(blank_norm)

    all_prototypes = np.concatenate(prototypes_list, axis=0)
    return all_prototypes, mean, std, non_blank


def _trim_query_vectors(
    q_vectors: np.ndarray,
    percentile: float,
) -> np.ndarray:
    """Filter out extreme outliers (borders/text) from query box cells."""
    if len(q_vectors) <= 4 or percentile <= 0.0:
        return q_vectors
    lower = float(percentile)
    upper = float(100.0 - percentile)
    q_low = np.percentile(q_vectors, lower, axis=0, keepdims=True)
    q_high = np.percentile(q_vectors, upper, axis=0, keepdims=True)
    in_range = (q_vectors >= q_low) & (q_vectors <= q_high)
    valid_mask = np.mean(in_range, axis=1) >= 0.5
    if np.any(valid_mask):
        return q_vectors[valid_mask]
    return q_vectors


def extract_query_prototype(
    flat_features: np.ndarray,
    flat_ink: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
    query_box: tuple[int, int, int, int],
    mean: np.ndarray,
    std: np.ndarray,
    config: MatcherConfig,
) -> np.ndarray:
    """Compute normalized query prototype vector from cells inside query box."""
    x0, y0, x1, y1 = query_box
    mask_y = (ys >= y0) & (ys < y1)
    mask_x = (xs >= x0) & (xs < x1)
    q_grid = np.outer(mask_y, mask_x).reshape(-1)

    q_ink = q_grid & (flat_ink > config.blank_ink_threshold)
    if np.count_nonzero(q_ink) == 0:
        q_ink = q_grid
    if np.count_nonzero(q_ink) == 0:
        cy = (y0 + y1) // 2
        cx = (x0 + x1) // 2
        iy = int(np.argmin(np.abs(ys - cy)))
        ix = int(np.argmin(np.abs(xs - cx)))
        q_vectors = flat_features[iy * len(xs) + ix : iy * len(xs) + ix + 1]
    else:
        q_vectors = flat_features[q_ink]

    trimmed = _trim_query_vectors(q_vectors, config.query_trim_percentile)
    normalized_q = _normalize_vectors(trimmed, mean, std)
    median_q = np.median(normalized_q, axis=0, keepdims=True)
    return median_q / (np.linalg.norm(median_q, axis=1, keepdims=True) + 1e-6)


def _split_prototypes(
    prototypes: np.ndarray,
    query_proto: np.ndarray,
    config: MatcherConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Separate prototypes into query-matching prototypes and competitors."""
    sims = np.dot(prototypes, query_proto.T).squeeze(-1)
    best_sim = float(np.max(sims))

    threshold = max(config.min_similarity_same, best_sim - config.competitor_margin_threshold)
    is_competitor = sims < threshold

    if not np.any(is_competitor):
        is_competitor = sims < best_sim
    if not np.any(is_competitor):
        is_competitor = np.ones(len(sims), dtype=bool)
        is_competitor[int(np.argmax(sims))] = False

    comp_protos = prototypes[is_competitor]
    same_protos = prototypes[~is_competitor]
    return same_protos, comp_protos


def _map_margin_to_score(
    margin: np.ndarray,
    config: MatcherConfig,
) -> np.ndarray:
    """Map real-valued competition margin to [0, 1] probability range."""
    if config.score_mapping_mode == "sigmoid":
        z = config.sigmoid_gain * (margin - config.sigmoid_bias)
        z = np.clip(z, -20.0, 20.0)
        return 1.0 / (1.0 + np.exp(-z))
    scaled = (margin + config.score_mapping_offset) * config.score_mapping_scale
    return np.clip(scaled, 0.0, 1.0)


def compute_competition_scores(
    normalized_all: np.ndarray,
    query_proto: np.ndarray,
    same_protos: np.ndarray,
    comp_protos: np.ndarray,
    config: MatcherConfig,
) -> np.ndarray:
    """Score cells by similarity to query minus best similarity to competitors."""
    sim_q = np.dot(normalized_all, query_proto.T).squeeze(-1)
    if len(same_protos) > 1:
        sim_same = np.max(np.dot(normalized_all, same_protos.T), axis=1)
        sim_q = np.maximum(sim_q, sim_same)

    if len(comp_protos) > 0:
        sim_comp_matrix = np.dot(normalized_all, comp_protos.T)
        if config.use_softmax_margin:
            temp = config.softmax_temperature
            max_comp = np.max(sim_comp_matrix, axis=1)
            exp_sum = np.sum(np.exp(temp * (sim_comp_matrix - max_comp[:, None])), axis=1)
            lse_comp = max_comp + np.log(exp_sum + 1e-6) / temp
            margin = sim_q - lse_comp
        else:
            sim_comp = np.max(sim_comp_matrix, axis=1)
            margin = sim_q - sim_comp
    else:
        margin = sim_q

    return _map_margin_to_score(margin, config)


def smooth_score_grid(
    score_grid: np.ndarray,
    config: MatcherConfig,
) -> np.ndarray:
    """Apply spatial smoothing over the 2D score grid."""
    rad = config.spatial_blur_radius
    if rad <= 1:
        return score_grid
    if rad % 2 == 0:
        rad += 1
    if config.use_bilateral_smoothing:
        smoothed = cv2.bilateralFilter(score_grid.astype(np.float32), rad, 0.2, rad)
    else:
        smoothed = cv2.GaussianBlur(score_grid, (rad, rad), config.spatial_blur_sigma)
    return np.clip(smoothed, 0.0, 1.0)


def sample_score_coordinates(
    score_grid: np.ndarray,
    ys_eval: np.ndarray,
    xs_eval: np.ndarray,
    stride: int,
) -> np.ndarray:
    """Sample score grid directly at evaluated pixel coordinates."""
    gh, gw = score_grid.shape
    iy = np.minimum(ys_eval // stride, gh - 1)
    ix = np.minimum(xs_eval // stride, gw - 1)
    return score_grid[iy, ix]


class CompetitionMatcher:
    """Drawing-level pattern competition matcher."""

    def __init__(self, config: MatcherConfig | None = None) -> None:
        self.config = config or MatcherConfig()

    def process_drawing(
        self, rgb: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract features and discover prototypes for a sheet."""
        feats, ink_grid, ys, xs = extract_texture_grid(rgb, stride=self.config.stride)
        gh, gw, channels = feats.shape
        flat_feats = feats.reshape(-1, channels)
        flat_ink = ink_grid.reshape(-1)

        protos, mean, std, non_blank = discover_prototypes(flat_feats, flat_ink, self.config)
        norm_all = _normalize_vectors(flat_feats, mean, std)
        return feats, ink_grid, ys, xs, protos, mean, std, norm_all

    def score_query(
        self,
        request: dict,
        drawing_context: tuple,
    ) -> np.ndarray:
        """Score full grid for a specific query within the preprocessed drawing."""
        feats, ink_grid, ys, xs, protos, mean, std, norm_all = drawing_context
        gh, gw, channels = feats.shape
        flat_feats = feats.reshape(-1, channels)

        query_box = tuple(request["query_box"])
        query_proto = extract_query_prototype(
            flat_feats,
            ink_grid.reshape(-1),
            ys,
            xs,
            query_box,
            mean,
            std,
            self.config,
        )

        same_protos, comp_protos = _split_prototypes(protos, query_proto, self.config)
        scores_flat = compute_competition_scores(norm_all, query_proto, same_protos, comp_protos, self.config)

        score_grid = scores_flat.reshape(gh, gw)
        return smooth_score_grid(score_grid, self.config)
