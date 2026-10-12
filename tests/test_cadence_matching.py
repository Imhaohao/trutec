"""Synthetic checks for the cadence pair-feature integration."""
import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cadence_matching
import hatch_cadence
import hatch_texture
import texture_learning


def _shared(height, width, seed=8):
    rng = np.random.default_rng(seed)
    return rng.random((math.ceil(height / 4), math.ceil(width / 4), 46), dtype=np.float32)


def _hatch(height, width, spacing, angle):
    yy, xx = np.indices((height, width))
    phase = xx * math.cos(angle) + yy * math.sin(angle)
    return np.where(phase % spacing < spacing / 2, 0, 255).astype(np.uint8)


class CadenceMatchingTests(unittest.TestCase):
    def test_query_descriptor_uses_exact_crop_base_and_cadence_medians(self):
        gray = _hatch(96, 112, 8, 0.3)
        shared = _shared(*gray.shape)
        box = (16, 12, 88, 84)
        descriptor, lag = cadence_matching.query_descriptor(gray, shared, box)
        base = hatch_texture.descriptor_for_query(
            gray, shared, box, stride=4, source="crop", feature_set="base"
        )
        crop = gray[box[1]:box[3], box[0]:box[2]]
        cadence = hatch_cadence.query_cadence_descriptor(
            crop, hatch_cadence.estimate_query_lag(crop)
        )
        self.assertEqual(descriptor.shape, (51,))
        self.assertEqual(descriptor.dtype, np.float32)
        self.assertTrue(np.array_equal(descriptor[:46], base))
        self.assertTrue(np.array_equal(descriptor[46:], cadence))
        self.assertEqual(lag.available, hatch_cadence.estimate_query_lag(crop).available)

    def test_base_184_pair_channels_remain_at_the_original_group_offsets(self):
        gray = _hatch(80, 84, 8, 0.5)
        shared = _shared(*gray.shape)
        before = shared.copy()
        coords = np.array([[2, 3], [8, 9], [18, 19]])
        query46 = np.linspace(0.1, 0.9, 46, dtype=np.float32)
        query51 = np.concatenate((query46, np.array([0.6, 0.4, 0.7, 0.2, 1], np.float32)))
        lag = hatch_cadence.LagState(8, 0.5, (None,) * 4)
        local46 = shared[coords[:, 0], coords[:, 1]]
        expected = texture_learning.pair_features(local46, query46)
        augmented = cadence_matching.pair_features_for_cells(gray, shared, coords, query51, lag)
        self.assertEqual(expected.shape, (3, 184))
        self.assertEqual(augmented.shape, (3, 204))
        for base_group, augmented_group in ((slice(0, 46), slice(0, 46)),
                                            (slice(46, 92), slice(51, 97)),
                                            (slice(92, 138), slice(102, 148)),
                                            (slice(138, 184), slice(153, 199))):
            self.assertTrue(np.allclose(expected[:, base_group], augmented[:, augmented_group]))
        self.assertTrue(np.array_equal(shared, before))

    def test_different_query_cadences_reuse_shared_grid_without_mutation(self):
        gray_a = _hatch(96, 96, 8, 0.0)
        gray_b = _hatch(96, 96, 12, 0.7)
        shared = _shared(96, 96)
        before = shared.copy()
        coords = np.array([[4, 4], [12, 12], [20, 20]])
        q_a = np.linspace(0.2, 0.8, 51, dtype=np.float32)
        q_b = np.linspace(0.8, 0.2, 51, dtype=np.float32)
        lag_a = hatch_cadence.LagState(8, 0.0, (None,) * 4)
        lag_b = hatch_cadence.LagState(12, 0.7, (None,) * 4)
        pairs_a = cadence_matching.pair_features_for_cells(gray_a, shared, coords, q_a, lag_a)
        pairs_b = cadence_matching.pair_features_for_cells(gray_b, shared, coords, q_b, lag_b)
        self.assertTrue(np.array_equal(pairs_a[:, :46], pairs_b[:, :46]))
        self.assertFalse(np.array_equal(pairs_a[:, 46:51], pairs_b[:, 46:51]))
        self.assertTrue(np.array_equal(shared, before))

    def test_unavailable_lag_fallback_has_expected_pair_shape_and_values(self):
        gray = np.full((48, 52), 255, np.uint8)
        shared = _shared(*gray.shape)
        box = (4, 4, 44, 44)
        query, lag = cadence_matching.query_descriptor(gray, shared, box)
        self.assertFalse(lag.available)
        self.assertTrue(np.array_equal(query[-5:], np.array([0.5, 0.5, 0.5, 0.5, 0], np.float32)))
        coords = np.array([[0, 0], [5, 6]])
        pairs = cadence_matching.pair_features_for_cells(gray, shared, coords, query, lag)
        self.assertEqual(pairs.shape, (2, 204))
        self.assertTrue(np.isfinite(pairs).all())

    def test_sparse_pairs_equal_selected_dense_pairs(self):
        gray = _hatch(72, 80, 8, 0.4)
        shared = _shared(*gray.shape)
        query = np.linspace(0.1, 0.9, 51, dtype=np.float32)
        lag = hatch_cadence.LagState(8.0, 0.4, (None,) * 4)
        gy, gx = np.indices(shared.shape[:2])
        dense_coords = np.column_stack((gy.ravel(), gx.ravel()))
        dense = cadence_matching.pair_features_for_cells(gray, shared, dense_coords, query, lag)
        selected = np.array([0, 1, 57, len(dense) - 1, 130])
        sparse = cadence_matching.pair_features_for_cells(
            gray, shared, dense_coords[selected], query, lag
        )
        self.assertTrue(np.allclose(sparse, dense[selected], atol=1e-6, rtol=0))

    def test_invalid_shapes_coordinates_queries_and_boxes_are_rejected(self):
        gray = np.full((32, 36), 255, np.uint8)
        shared = _shared(*gray.shape)
        lag = hatch_cadence.LagState(None, None, (None,) * 4)
        query = np.full(51, 0.5, dtype=np.float32)
        with self.assertRaises(ValueError):
            cadence_matching.pair_features_for_cells(gray, shared[..., :45], np.array([[0, 0]]), query, lag)
        with self.assertRaises(ValueError):
            cadence_matching.pair_features_for_cells(gray, shared, np.array([[8, 0]]), query, lag)
        with self.assertRaises(ValueError):
            cadence_matching.pair_features_for_cells(gray, shared, np.array([[0, 0]]), query[:50], lag)
        with self.assertRaises(ValueError):
            cadence_matching.query_descriptor(gray, shared, (-1, 0, 12, 12))
        with self.assertRaises(ValueError):
            cadence_matching.query_descriptor(gray, shared, (10, 10, 10, 20))


if __name__ == "__main__":
    unittest.main()
