"""Synthetic checks for deterministic query-conditioned cadence features."""
import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hatch_cadence


def _stripes(height=128, width=128, spacing=8, angle=0.0):
    yy, xx = np.indices((height, width))
    phase = xx * math.cos(angle) + yy * math.sin(angle)
    return np.where((phase % spacing) < spacing / 2, 0, 255).astype(np.uint8)


class CadenceTests(unittest.TestCase):
    def test_constant_and_tiny_queries_fall_back(self):
        for image in (np.full((64, 64), 255, np.uint8), np.zeros((1, 1), np.uint8),
                      np.full((3, 5), 127, np.uint8)):
            lag = hatch_cadence.estimate_query_lag(image)
            self.assertFalse(lag.available)
            self.assertTrue(np.array_equal(
                hatch_cadence.cadence_features(image, lag, np.array([[0, 0]])),
                np.array([[0.5, 0.5, 0.5, 0.5, 0]], np.float32)))

    def test_known_sinusoidal_lag_and_orientation(self):
        yy, xx = np.indices((128, 128))
        image = (127.5 + 100 * np.cos(2 * np.pi * (xx * .125 + yy * .0625))).astype(np.uint8)
        lag = hatch_cadence.estimate_query_lag(image)
        self.assertTrue(lag.available)
        self.assertAlmostEqual(lag.period, 1 / math.hypot(.125, .0625), delta=1.0)
        self.assertAlmostEqual(lag.angle, math.atan2(.0625, .125), delta=math.radians(2))
        horizontal = hatch_cadence.estimate_query_lag(_stripes(angle=0.0))
        self.assertTrue(horizontal.available)
        self.assertAlmostEqual(horizontal.angle, 0.0, delta=math.radians(3))

    def test_unstable_quadrants_are_rejected(self):
        image = np.full((128, 128), 255, np.uint8)
        image[:64, :64] = _stripes(64, 64, spacing=8)
        image[:64, 64:] = _stripes(64, 64, spacing=20)
        image[64:, :64] = _stripes(64, 64, spacing=8)
        image[64:, 64:] = _stripes(64, 64, spacing=20)
        self.assertFalse(hatch_cadence.estimate_query_lag(image).available)

    def test_sparse_dense_batch_and_edge_results_agree(self):
        image = _stripes(81, 93, spacing=8, angle=.35)
        lag = hatch_cadence.estimate_query_lag(_stripes(128, 128, spacing=8, angle=.35))
        self.assertTrue(lag.available)
        gy, gx = np.indices((math.ceil(image.shape[0] / 4), math.ceil(image.shape[1] / 4)))
        dense_coords = np.column_stack((gy.ravel(), gx.ravel()))
        dense = hatch_cadence.cadence_features(image, lag, dense_coords, batch_size=7)
        sparse_indices = np.array([0, 1, 9, len(dense) - 1, 40])
        sparse = hatch_cadence.cadence_features(image, lag, dense_coords[sparse_indices], batch_size=2)
        self.assertTrue(np.allclose(sparse, dense[sparse_indices], atol=1e-6, rtol=0))
        tiled_reference = hatch_cadence.cadence_features_tiled(
            image, lag, dense_coords, core_size=32, batch_size=13
        )
        self.assertTrue(np.allclose(dense, tiled_reference, atol=1e-6, rtol=0))
        self.assertTrue(np.isfinite(dense).all())
        self.assertGreaterEqual(float(dense.min()), 0)
        self.assertLessEqual(float(dense.max()), 1)
        self.assertEqual(dense.dtype, np.float32)

    def test_reflect101_and_query_median_descriptor(self):
        ramp = np.arange(17 * 19, dtype=np.uint16).reshape(17, 19)
        image = (ramp % 256).astype(np.uint8)
        lag = hatch_cadence.LagState(8.0, 0.0, (None, None, None, None))
        coords = np.array([[0, 0], [4, 4], [2, 3]])
        result = hatch_cadence.cadence_features(image, lag, coords, batch_size=1)
        descriptor = hatch_cadence.query_cadence_descriptor(image, lag)
        self.assertEqual(result.shape, (3, 5))
        self.assertEqual(descriptor.shape, (5,))
        self.assertTrue(np.isfinite(descriptor).all())
        self.assertTrue(np.all((descriptor >= 0) & (descriptor <= 1)))

    def test_integral_path_matches_direct_for_random_boundaries_and_lag_extremes(self):
        rng = np.random.default_rng(4061)
        cases = ((1, 1, 4.0, 0.37), (7, 11, 12.5, 1.21),
                 (17, 19, 32.0, 2.71), (65, 71, 4.25, 0.0),
                 (81, 93, 19.75, math.pi - 0.01))
        for height, width, period, angle in cases:
            gray = rng.integers(0, 256, (height, width), dtype=np.uint8)
            if height == 7:
                gray.fill(128)
            lag = hatch_cadence.LagState(period, angle, (None,) * 4)
            gy, gx = np.indices((math.ceil(height / 4), math.ceil(width / 4)))
            coords = np.column_stack((gy.ravel(), gx.ravel()))
            direct = hatch_cadence.cadence_features(gray, lag, coords, batch_size=5)
            tiled = hatch_cadence.cadence_features_tiled(
                gray, lag, coords, core_size=32, batch_size=5
            )
            self.assertTrue(np.allclose(direct, tiled, atol=1e-6, rtol=0),
                            msg=f"integral mismatch for {(height, width, period, angle)}")

    def test_unavailable_lag_still_validates_coordinates_and_lag_state(self):
        image = np.full((16, 16), 255, np.uint8)
        unavailable = hatch_cadence.LagState(None, None, (None,) * 4)
        with self.assertRaises(ValueError):
            hatch_cadence.cadence_features(image, unavailable, np.array([[4, 0]]))
        with self.assertRaises(ValueError):
            hatch_cadence.cadence_features_tiled(image, unavailable, np.array([[0, 0]]), core_size=676)
        malformed = hatch_cadence.LagState(8.0, None, (None,) * 4)
        with self.assertRaises(ValueError):
            hatch_cadence.cadence_features(image, malformed, np.array([[0, 0]]))

    def test_low_contrast_constants_match_at_full_core_boundaries(self):
        coordinates = np.array([[0, 0], [1, 1], [164, 164], [165, 165],
                                [166, 166], [167, 167], [168, 168], [169, 169],
                                [100, 168], [168, 100]])
        lag = hatch_cadence.LagState(32.0, 0.731, (None,) * 4)
        for value in (127, 128, 254):
            gray = np.full((680, 680), value, dtype=np.uint8)
            direct = hatch_cadence.cadence_features(gray, lag, coordinates, batch_size=3)
            tiled = hatch_cadence.cadence_features_tiled(
                gray, lag, coordinates, core_size=672, batch_size=3
            )
            self.assertTrue(np.allclose(direct, tiled, atol=1e-6, rtol=0))

    def test_sparse_single_level_impulses_match_in_nearly_constant_large_images(self):
        coordinates = np.array([[0, 0], [167, 167], [168, 168], [175, 182],
                                [90, 90], [167, 168], [168, 167]])
        lag = hatch_cadence.LagState(32.0, 1.13, (None,) * 4)
        locations = ((666, 665), (670, 671), (672, 672), (10, 10), (700, 730))
        for y, x in locations:
            gray = np.full((705, 731), 127, dtype=np.uint8)
            gray[y, x] = 128
            direct = hatch_cadence.cadence_features(gray, lag, coordinates)
            tiled = hatch_cadence.cadence_features_tiled(gray, lag, coordinates)
            self.assertTrue(np.allclose(direct, tiled, atol=1e-6, rtol=0),
                            msg=f"single-level impulse mismatch at {(y, x)}")
        rng = np.random.default_rng(992)
        gray = rng.choice(np.array([127, 128], dtype=np.uint8), size=(705, 731))
        direct = hatch_cadence.cadence_features(gray, lag, coordinates)
        tiled = hatch_cadence.cadence_features_tiled(gray, lag, coordinates)
        self.assertTrue(np.allclose(direct, tiled, atol=1e-6, rtol=0))


if __name__ == "__main__":
    unittest.main()
