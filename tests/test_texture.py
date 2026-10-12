"""Synthetic checks for query-conditioned hatch correlation and texture descriptors."""
import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hatch_texture


def _expected_names():
    names = []
    for window in (16, 32, 64):
        names.append(f"ink_density_w{window}")
        names.append(f"ink_variance_w{window}")
    names.extend((
        "grad_energy_horizontal",
        "grad_energy_vertical",
        "grad_energy_diagonal",
        "grad_energy_antidiagonal",
    ))
    for wavelength in (4, 8, 16):
        for angle in (0, 30, 60, 90, 120, 150):
            for pool in (16, 32):
                names.append(f"gabor_o{angle}_l{wavelength}_p{pool}")
    return names


def _hatch(height, width, kind, spacing=8, thickness=2, phase=0):
    image = np.full((height, width), 255, np.uint8)
    yy, xx = np.indices((height, width))
    if kind == "diag":
        ink = np.mod(xx + yy + phase, spacing) < thickness
    elif kind == "horiz":
        ink = np.mod(yy + phase, spacing) < thickness
    elif kind == "vert":
        ink = np.mod(xx + phase, spacing) < thickness
    else:
        raise AssertionError(kind)
    image[ink] = 0
    return image


def _paste(canvas, tile, y0, x0):
    canvas[y0:y0 + tile.shape[0], x0:x0 + tile.shape[1]] = tile


def _repeated_diagonal():
    canvas = np.full((200, 200), 255, np.uint8)
    _paste(canvas, _hatch(64, 64, "diag", phase=0), 8, 8)
    _paste(canvas, _hatch(64, 64, "diag", phase=2), 128, 128)
    return canvas


def _mean(values):
    return float(np.mean(values))


class TemplateResponseTests(unittest.TestCase):
    def test_odd_native_shape_stays_in_unit_interval(self):
        gray = _hatch(31, 45, "diag", spacing=6, thickness=2)
        response = hatch_texture.normalized_template_response(gray, (4, 3, 22, 19))
        self.assertEqual(response.shape, (31, 45))
        self.assertEqual(response.dtype, np.float32)
        self.assertTrue(np.isfinite(response).all())
        self.assertGreaterEqual(float(response.min()), 0.0)
        self.assertLessEqual(float(response.max()), 1.0)
        mask = hatch_texture.template_response_to_mask(response, 0.5)
        self.assertEqual(mask.dtype, np.bool_)
        self.assertEqual(mask.shape, response.shape)

    def test_constant_and_white_queries_do_not_paint(self):
        white = np.full((48, 48), 255, np.uint8)
        white_response = hatch_texture.normalized_template_response(white, (6, 6, 28, 28))
        self.assertEqual(float(white_response.max()), 0.0)
        self.assertTrue(np.isfinite(white_response).all())

        drawing = np.full((80, 80), 255, np.uint8)
        _paste(drawing, _hatch(30, 30, "diag"), 40, 40)
        paper = hatch_texture.normalized_template_response(drawing, (2, 2, 24, 24))
        self.assertEqual(float(paper.max()), 0.0)

        solid = _hatch(60, 60, "vert")
        solid[0:16, 0:16] = 0
        black = hatch_texture.normalized_template_response(solid, (0, 0, 16, 16))
        self.assertEqual(float(black.max()), 0.0)
        self.assertFalse(np.isnan(black).any())

    def test_repeated_diagonal_hatch_is_found_away_from_the_query(self):
        image = _repeated_diagonal()
        query = (16, 16, 48, 48)
        response = hatch_texture.normalized_template_response(image, query)
        away = response[140:184, 140:184]
        white = response[8:40, 150:190]
        query_only = np.zeros(response.shape, dtype=bool)
        query_only[query[1]:query[3], query[0]:query[2]] = True
        detected = response >= 0.6
        self.assertGreater(_mean(away), 0.6)
        self.assertLess(_mean(white), 0.2)
        self.assertGreater(int(detected.sum()), int(query_only.sum()))
        self.assertTrue(detected[160, 160])
        self.assertFalse(np.array_equal(detected, query_only))

    def test_small_query_still_finds_the_repeated_hatch(self):
        image = _repeated_diagonal()
        response = hatch_texture.normalized_template_response(image, (20, 20, 28, 28))
        self.assertGreater(_mean(response[144:180, 144:180]), 0.45)
        self.assertLess(_mean(response[8:40, 150:190]), 0.25)

    def test_different_orientation_scores_lower_than_the_query_pattern(self):
        image = np.full((200, 180), 255, np.uint8)
        _paste(image, _hatch(64, 180, "horiz"), 0, 0)
        _paste(image, _hatch(64, 180, "diag"), 136, 0)
        response = hatch_texture.normalized_template_response(image, (40, 148, 80, 188))
        diagonal = _mean(response[150:190, 110:170])
        horizontal = _mean(response[8:40, 110:170])
        self.assertGreater(diagonal, horizontal + 0.25)

    def test_invalid_rectangles_and_images_are_rejected(self):
        gray = np.full((16, 20), 255, np.uint8)
        bad_boxes = (
            (-1, 0, 4, 4),
            (0, 0, 0, 4),
            (8, 2, 4, 6),
            (0, 0, 21, 4),
            (0, 14, 4, 14),
            (0, 0, 4, 4, 1),
            (0, 0, 4),
        )
        for box in bad_boxes:
            with self.subTest(box=box):
                with self.assertRaises((TypeError, ValueError)):
                    hatch_texture.normalized_template_response(gray, box)
        with self.assertRaises((TypeError, ValueError)):
            hatch_texture.normalized_template_response(gray, [0, 0, 4, 4])
        with self.assertRaises((TypeError, ValueError)):
            hatch_texture.normalized_template_response(gray, (0.0, 0, 4, 4))
        with self.assertRaises((TypeError, ValueError)):
            hatch_texture.normalized_template_response(gray.astype(np.float32), (0, 0, 4, 4))
        with self.assertRaises((TypeError, ValueError)):
            hatch_texture.normalized_template_response(np.dstack([gray, gray, gray]), (0, 0, 4, 4))
        with self.assertRaises((TypeError, ValueError)):
            hatch_texture.normalized_template_response(gray, (0, 0, 4, 4), max_template_side=0)

    def test_mask_threshold_validation(self):
        response = np.array([[0.0, 0.2], [0.8, 1.0]], dtype=np.float32)
        mask = hatch_texture.template_response_to_mask(response, 0.5)
        self.assertTrue(np.array_equal(mask, np.array([[False, False], [True, True]])))
        for bad in (math.nan, math.inf, -0.1, 1.1, True):
            with self.subTest(threshold=bad):
                with self.assertRaises((TypeError, ValueError)):
                    hatch_texture.template_response_to_mask(response, bad)
        dirty = response.copy()
        dirty[0, 0] = np.nan
        with self.assertRaises(ValueError):
            hatch_texture.template_response_to_mask(dirty, 0.5)
        with self.assertRaises((TypeError, ValueError)):
            hatch_texture.template_response_to_mask(response.astype(np.uint8), 0.5)


class TextureFeatureTests(unittest.TestCase):
    def test_dimensions_order_and_odd_sizes_are_stable(self):
        names = hatch_texture.texture_feature_names()
        self.assertEqual(names, _expected_names())
        self.assertEqual(len(names), len(set(names)))
        odd = _hatch(31, 45, "vert", spacing=6)
        features = hatch_texture.texture_features(odd)
        again = hatch_texture.texture_features(odd)
        self.assertEqual(features.shape, (math.ceil(31 / 4), math.ceil(45 / 4), len(names)))
        self.assertEqual(features.dtype, np.float32)
        self.assertTrue(np.isfinite(features).all())
        np.testing.assert_array_equal(features, again)
        tiny = hatch_texture.texture_features(np.zeros((3, 5), np.uint8))
        pixel = hatch_texture.texture_features(np.array([[128]], np.uint8))
        self.assertEqual(tiny.shape, (1, 2, len(names)))
        self.assertEqual(pixel.shape, (1, 1, len(names)))
        self.assertTrue(np.isfinite(tiny).all())
        self.assertTrue(np.isfinite(pixel).all())
        wide = hatch_texture.texture_features(np.full((8, 20), 255, np.uint8))
        self.assertEqual(wide.shape[:2], (2, 5))

    def test_ink_and_orientation_follow_the_documented_channels(self):
        names = hatch_texture.texture_feature_names()
        white = hatch_texture.texture_features(np.full((64, 64), 255, np.uint8))
        black = hatch_texture.texture_features(np.zeros((64, 64), np.uint8))
        density = names.index("ink_density_w16")
        variance = names.index("ink_variance_w32")
        self.assertLess(float(white[:, :, density].max()), 0.05)
        self.assertGreater(float(black[:, :, density].min()), 0.9)
        self.assertLess(float(white[:, :, variance].max()), 0.01)

        horizontal = self._interior(hatch_texture.texture_features(_hatch(96, 96, "horiz")))
        vertical = self._interior(hatch_texture.texture_features(_hatch(96, 96, "vert")))
        gx = names.index("grad_energy_horizontal")
        gy = names.index("grad_energy_vertical")
        self.assertGreater(horizontal[gy], horizontal[gx] * 1.5)
        self.assertGreater(vertical[gx], vertical[gy] * 1.5)
        gabor_vertical = names.index("gabor_o0_l8_p16")
        gabor_horizontal = names.index("gabor_o90_l8_p16")
        self.assertGreater(vertical[gabor_vertical], vertical[gabor_horizontal])
        self.assertGreater(horizontal[gabor_horizontal], horizontal[gabor_vertical])

    def test_phase_shift_moves_the_descriptor_less_than_an_orientation_change(self):
        box = (16, 16, 80, 80)
        shape = (96, 96)
        diagonal = self._describe(_hatch(96, 96, "diag", phase=0), box, shape)
        shifted = self._describe(_hatch(96, 96, "diag", phase=3), box, shape)
        horizontal = self._describe(_hatch(96, 96, "horiz"), box, shape)
        phase_gap = float(np.linalg.norm(diagonal - shifted))
        orientation_gap = float(np.linalg.norm(diagonal - horizontal))
        self.assertLess(phase_gap, orientation_gap)
        self.assertGreater(orientation_gap, 0.0)

    def test_last_column_of_an_odd_image_lands_on_the_last_cell(self):
        image = np.full((30, 81), 255, np.uint8)
        image[:, 76:] = 0
        features = hatch_texture.texture_features(image)
        density = features[:, :, hatch_texture.texture_feature_names().index("ink_density_w16")]
        self.assertGreater(float(density[:, -1].mean()), float(density[:, 0].mean()) + 0.05)

    def _interior(self, features):
        return features[2:-2, 2:-2].mean(axis=(0, 1))

    def _describe(self, image, box, shape):
        return hatch_texture.query_descriptor(hatch_texture.texture_features(image), box, shape)


class QueryDescriptorTests(unittest.TestCase):
    def _grid(self):
        grid_h = math.ceil(16 / 4)
        grid_w = math.ceil(16 / 4)
        features = np.zeros((grid_h, grid_w, 2), dtype=np.float32)
        features[0, 0] = (1.0, 4.0)
        features[0, 1] = (1.0, 0.0)
        features[1, 0] = (1.0, 0.0)
        features[1, 1] = (100.0, 0.0)
        features[3, 3] = (7.0, 9.0)
        return features

    def test_median_ignores_a_single_outlier_inside_the_query(self):
        descriptor = hatch_texture.query_descriptor(self._grid(), (0, 0, 8, 8), (16, 16))
        self.assertEqual(descriptor.shape, (2,))
        self.assertEqual(descriptor.dtype, np.float32)
        self.assertAlmostEqual(float(descriptor[0]), 1.0)
        self.assertAlmostEqual(float(descriptor[1]), 0.0)

    def test_tiny_rectangle_uses_the_nearest_center(self):
        features = self._grid()
        single = hatch_texture.query_descriptor(features, (3, 3, 4, 4), (16, 16))
        np.testing.assert_array_equal(single, features[0, 0])
        tied = hatch_texture.query_descriptor(features, (4, 4, 5, 5), (16, 16))
        np.testing.assert_array_equal(tied, features[0, 0])
        corner = hatch_texture.query_descriptor(features, (15, 15, 16, 16), (16, 16))
        np.testing.assert_array_equal(corner, features[3, 3])

    def test_mismatched_grid_and_bad_values_are_rejected(self):
        features = self._grid()
        with self.assertRaises(ValueError):
            hatch_texture.query_descriptor(features, (0, 0, 4, 4), (16, 20))
        broken = features.copy()
        broken[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            hatch_texture.query_descriptor(broken, (0, 0, 4, 4), (16, 16))
        with self.assertRaises((TypeError, ValueError)):
            hatch_texture.query_descriptor(features, (0, 0, 4, 4), (16, 16), stride=0)


if __name__ == "__main__":
    unittest.main()
