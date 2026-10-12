"""Focused tests for controlled texture feature set (fine variance8 experiment)."""
from __future__ import annotations

import math
import argparse
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate_texture_sweep import _sweep_descriptor
import hatch_texture
from train_texture import build_metadata

_LEGACY_BASE_COUNT = 46
_VARIANCE8_COUNT = 47


def _synthetic_test_image(height: int = 64, width: int = 64) -> np.ndarray:
    """Deterministic image with sharp local contrast."""
    image = np.full((height, width), 240, dtype=np.uint8)
    image[::8, :] = 20
    image[:, ::8] = 40
    return image


class FineVarianceFeatureTests(unittest.TestCase):
    """Verify base preservation, variance8 calculation, and shape stability."""

    def test_base_and_variance8_names_and_values(self) -> None:
        """Requirement 1: Base remains legacy and variance8 appends ink_variance_w8."""
        default_names = hatch_texture.texture_feature_names()
        base_names = hatch_texture.texture_feature_names("base")
        self.assertEqual(default_names, base_names)
        self.assertEqual(len(base_names), _LEGACY_BASE_COUNT)

        image = _synthetic_test_image(48, 48)
        default_features = hatch_texture.texture_features(image)
        base_features = hatch_texture.texture_features(image, feature_set="base")
        self.assertEqual(default_features.shape, base_features.shape)
        np.testing.assert_array_equal(default_features, base_features)

        variance8_names = hatch_texture.texture_feature_names("variance8")
        self.assertEqual(len(variance8_names), _VARIANCE8_COUNT)
        self.assertEqual(variance8_names[:_LEGACY_BASE_COUNT], base_names)
        self.assertEqual(variance8_names[-1], "ink_variance_w8")

        variance8_features = hatch_texture.texture_features(image, feature_set="variance8")
        self.assertEqual(variance8_features.shape, (base_features.shape[0], base_features.shape[1], _VARIANCE8_COUNT))
        np.testing.assert_array_equal(variance8_features[:, :, :_LEGACY_BASE_COUNT], base_features)

    def test_variance8_signal_and_direct_boxfilter_equivalence(self) -> None:
        """Requirement 2: Nonzero variance8 channel equals direct 4x4 cv2.boxFilter variance."""
        image = _synthetic_test_image(64, 64)
        stride = 4
        features = hatch_texture.texture_features(image, stride=stride, feature_set="variance8")
        variance8_channel = features[:, :, 46]

        self.assertGreater(float(np.max(variance8_channel)), 0.0)

        norm = hatch_texture._half_luminance(image)
        mean = cv2.boxFilter(
            norm,
            cv2.CV_32F,
            (4, 4),
            normalize=True,
            borderType=cv2.BORDER_REFLECT_101,
        )
        mean_sq = cv2.boxFilter(
            norm * norm,
            cv2.CV_32F,
            (4, 4),
            normalize=True,
            borderType=cv2.BORDER_REFLECT_101,
        )
        direct_variance = np.maximum(mean_sq - mean * mean, np.float32(0.0))
        rows = hatch_texture._cell_axis(image.shape[0], stride)
        cols = hatch_texture._cell_axis(image.shape[1], stride)
        expected = hatch_texture._sample_map(direct_variance, rows, cols)

        self.assertEqual(variance8_channel.dtype, np.float32)
        np.testing.assert_allclose(variance8_channel, expected, rtol=1e-5, atol=1e-6)

    def test_tiny_and_odd_shaped_arrays_stability(self) -> None:
        """Requirement 3: Tiny and odd-shaped uint8 arrays remain finite, stable, and correctly shaped."""
        shapes = ((1, 1), (2, 2), (3, 5), (7, 11), (15, 1), (1, 17), (31, 45))
        configs = (("base", _LEGACY_BASE_COUNT), ("variance8", _VARIANCE8_COUNT))

        for h, w in shapes:
            grid = np.arange(h * w, dtype=np.uint8).reshape(h, w)
            expected_h = math.ceil(h / 4)
            expected_w = math.ceil(w / 4)

            for fset, expected_c in configs:
                feat = hatch_texture.texture_features(grid, stride=4, feature_set=fset)
                self.assertEqual(feat.shape, (expected_h, expected_w, expected_c))
                self.assertEqual(feat.dtype, np.float32)
                self.assertTrue(bool(np.all(np.isfinite(feat))))

                again = hatch_texture.texture_features(grid, stride=4, feature_set=fset)
                np.testing.assert_array_equal(feat, again)

    def test_descriptors_dimension_and_crop_invariance(self) -> None:
        """Requirement 4: Descriptors have length 47; crop descriptor is invariant to external changes."""
        box = (16, 16, 32, 32)
        image1 = np.full((64, 64), 220, dtype=np.uint8)
        image1[16:32, 16:32] = 40
        features1 = hatch_texture.texture_features(image1, stride=4, feature_set="variance8")

        desc_drawing = hatch_texture.descriptor_for_query(
            image1, features1, box, stride=4, source="drawing", feature_set="variance8"
        )
        self.assertEqual(desc_drawing.shape, (_VARIANCE8_COUNT,))
        self.assertEqual(desc_drawing.dtype, np.float32)
        self.assertTrue(bool(np.all(np.isfinite(desc_drawing))))

        desc_crop1 = hatch_texture.descriptor_for_query(
            image1, features1, box, stride=4, source="crop", feature_set="variance8"
        )
        self.assertEqual(desc_crop1.shape, (_VARIANCE8_COUNT,))
        self.assertEqual(desc_crop1.dtype, np.float32)
        self.assertTrue(bool(np.all(np.isfinite(desc_crop1))))

        image2 = image1.copy()
        image2[0:10, 0:10] = 10
        image2[40:60, 40:60] = 100
        features2 = hatch_texture.texture_features(image2, stride=4, feature_set="variance8")
        desc_crop2 = hatch_texture.descriptor_for_query(
            image2, features2, box, stride=4, source="crop", feature_set="variance8"
        )
        np.testing.assert_array_equal(desc_crop1, desc_crop2)

        image3 = image1.copy()
        image3[16:32, 16:32] = 180
        desc_crop3 = hatch_texture.descriptor_for_query(
            image3, features1, box, stride=4, source="crop", feature_set="variance8"
        )
        self.assertFalse(np.array_equal(desc_crop1, desc_crop3))

    def test_feature_set_from_metadata_routing_and_validation(self) -> None:
        """Requirement 5: Routes missing to base, accepts base/variance8, rejects invalid values and types."""
        self.assertEqual(hatch_texture.feature_set_from_metadata({}), "base")
        self.assertEqual(hatch_texture.feature_set_from_metadata({"stride": 4}), "base")
        self.assertEqual(
            hatch_texture.feature_set_from_metadata({"texture_feature_set": "base"}),
            "base",
        )
        self.assertEqual(
            hatch_texture.feature_set_from_metadata({"texture_feature_set": "variance8"}),
            "variance8",
        )

        invalid_values = ("unknown", "var8", 123, None, "")
        for val in invalid_values:
            with self.subTest(value=val):
                with self.assertRaises(ValueError):
                    hatch_texture.feature_set_from_metadata({"texture_feature_set": val})

        invalid_types = (None, "base", [1, 2], 42)
        for val in invalid_types:
            with self.subTest(metadata=val):
                with self.assertRaises(TypeError):
                    hatch_texture.feature_set_from_metadata(val)

        with self.assertRaises(ValueError):
            hatch_texture.texture_feature_names("unknown")

        dummy = np.zeros((16, 16), dtype=np.uint8)
        with self.assertRaises(ValueError):
            hatch_texture.texture_features(dummy, feature_set="unknown")

        dummy_grid = np.zeros((4, 4, _LEGACY_BASE_COUNT), dtype=np.float32)
        with self.assertRaises(ValueError):
            hatch_texture.descriptor_for_query(
                dummy, dummy_grid, (0, 0, 8, 8), feature_set="unknown"
            )

    def test_train_and_sweep_metadata_routing(self) -> None:
        """Verify train_texture metadata creation and sweep feature_set extraction."""
        args_base = argparse.Namespace(
            features="base",
            query_source="drawing",
            max_iter=10,
            seed=42,
            samples_per_class=10,
        )
        meta_base = build_metadata(
            args_base,
            "sha_dummy",
            hatch_texture.texture_feature_names("base"),
            [{"id": "q1", "document_id": "d1"}],
            np.array([1, 0]),
            0.5,
        )
        self.assertEqual(meta_base["texture_feature_set"], "base")
        self.assertEqual(len(meta_base["base_feature_names"]), _LEGACY_BASE_COUNT)

        args_v8 = argparse.Namespace(
            features="variance8",
            query_source="crop",
            max_iter=10,
            seed=42,
            samples_per_class=10,
        )
        meta_v8 = build_metadata(
            args_v8,
            "sha_dummy",
            hatch_texture.texture_feature_names("variance8"),
            [{"id": "q1", "document_id": "d1"}],
            np.array([1, 0]),
            0.5,
        )
        self.assertEqual(meta_v8["texture_feature_set"], "variance8")
        self.assertEqual(len(meta_v8["base_feature_names"]), _VARIANCE8_COUNT)
        self.assertEqual(len(meta_v8["pair_feature_names"]), _VARIANCE8_COUNT * 4)

        self.assertEqual(hatch_texture.feature_set_from_metadata(meta_base), "base")
        self.assertEqual(hatch_texture.feature_set_from_metadata(meta_v8), "variance8")

        image = _synthetic_test_image(32, 32)
        feats_v8 = hatch_texture.texture_features(image, stride=4, feature_set="variance8")
        q_desc_draw = _sweep_descriptor(
            feats_v8, (8, 8, 20, 20), (32, 32), 4, image, "drawing", "variance8"
        )
        self.assertEqual(q_desc_draw.shape, (_VARIANCE8_COUNT,))
        q_desc_crop = _sweep_descriptor(
            feats_v8, (8, 8, 20, 20), (32, 32), 4, image, "crop", "variance8"
        )
        self.assertEqual(q_desc_crop.shape, (_VARIANCE8_COUNT,))


if __name__ == "__main__":
    unittest.main()
