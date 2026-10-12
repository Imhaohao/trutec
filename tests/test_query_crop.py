"""Query descriptor source: exact crop median versus the legacy drawing descriptor."""
from __future__ import annotations

import argparse
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hatch_texture
from evaluate import EvaluationError, reviewed_domains
from evaluate_texture_calibration import run_calibration
from evaluate_texture_sweep import process_sweep_query, run_sweep
from inference_texture import run_texture_inference
from train_texture import build_metadata, collect_training_dataset, parse_args

CONTRAST_BOX = (40, 40, 56, 52)
ROUTE_BOX = (8, 8, 20, 18)


class _FlatModel:
    def predict_proba(self, pairs: np.ndarray) -> np.ndarray:
        positive = np.full(pairs.shape[0], 0.2, dtype=np.float64)
        return np.column_stack((1.0 - positive, positive))


def _contrast(outside: int, inside: int) -> np.ndarray:
    image = np.full((96, 96), outside, np.uint8)
    image[40:52, 40:56] = inside
    return image


def _crop_descriptor(image: np.ndarray) -> np.ndarray:
    features = hatch_texture.texture_features(image, 4)
    return hatch_texture.descriptor_for_query(image, features, CONTRAST_BOX, 4, "crop")


def _median_of_crop(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    crop = np.ascontiguousarray(image[y0:y1, x0:x1])
    cells = hatch_texture.texture_features(crop, 4)
    median = np.median(cells.reshape(-1, cells.shape[-1]), axis=0)
    return np.asarray(median, dtype=np.float32)


def _prepare(root: Path) -> tuple[np.ndarray, dict[str, object], dict[str, object]]:
    gray = np.full((32, 32), 255, np.uint8)
    gray[8:18, 8:20] = 0
    cv2.imwrite(str(root / "sheet.png"), gray)
    loaded = cv2.imread(str(root / "sheet.png"), cv2.IMREAD_GRAYSCALE)
    if loaded is None:
        raise AssertionError("failed to reread drawing")
    labeled: dict[str, object] = {
        "id": "qcrop",
        "document_id": "doccrop",
        "image": "sheet.png",
        "kind": "hatch",
        "width": 32,
        "height": 32,
        "query_box": [8, 8, 20, 18],
        "labels": {"positive_boxes": [[22, 22, 30, 30]]},
    }
    request: dict[str, object] = {
        "id": "qcrop",
        "image": "sheet.png",
        "width": 32,
        "height": 32,
        "query_box": [8, 8, 20, 18],
        "context_boxes": [],
    }
    _write_probability(root, labeled)
    (root / "pred").mkdir()
    (root / "sweep").mkdir()
    return loaded, labeled, request


def _write_probability(root: Path, row: dict[str, object]) -> None:
    positive, known, blank, _meta = reviewed_domains(row, root)
    folder = root / "probs"
    folder.mkdir(parents=True, exist_ok=True)
    probability = np.full(int(known.sum()), np.float32(0.25), dtype=np.float32)
    np.savez_compressed(
        folder / "qcrop.npz",
        probability=probability,
        positive=np.asarray(positive[known]),
        blank=np.asarray(blank[known]),
    )


def _exercise_routes(
    root: Path,
    gray: np.ndarray,
    labeled: dict[str, object],
    request: dict[str, object],
    metadata: dict[str, object],
    source: str,
) -> tuple[list[str], list[int], list[np.ndarray]]:
    seen: list[str] = []
    corners: list[int] = []
    values: list[np.ndarray] = []
    real = hatch_texture.descriptor_for_query

    def spy(
        image: np.ndarray,
        features: np.ndarray,
        query_box: tuple[int, int, int, int],
        stride: int = 4,
        source: str = "drawing",
    ) -> np.ndarray:
        seen.append(source)
        corners.append(int(image[0, 0]))
        value = real(image, features, query_box, stride, source)
        values.append(value.copy())
        return value

    model = _FlatModel()
    with patch("hatch_texture.descriptor_for_query", spy):
        collect_training_dataset([labeled], root, 4, 4, 0, query_source=source)
        run_texture_inference([request], root, root / "pred", model, metadata, 0.5)
        run_sweep([labeled], root, model, metadata, root / "sweep", thresholds=(0.5,))
        run_calibration([labeled], root, model, metadata, root / "probs")
    expected = real(gray, hatch_texture.texture_features(gray, 4), ROUTE_BOX, 4, source)
    for value in values:
        np.testing.assert_array_equal(value, expected)
    return seen, corners, values


class QueryCropTests(unittest.TestCase):
    def test_crop_depends_only_on_query_pixels(self) -> None:
        base = _crop_descriptor(_contrast(255, 0))
        outside = _crop_descriptor(_contrast(0, 0))
        inside = _crop_descriptor(_contrast(255, 255))
        np.testing.assert_array_equal(base, outside)
        self.assertFalse(np.array_equal(base, inside))

    def test_crop_is_median_of_exact_feature_cells(self) -> None:
        image = _contrast(180, 40)
        ignored = np.zeros((1, 1, 1), np.float32)
        for box in ((4, 6, 9, 9), (15, 15, 16, 16)):
            got = hatch_texture.descriptor_for_query(image, ignored, box, 4, "crop")
            np.testing.assert_array_equal(got, _median_of_crop(image, box))

    def test_drawing_matches_query_descriptor(self) -> None:
        image = _contrast(255, 0)
        features = hatch_texture.texture_features(image, 4)
        drawing = hatch_texture.descriptor_for_query(image, features, CONTRAST_BOX, 4, "drawing")
        legacy = hatch_texture.query_descriptor(features, CONTRAST_BOX, (96, 96), 4)
        default = hatch_texture.descriptor_for_query(image, features, CONTRAST_BOX, 4)
        crop = hatch_texture.descriptor_for_query(image, features, CONTRAST_BOX, 4, "crop")
        np.testing.assert_array_equal(drawing, legacy)
        np.testing.assert_array_equal(default, legacy)
        self.assertFalse(np.array_equal(drawing, crop))

    def test_invalid_source_and_crop_bounds(self) -> None:
        image = np.zeros((8, 8), np.uint8)
        features = hatch_texture.texture_features(image, 4)
        with self.assertRaises(ValueError):
            hatch_texture.descriptor_for_query(image, features, (1, 1, 4, 4), 4, "window")
        with self.assertRaises(ValueError):
            hatch_texture.descriptor_for_query(image, features, (0, 0, 20, 4), 4, "crop")
        with self.assertRaises(ValueError):
            hatch_texture.descriptor_for_query(image.astype(np.float32), features, (1, 1, 4, 4), 4, "crop")
        with self.assertRaises(ValueError):
            hatch_texture.descriptor_source_from_metadata({"query_descriptor_source": "full"})
        self.assertEqual(hatch_texture.descriptor_source_from_metadata({"stride": 4}), "drawing")

    def test_metadata_records_the_cli_choice(self) -> None:
        names = hatch_texture.texture_feature_names()
        rows = [{"id": "qcrop", "document_id": "doccrop"}]
        labels = np.array([1, 0], dtype=np.int32)
        crop_args = argparse.Namespace(query_source="crop", max_iter=1, seed=0, samples_per_class=1)
        crop_meta = build_metadata(crop_args, "ab", names, rows, labels, 0.1)
        self.assertEqual(crop_meta["query_descriptor_source"], "crop")
        drawing_args = argparse.Namespace(query_source="drawing", max_iter=1, seed=0, samples_per_class=1)
        drawing_meta = build_metadata(drawing_args, "ab", names, rows, labels, 0.1)
        self.assertEqual(drawing_meta["query_descriptor_source"], "drawing")
        with patch("sys.argv", ["train_texture.py", "--manifest", "m.json", "--data-root", "data", "--output", "out.joblib"]):
            self.assertEqual(parse_args().query_source, "drawing")
        with patch("sys.argv", ["train_texture.py", "--manifest", "m.json", "--data-root", "data", "--output", "out.joblib", "--query-source", "crop"]):
            self.assertEqual(parse_args().query_source, "crop")
        with patch("sys.argv", ["train_texture.py", "--manifest", "m.json", "--data-root", "data", "--output", "out.joblib", "--query-source", "window"]):
            with self.assertRaises(SystemExit):
                parse_args()

    def test_legacy_metadata_routes_use_drawing(self) -> None:
        self._assert_routes({"stride": 4}, "drawing")

    def test_crop_metadata_routes_use_crop(self) -> None:
        args = argparse.Namespace(query_source="crop", max_iter=1, seed=0, samples_per_class=1)
        metadata = build_metadata(
            args,
            "ab",
            hatch_texture.texture_feature_names(),
            [{"id": "qcrop", "document_id": "doccrop"}],
            np.array([1, 0], dtype=np.int32),
            0.1,
        )
        self._assert_routes(metadata, "crop")

    def test_sweep_without_gray_stays_on_drawing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gray, labeled, _request = _prepare(root)
            features = hatch_texture.texture_features(gray, 4)
            expected = hatch_texture.descriptor_for_query(gray, features, ROUTE_BOX, 4, "drawing")
            seen: list[np.ndarray] = []
            real = hatch_texture.query_descriptor

            def spy(
                grid: np.ndarray,
                query_box: tuple[int, int, int, int],
                image_shape: tuple[int, int],
                stride: int = 4,
            ) -> np.ndarray:
                value = real(grid, query_box, image_shape, stride)
                seen.append(value.copy())
                return value

            model = _FlatModel()
            with self.assertRaises(EvaluationError):
                process_sweep_query(
                    labeled, features, gray.shape, model, root, 4, (0.5,), source="crop"
                )
            with patch("hatch_texture.query_descriptor", spy):
                process_sweep_query(labeled, features, gray.shape, model, root, 4, (0.5,))
            self.assertEqual(len(seen), 1)
            np.testing.assert_array_equal(seen[0], expected)
            with self.assertRaises(EvaluationError):
                process_sweep_query(
                    labeled, features, gray.shape, model, root, 4, (0.5,),
                    gray=np.zeros((4, 4), np.uint8), source="drawing",
                )

    def _assert_routes(self, metadata: dict[str, object], source: str) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gray, labeled, request = _prepare(root)
            seen, corners, values = _exercise_routes(root, gray, labeled, request, metadata, source)
        self.assertEqual(seen, [source, source, source, source])
        self.assertEqual(corners, [255, 255, 255, 255])
        self.assertEqual(len(values), 4)


if __name__ == "__main__":
    unittest.main()
