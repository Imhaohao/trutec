"""CPU tests. No dataset file is opened."""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import unittest
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .constants import deepest_native_receptive_field
from .data import one_pass
from .geom import apply_dihedral, cell_fractions, tile_repeat
from .infer import forward_logits, lookup_sigmoid, predict, stitch_logits
from .loss import masked_loss
from .model import HatchContextNet, parameter_count, stride_count


class LargeContextTests(unittest.TestCase):
    def test_parameter_budget_and_receptive_field(self) -> None:
        model = HatchContextNet()
        count = parameter_count(model)
        self.assertGreaterEqual(count, 4_000_000)
        self.assertLessEqual(count, 10_000_000)
        self.assertGreaterEqual(deepest_native_receptive_field(), 1000)
        drawing = stride_count(model.encoder, 2) + stride_count(model.down16, 2) + stride_count(model.down32, 2)
        self.assertEqual(drawing, 5)
        self.assertEqual(stride_count(model.query_encoder, 2), 3)

    def test_decode_matches_forward(self) -> None:
        torch.manual_seed(0)
        model = HatchContextNet().eval()
        drawing = torch.rand(1, 4, 64, 64)
        query = torch.rand(1, 4, 96, 96)
        with torch.no_grad():
            whole = model(drawing, query)
            vec, feat = model.query_encoder(query)
            other = model.decode(model.encoder(drawing), vec, feat)
        self.assertTrue(torch.allclose(whole, other, atol=1e-5))
        self.assertEqual(tuple(whole.shape), (1, 1, 32, 32))

    def test_unknown_cells_have_zero_loss_gradient(self) -> None:
        logits = torch.zeros(1, 1, 2, 2, requires_grad=True)
        known = torch.zeros(1, 2, 2)
        positive = torch.zeros(1, 2, 2)
        known[0, 0, 0] = 12 / 16
        known[0, 0, 1] = 11 / 16
        positive[0, 0, 0] = 0.5
        positive[0, 0, 1] = 1.0
        loss, stats = masked_loss(logits, known, positive)
        loss.backward()
        self.assertEqual(stats["known_cells"], 1)
        self.assertEqual(float(logits.grad[0, 0, 0, 1]), 0.0)
        self.assertNotEqual(float(logits.grad[0, 0, 0, 0]), 0.0)
        changed = logits.detach().clone()
        changed[0, 0, 0, 1] = 4
        other, _ = masked_loss(changed, known, positive)
        self.assertTrue(torch.allclose(loss.detach(), other))

    def test_known_cell_needs_three_quarters_of_its_pixels(self) -> None:
        known = np.zeros((8, 8), dtype=bool)
        positive = np.zeros((8, 8), dtype=bool)
        known[:4, :3] = True
        positive[:4, :2] = True
        known_frac, pos_frac = cell_fractions(known, positive)
        self.assertAlmostEqual(float(known_frac[0, 0]), 12 / 16)
        self.assertAlmostEqual(float(pos_frac[0, 0]), 8 / 16)
        self.assertLess(float(known_frac[1, 1]), 0.75)

    def test_rotation_augments_query_and_drawing_together(self) -> None:
        drawing = np.zeros((16, 20, 3), np.uint8)
        query = np.zeros((6, 8, 3), np.uint8)
        known = np.zeros((16, 20), dtype=bool)
        positive = np.zeros((16, 20), dtype=bool)
        drawing[1, 2] = (9, 8, 7)
        query[1, 2] = (9, 8, 7)
        known[1, 2] = True
        positive[1, 2] = True
        got = apply_dihedral([drawing, query, known, positive], turns=1, flip_h=True, flip_v=True)
        manual = [_manual_dihedral(array) for array in (drawing, query, known, positive)]
        for left, right in zip(got, manual):
            self.assertTrue(np.array_equal(left, right))
        self.assertEqual(int(got[0][..., 0].max()), 9)
        self.assertEqual(int(got[1][..., 0].max()), 9)

    def test_query_is_repeated_rather_than_stretched(self) -> None:
        image = np.arange(6, dtype=np.float32).reshape(2, 3, 1)
        tiled = tile_repeat(image, 5)
        self.assertEqual(tiled.shape, (5, 5, 1))
        self.assertEqual(float(tiled[0, 0, 0]), float(image[0, 0, 0]))
        self.assertEqual(float(tiled[2, 0, 0]), float(image[0, 0, 0]))
        self.assertEqual(float(tiled[0, 3, 0]), float(image[0, 0, 0]))

    def test_inference_does_not_read_labels(self) -> None:
        source = Path(__file__).with_name("infer.py").read_text(encoding="utf-8")
        for banned in ("reviewed_domains", "positive_mask", "known_mask", "fit.json", "development-real", "labels"):
            self.assertNotIn(banned, source)
        model = HatchContextNet().eval()
        rgb = np.full((64, 64, 3), 255, np.uint8)
        rgb[16:48, 16:48] = 0
        y = np.array([8, 20, 40], dtype=np.int32)
        x = np.array([8, 20, 40], dtype=np.int32)
        scores = predict(model, _Trap(_request()), rgb, y, x, torch.device("cpu"), tile=64, overlap=32)
        self.assertEqual(scores.shape, (3,))
        self.assertGreaterEqual(float(scores.min()), 0.0)
        self.assertLessEqual(float(scores.max()), 1.0)

    def test_tiled_output_matches_untiled_on_a_small_image(self) -> None:
        torch.manual_seed(0)
        device = torch.device("cpu")
        packed = torch.rand(4, 96, 80).numpy()
        query = torch.rand(4, 96, 96).numpy()
        model = HatchContextNet().eval()
        whole = forward_logits(model, packed, query, device)
        tiled = stitch_logits(model, packed, query, device, tile=128, overlap=32)
        _assert_center(whole, tiled, whole.shape[0] // 4)
        tiny = _TinyNet().eval()
        image = torch.rand(4, 96, 96).numpy()
        whole_tiny = forward_logits(tiny, image, query, device)
        tiled_tiny = stitch_logits(tiny, image, query, device, tile=64, overlap=32)
        # The overlap edge is one kernel away from a full view. The middle is not.
        _assert_center(whole_tiny, tiled_tiny, 18)

    def test_lookup_uses_the_same_cell_for_every_call(self) -> None:
        logits = np.arange(20, dtype=np.float32).reshape(4, 5)
        y = np.array([0, 4, 7, 100])
        x = np.array([0, 4, 9, 100])
        scores = lookup_sigmoid(logits, y, x)
        flipped = lookup_sigmoid(logits, y[::-1], x[::-1])
        np.testing.assert_allclose(scores, flipped[::-1])
        self.assertAlmostEqual(float(scores[0]), float(_expected_sigmoid(logits[0, 0])))

    def test_documents_are_visited_equally(self) -> None:
        counts = Counter(one_pass(("a", "b", "c"), 4, np.random.default_rng(1234)))
        self.assertEqual(counts, Counter({"a": 4, "b": 4, "c": 4}))


class _TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(4, 1, kernel_size=3, padding=1, bias=False)

    def forward(self, drawing, query):
        del query
        return F.avg_pool2d(self.conv(drawing), kernel_size=2)


class _Trap(dict):
    allowed = {"query_box", "width", "height", "id", "image", "context_boxes"}

    def __getitem__(self, key):
        if key not in self.allowed:
            raise AssertionError("inference read " + str(key))
        return dict.__getitem__(self, key)


def _request() -> dict:
    return {
        "id": "q-test",
        "image": "synthetic.png",
        "width": 64,
        "height": 64,
        "query_box": [16, 16, 40, 40],
        "context_boxes": [],
        "labels": {"positive_mask": "hidden.png"},
        "document_id": "doc-secret",
    }


def _manual_dihedral(array: np.ndarray) -> np.ndarray:
    spun = np.rot90(array, 1)
    spun = np.flip(spun, axis=0)
    spun = np.flip(spun, axis=1)
    return np.ascontiguousarray(spun)


def _assert_center(whole: np.ndarray, tiled: np.ndarray, margin: int) -> None:
    if whole.shape != tiled.shape:
        raise AssertionError(f"shape {whole.shape} != {tiled.shape}")
    center_w = whole[margin:-margin, margin:-margin]
    center_t = tiled[margin:-margin, margin:-margin]
    if center_w.size == 0:
        raise AssertionError("central region is empty")
    delta = float(np.max(np.abs(center_w - center_t)))
    if delta > 1e-4:
        raise AssertionError(f"central max abs diff {delta}")


def _expected_sigmoid(value: float) -> float:
    clipped = float(np.clip(value, -40.0, 40.0))
    return 1.0 / (1.0 + np.exp(-clipped))


if __name__ == "__main__":
    unittest.main()
