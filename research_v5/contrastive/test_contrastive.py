"""CPU tests on synthetic images. No dataset and no labels file is opened."""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import unittest
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from constants import EMBED_DIM, HALO, NEG, POS, STRIDE, TILE, UNK
from grids import build_label_grid, labels_for_loss
from infer import embed_tiled, embed_whole, predict, prototype_from_embedding, scores_at_pixels
from loss import contrastive_loss
from model import HatchEncoder, parameter_count


class ContrastiveTests(unittest.TestCase):
    def test_parameter_count_is_in_the_requested_band(self) -> None:
        count = parameter_count(HatchEncoder())
        self.assertGreaterEqual(count, 1_000_000)
        self.assertLessEqual(count, 3_000_000)
        self.assertEqual(HALO % STRIDE, 0)
        self.assertEqual(TILE % STRIDE, 0)
        self.assertGreater(TILE, 2 * HALO)

    def test_output_stride_and_unit_norm(self) -> None:
        model = HatchEncoder().eval()
        image = torch.rand(1, 4, 64, 80)
        with torch.no_grad():
            out = model(image)
        self.assertEqual(tuple(out.shape), (1, EMBED_DIM, 8, 10))
        norms = out.norm(dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))

    def test_unknown_pixels_do_not_contribute_to_the_loss(self) -> None:
        grid = _synthetic_grid()
        labels = torch.from_numpy(labels_for_loss(grid).reshape(-1))
        self.assertEqual(int(grid[0, 0]), UNK)
        self.assertEqual(int(grid[1, 1]), POS)
        self.assertEqual(int(grid[2, 0]), NEG)
        self.assertTrue(np.all(labels.numpy().reshape(grid.shape)[grid == UNK] == -1))
        anchor = torch.randn(8)
        cells = torch.randn(labels.numel(), 8, requires_grad=True)
        loss, _stats = contrastive_loss(anchor, cells, labels, 0.1)
        loss.backward()
        unknown = labels < 0
        self.assertEqual(int(torch.count_nonzero(cells.grad[unknown])), 0)
        self.assertGreater(int(torch.count_nonzero(cells.grad[~unknown])), 0)
        changed = cells.detach().clone()
        changed[unknown] += 4
        base, _ = contrastive_loss(anchor, cells.detach(), labels, 0.1)
        other, _ = contrastive_loss(anchor, changed, labels, 0.1)
        self.assertTrue(torch.allclose(base, other))

    def test_inference_does_not_read_labels(self) -> None:
        source = Path(__file__).with_name("infer.py").read_text(encoding="utf-8")
        for banned in ("reviewed_domains", "positive_mask", "known_mask", "fit.json", "development-real"):
            self.assertNotIn(banned, source)
        model = HatchEncoder().eval()
        rgb = np.full((64, 64, 3), 255, np.uint8)
        rgb[16:48, 16:48] = 0
        y = np.array([8, 20, 40], dtype=np.int32)
        x = np.array([8, 20, 40], dtype=np.int32)
        request = _Trap(_request())
        scores = predict(model, request, rgb, y, x, torch.device("cpu"), tile=TILE, halo=HALO)
        self.assertEqual(scores.shape, (3,))
        self.assertGreaterEqual(float(scores.min()), 0.0)
        self.assertLessEqual(float(scores.max()), 1.0)

    def test_tiled_embedding_matches_the_whole_image(self) -> None:
        torch.manual_seed(0)
        model = HatchEncoder().eval()
        device = torch.device("cpu")
        _assert_tiled_matches(model, torch.rand(1, 4, 64, 70), device)
        _assert_tiled_matches(model, torch.rand(1, 4, 500, 470), device)

    def test_coordinate_lookup_matches_the_cell_of_that_pixel(self) -> None:
        embedding = F.normalize(torch.randn(8, 6, 7), dim=0)
        prototype = F.normalize(torch.randn(8), dim=0)
        y = np.array([0, 5, 10, 20])
        x = np.array([0, 3, 8, 15])
        full = scores_at_pixels(embedding, prototype, y, x)
        flipped = scores_at_pixels(embedding, prototype, y[::-1], x[::-1])
        np.testing.assert_allclose(full, flipped[::-1])
        np.testing.assert_allclose(scores_at_pixels(embedding, prototype, y[:2], x[:2]), full[:2])

    def test_prototype_is_the_mean_of_cells_inside_the_query_box(self) -> None:
        torch.manual_seed(1)
        model = HatchEncoder().eval()
        image = torch.rand(1, 4, 128, 128)
        with torch.no_grad():
            embedding = model(image)[0]
        box = (16, 24, 80, 88)
        proto = prototype_from_embedding(embedding, box)
        manual = _manual_prototype(embedding, box)
        self.assertTrue(torch.allclose(proto, manual, atol=1e-5))


def _synthetic_grid() -> np.ndarray:
    positive = np.zeros((32, 32), dtype=bool)
    known = np.zeros((32, 32), dtype=bool)
    positive[8:16, 8:24] = True
    known[8:16, 8:24] = True
    known[20:28, 4:16] = True
    return build_label_grid(positive, known, STRIDE)


def _assert_tiled_matches(model, image, device) -> None:
    whole = embed_whole(model, image, device)
    tiled = embed_tiled(model, image, 448, HALO, device)
    delta = float((whole - tiled).abs().max())
    if delta > 1e-4:
        raise AssertionError(f"max abs diff {delta}")


def _manual_prototype(embedding: torch.Tensor, box) -> torch.Tensor:
    x0, y0, x1, y1 = box
    ys = torch.arange(embedding.shape[1]) * STRIDE + STRIDE // 2
    xs = torch.arange(embedding.shape[2]) * STRIDE + STRIDE // 2
    row = (ys >= y0) & (ys < y1)
    col = (xs >= x0) & (xs < x1)
    mask = row[:, None] & col[None, :]
    return F.normalize(embedding[:, mask].mean(dim=1), dim=0)


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


class _Trap(dict):
    allowed = {"query_box", "width", "height", "id", "image", "context_boxes"}

    def __getitem__(self, key):
        if key not in self.allowed:
            raise AssertionError(f"inference read {key}")
        return dict.__getitem__(self, key)


if __name__ == "__main__":
    unittest.main()
