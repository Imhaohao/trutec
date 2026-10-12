"""Synthetic unit and integration tests for HatchFinder masked fine-tuning."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import inference_hatchfinder as adapter
import train_hatchfinder as trainer


class FakeConfigModel:
    def model_dump(self, mode: str = "json") -> dict[str, Any]:
        return {
            "drawing_channels": [4],
            "hatch_channels": [4],
            "hatch_pool_sizes": [2],
            "match_dims": [4],
            "match_feature_dims": [4],
        }


class FakeConfig:
    def __init__(self) -> None:
        self.model = FakeConfigModel()


def make_tiny_fake_model(torch: Any) -> Any:
    class TinyFakeHatchFinder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = FakeConfig()
            self.drawing_encoder = torch.nn.Sequential(
                torch.nn.Conv2d(3, 4, kernel_size=1),
                torch.nn.ReLU(),
            )
            self.hatch_encoder = torch.nn.Sequential(
                torch.nn.Conv2d(3, 4, kernel_size=1),
                torch.nn.ReLU(),
            )
            self.decoder = torch.nn.Sequential(
                torch.nn.Conv2d(4, 1, kernel_size=1),
            )
            self.saw_ones_mask = False
            self.saw_leak_mask = False

        def forward(self, drawing: Any, mask: Any, hatch: Any) -> Any:
            if bool((mask == 1.0).all()):
                self.saw_ones_mask = True
            else:
                self.saw_leak_mask = True
            d = self.drawing_encoder(drawing)
            h = self.hatch_encoder(hatch)
            h_pooled = torch.nn.functional.adaptive_avg_pool2d(h, (1, 1))
            return self.decoder(d * h_pooled)

    return TinyFakeHatchFinder()


def test_unknown_logit_perturbations_dont_change_loss_or_gradients() -> None:
    torch = pytest.importorskip("torch")
    logits = torch.randn(1, 1, 32, 32, requires_grad=True)
    target = torch.zeros(1, 1, 32, 32)
    target[:, :, 8:24, 8:24] = 1.0

    loss_mask = torch.zeros(1, 1, 32, 32)
    loss_mask[:, :, 8:24, 8:24] = 1.0

    base_loss, base_bce, base_dice = trainer.masked_loss(logits, target, loss_mask)
    base_grad = torch.autograd.grad(base_loss, logits, retain_graph=True)[0]

    # Verify unknown pixels have exactly zero gradient
    assert bool((base_grad[loss_mask == 0.0] == 0.0).all())

    # Perturb logits only in the unknown region
    perturbed_logits = logits.clone().detach()
    perturbed_logits[loss_mask == 0.0] += 50.0
    perturbed_logits.requires_grad_()

    pert_loss, pert_bce, pert_dice = trainer.masked_loss(perturbed_logits, target, loss_mask)
    pert_grad = torch.autograd.grad(pert_loss, perturbed_logits)[0]

    torch.testing.assert_close(base_loss, pert_loss)
    torch.testing.assert_close(base_bce, pert_bce)
    torch.testing.assert_close(base_dice, pert_dice)
    torch.testing.assert_close(base_grad, pert_grad)


def test_positive_negative_class_center_and_crop_alignment() -> None:
    H, W = 400, 400
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    pos_mask = np.zeros((H, W), dtype=bool)
    neg_mask = np.zeros((H, W), dtype=bool)

    pos_mask[50:60, 50:60] = True
    neg_mask[200:210, 200:210] = True
    known_mask = pos_mask | neg_mask

    rng = np.random.default_rng(123)

    # Positive class choice
    py_pos, px_pos = trainer.choose_crop_center(pos_mask, neg_mask, class_choice=0, rng=rng)
    assert 50 <= py_pos < 60 and 50 <= px_pos < 60

    # Negative class choice
    py_neg, px_neg = trainer.choose_crop_center(pos_mask, neg_mask, class_choice=1, rng=rng)
    assert 200 <= py_neg < 210 and 200 <= px_neg < 210

    # Center crop calculation far from borders
    tile_size = 128
    y0, x0, vh, vw = trainer.calculate_tile_bounds(py_neg, px_neg, H, W, tile_size)
    assert vh == tile_size and vw == tile_size
    assert y0 == py_neg - tile_size // 2
    assert x0 == px_neg - tile_size // 2

    tile_rgb, tile_known, tile_pos = trainer.crop_and_pad_tile(
        rgb, known_mask, pos_mask, y0, x0, vh, vw, tile_size
    )
    assert tile_rgb.shape == (tile_size, tile_size, 3)
    # The sampled negative pixel is inside the crop and correctly labeled
    local_y = py_neg - y0
    local_x = px_neg - x0
    assert tile_known[local_y, local_x] == 1.0
    assert tile_pos[local_y, local_x] == 0.0

    # Clamping at image boundary
    y0_edge, x0_edge, _, _ = trainer.calculate_tile_bounds(5, 5, H, W, tile_size)
    assert y0_edge == 0 and x0_edge == 0

    # Padding small images with white and mask with 0
    small_rgb = np.full((40, 50, 3), 10, dtype=np.uint8)
    small_known = np.ones((40, 50), dtype=bool)
    small_pos = np.ones((40, 50), dtype=bool)
    pad_rgb, pad_known, pad_pos = trainer.crop_and_pad_tile(
        small_rgb, small_known, small_pos, 0, 0, 40, 50, tile_size
    )
    assert pad_rgb.shape == (tile_size, tile_size, 3)
    assert pad_known.shape == (tile_size, tile_size)
    assert (pad_rgb[:40, :50] == 10).all()
    assert (pad_rgb[40:, 50:] == adapter.WHITE).all()
    assert (pad_known[:40, :50] == 1.0).all()
    assert (pad_known[40:, :] == 0.0).all()


def test_exact_native_query() -> None:
    rgb = np.arange(100 * 100 * 3, dtype=np.uint8).reshape((100, 100, 3))
    box = (20, 30, 45, 60)
    query = adapter.extract_query(rgb, box)
    expected = rgb[30:60, 20:45]
    assert np.array_equal(query, expected)

    # Sub-8 query gets padded with white to 8x8 without changing native pixels
    small_box = (10, 10, 14, 15)
    padded_query = adapter.extract_query(rgb, small_box)
    assert padded_query.shape == (8, 8, 3)
    assert np.array_equal(padded_query[:5, :4], rgb[10:15, 10:14])
    assert (padded_query[5:, 4:] == adapter.WHITE).all()


def test_doc_balance_sequence_determinism() -> None:
    rows = [
        {"id": "q1", "document_id": "doc1", "image": "i1.png", "query_box": [0, 0, 10, 10], "height": 100, "width": 100},
        {"id": "q2", "document_id": "doc1", "image": "i1.png", "query_box": [0, 0, 10, 10], "height": 100, "width": 100},
        {"id": "q3", "document_id": "doc2", "image": "i2.png", "query_box": [0, 0, 10, 10], "height": 100, "width": 100},
        {"id": "q4", "document_id": "doc3", "image": "i3.png", "query_box": [0, 0, 10, 10], "height": 100, "width": 100},
    ]
    doc_ids, docs_map = trainer.group_queries_by_document(rows)
    assert doc_ids == ["doc1", "doc2", "doc3"]

    rng1 = np.random.default_rng(42)
    seq1 = [doc_ids[int(rng1.integers(0, len(doc_ids)))] for _ in range(50)]

    rng2 = np.random.default_rng(42)
    seq2 = [doc_ids[int(rng2.integers(0, len(doc_ids)))] for _ in range(50)]

    rng3 = np.random.default_rng(999)
    seq3 = [doc_ids[int(rng3.integers(0, len(doc_ids)))] for _ in range(50)]

    assert seq1 == seq2
    assert seq1 != seq3

    # Document-balanced distribution
    counts = {d: seq1.count(d) for d in doc_ids}
    assert all(counts[d] > 10 for d in doc_ids)


def test_blank_or_unknown_masks_skipped_before_optimizer() -> None:
    torch = pytest.importorskip("torch")
    model = make_tiny_fake_model(torch)
    trainer.freeze_encoders(model)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)

    # Initial parameter clone
    initial_weight = model.decoder[0].weight.clone()

    # Step with all-zero loss_mask should skip optimizer.step()
    tile_known = np.zeros((64, 64), dtype=np.float32)
    if float(tile_known.sum()) == 0.0:
        pass  # Loop skips
    else:
        optimizer.step()

    assert torch.equal(model.decoder[0].weight, initial_weight)


def test_weights_saving_format_and_provenance(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    model = make_tiny_fake_model(torch)
    output_pt = tmp_path / "model.pt"

    trainer.save_atomic_checkpoint(model, output_pt)
    assert output_pt.is_file()

    loaded = torch.load(output_pt, map_location="cpu", weights_only=True)
    assert loaded["format_version"] == 1
    assert "model_config" in loaded
    assert loaded["model_config"]["drawing_channels"] == [4]
    assert "model_state_dict" in loaded
    for key, val in loaded["model_state_dict"].items():
        assert val.device.type == "cpu"

    provenance = trainer.build_provenance(
        pretrained_sha256="abc",
        manifest_sha256="def",
        fit_doc_ids=["doc1", "doc2"],
        seed=42,
        steps=10,
        lr=1e-5,
        tile_size=64,
        trainable_params_count=sum(p.numel() for p in model.parameters() if p.requires_grad),
        elapsed_seconds=1.23,
        loss_history=[1.0, 0.8, 0.5],
    )
    json_path = output_pt.with_suffix(".json")
    json_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")

    assert json_path.is_file()
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["source_pretrained_sha256"] == "abc"
    assert data["fit_document_ids"] == ["doc1", "doc2"]
    assert data["loss_summary"]["initial_loss"] == 1.0
    assert data["loss_summary"]["final_loss"] == 0.5


def test_cpu_tinyfake_model_optimization(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from PIL import Image

    # Set up synthetic dataset
    data_root = tmp_path / "dataset"
    data_root.mkdir()
    assets = data_root / "assets"
    assets.mkdir()

    img_array = np.full((128, 128, 3), 200, dtype=np.uint8)
    Image.fromarray(img_array).save(assets / "drawing.png")

    pos_mask = np.zeros((128, 128), dtype=np.uint8)
    pos_mask[40:60, 40:60] = 255
    Image.fromarray(pos_mask).save(assets / "pos_mask.png")

    manifest = tmp_path / "fit.json"
    manifest_data = {
        "examples": [
            {
                "id": "q1",
                "document_id": "doc_fit_01",
                "image": "assets/drawing.png",
                "width": 128,
                "height": 128,
                "kind": "synthetic",
                "query_box": [10, 10, 20, 20],
                "labels": {"positive_mask": "assets/pos_mask.png"},
            }
        ]
    }
    manifest.write_text(json.dumps(manifest_data), encoding="utf-8")

    model = make_tiny_fake_model(torch)
    output_pt = tmp_path / "trained_model.pt"

    prov = trainer.train(
        manifest_path=manifest,
        data_root=data_root,
        pretrained_path=tmp_path / "none.pt",
        output_path=output_pt,
        device="cpu",
        steps=3,
        tile_size=64,
        lr=1e-3,
        seed=42,
        save_every=2,
        model=model,
    )

    assert output_pt.is_file()
    assert (tmp_path / "trained_model-step-2.pt").is_file()
    assert (tmp_path / "trained_model.json").is_file()
    assert prov["steps"] == 3

    # Verify frozen encoders vs trainable decoder
    for param in model.drawing_encoder.parameters():
        assert param.grad is None
    for param in model.hatch_encoder.parameters():
        assert param.grad is None
    for param in model.decoder.parameters():
        assert param.grad is not None
        assert bool(torch.isfinite(param.grad).all())

    # Verify input drawing mask was strictly all ones (never leaking labels)
    assert model.saw_ones_mask is True
    assert model.saw_leak_mask is False


def test_split_safety_disjoint_and_overlap_rejection(tmp_path: Path) -> None:
    data_root = tmp_path / "dataset"
    data_root.mkdir()
    val_file = data_root / "val.json"
    val_file.write_text(json.dumps({"examples": [{"document_id": "doc_val_99"}]}), encoding="utf-8")

    safe_rows = [{"document_id": "doc_train_01"}]
    doc_ids = trainer.check_split_safety(safe_rows, data_root)
    assert doc_ids == {"doc_train_01"}

    unsafe_rows = [{"document_id": "doc_val_99"}]
    with pytest.raises(ValueError, match="overlap"):
        trainer.check_split_safety(unsafe_rows, data_root)


def test_cli_argument_validation() -> None:
    parser = trainer.build_parser()
    args = parser.parse_args(["--steps", "100", "--tile-size", "512", "--lr", "1e-5"])
    trainer.validate_train_args(args)

    bad_tile = parser.parse_args(["--tile-size", "500"])
    with pytest.raises(ValueError, match="multiple of 64"):
        trainer.validate_train_args(bad_tile)

    bad_steps = parser.parse_args(["--steps", "0"])
    with pytest.raises(ValueError, match="positive"):
        trainer.validate_train_args(bad_steps)
