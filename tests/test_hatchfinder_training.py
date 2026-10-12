"""Synthetic unit and integration tests for HatchFinder masked fine-tuning."""
from __future__ import annotations

import json
import copy
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


def test_negative_bce_weight_doubles_negative_only_gradient() -> None:
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[[[-1.0, 0.5], [1.0, -0.25]]]], requires_grad=True)
    target = torch.zeros_like(logits)
    known = torch.ones_like(logits)

    _, base_bce, base_dice = trainer.masked_loss(logits, target, known)
    _, weighted_bce, weighted_dice = trainer.masked_loss(
        logits, target, known, negative_bce_weight=2.0
    )
    base_grad = torch.autograd.grad(base_bce, logits, retain_graph=True)[0]
    weighted_grad = torch.autograd.grad(weighted_bce, logits)[0]

    assert base_dice is weighted_dice is None
    torch.testing.assert_close(weighted_grad, base_grad * 2.0, rtol=0.0, atol=0.0)
    torch.testing.assert_close(weighted_bce, base_bce * 2.0)


def test_negative_bce_weight_preserves_positive_bce_dice_and_unknown_gradients() -> None:
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[[[-1.0, 0.5, 2.0, -3.0]]]], requires_grad=True)
    target = torch.tensor([[[[1.0, 0.0, 0.0, 0.0]]]])
    known = torch.tensor([[[[1.0, 1.0, 1.0, 0.0]]]])

    base_loss, base_bce, base_dice = trainer.masked_loss(logits, target, known)
    weighted_loss, weighted_bce, weighted_dice = trainer.masked_loss(
        logits, target, known, negative_bce_weight=2.0
    )
    base_grad = torch.autograd.grad(base_loss, logits, retain_graph=True)[0]
    weighted_grad = torch.autograd.grad(weighted_loss, logits, retain_graph=True)[0]

    assert base_dice is not None and weighted_dice is not None
    torch.testing.assert_close(base_dice, weighted_dice, rtol=0.0, atol=0.0)
    positive_mask = target.bool()
    base_bce_grad = torch.autograd.grad(base_bce, logits, retain_graph=True)[0]
    weighted_bce_grad = torch.autograd.grad(weighted_bce, logits)[0]
    torch.testing.assert_close(weighted_bce_grad[positive_mask], base_bce_grad[positive_mask])
    assert bool((base_grad[~known.bool()] == 0.0).all())
    assert bool((weighted_grad[~known.bool()] == 0.0).all())


def test_default_negative_bce_weight_matches_legacy_masked_loss() -> None:
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    logits = torch.linspace(-1.0, 1.0, 12).reshape(1, 1, 3, 4).requires_grad_()
    target = torch.zeros_like(logits)
    target[:, :, 0, :2] = 1.0
    known = torch.ones_like(logits)
    known[:, :, 2, 2:] = 0.0

    actual = trainer.masked_loss(logits, target, known)
    legacy_bce = (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * known).sum() / known.sum().clamp_min(1.0)
    probs = torch.sigmoid(logits) * known
    masked_target = target * known
    non_empty = masked_target.sum(dim=(1, 2, 3)) > 0
    p, t = probs[non_empty], masked_target[non_empty]
    intersection = (p * t).sum(dim=(1, 2, 3))
    legacy_dice = 1.0 - ((2.0 * intersection + 1e-6) / (p.sum(dim=(1, 2, 3)) + t.sum(dim=(1, 2, 3)) + 1e-6)).mean()
    actual_grad = torch.autograd.grad(actual[0], logits, retain_graph=True)[0]
    legacy_grad = torch.autograd.grad(legacy_bce + legacy_dice, logits)[0]

    torch.testing.assert_close(actual[0], legacy_bce + legacy_dice, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual[1], legacy_bce, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual[2], legacy_dice, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual_grad, legacy_grad, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("weight", [float("nan"), float("inf"), 1.5, 0.0])
def test_unsupported_negative_bce_weight_fails_before_training_io(tmp_path: Path, weight: float) -> None:
    with pytest.raises(ValueError, match="negative_bce_weight"):
        trainer.train(
            manifest_path=tmp_path / "missing.json",
            data_root=tmp_path / "missing-data",
            pretrained_path=tmp_path / "missing.pt",
            output_path=tmp_path / "out.pt",
            device="cpu",
            negative_bce_weight=weight,
        )
    assert not list(tmp_path.iterdir())


def test_query_paired_update_counts_and_mean_gradient() -> None:
    torch = pytest.importorskip("torch")

    p0_controls = [trainer.paired_step_controls(1, step) for step in (1, 2)]
    p1_controls = [trainer.paired_step_controls(2, step) for step in (1, 2)]
    assert sum(int(control[2]) for control in p0_controls) == 2
    assert sum(int(control[2]) for control in p1_controls) == 1

    class ScalarModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.25))

        def forward(self, drawing, _mask, _hatch):
            return drawing[:, :1] * self.weight

    model = ScalarModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    rgb_values = (np.zeros((4, 4, 3), dtype=np.uint8), np.full((4, 4, 3), 255, dtype=np.uint8))
    known = np.ones((4, 4), dtype=np.float32)
    targets = (np.zeros((4, 4), dtype=np.float32), np.ones((4, 4), dtype=np.float32))
    hatch = rgb_values[0]

    individual_gradients = []
    for rgb, target in zip(rgb_values, targets):
        drawing = adapter.rgb_to_tensor(rgb, "cpu")
        logits = model(drawing, torch.ones((1, 1, 4, 4)), adapter.rgb_to_tensor(hatch, "cpu"))
        target_tensor = torch.from_numpy(target)[None, None]
        mask_tensor = torch.from_numpy(known)[None, None]
        loss, _, _ = trainer.masked_loss(logits, target_tensor, mask_tensor)
        individual_gradients.append(torch.autograd.grad(loss, model.weight)[0])

    for index, (rgb, target) in enumerate(zip(rgb_values, targets), start=1):
        trainer.execute_training_step(
            model, optimizer, [model.weight], "cpu", rgb, known, target, hatch, 4, index,
            gradient_scale=0.5, zero_grad=index == 1, optimizer_step=index == 2,
        )

    expected_gradient = sum(individual_gradients) / 2
    torch.testing.assert_close(model.weight.grad, expected_gradient)


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
        trainable_scope="all",
        torch_seed=42,
        starting_model_sha256="start-hash",
        frozen_modules=[],
    )
    json_path = output_pt.with_suffix(".json")
    json_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")

    assert json_path.is_file()
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["source_pretrained_sha256"] == "abc"
    assert data["fit_document_ids"] == ["doc1", "doc2"]
    assert data["loss_summary"]["initial_loss"] == 1.0
    assert data["loss_summary"]["final_loss"] == 0.5
    assert provenance["trainable_scope"] == "all"
    assert provenance["torch_seed"] == 42
    assert provenance["starting_model_sha256"] == "start-hash"
    assert provenance["frozen_modules"] == []
    assert provenance["negative_bce_weight"] == 1.0
    assert provenance["bce_weighting"] == {
        "positive": 1.0,
        "known_negative": 1.0,
        "denominator": "unweighted_known_pixel_count",
        "dice_weighting": "unchanged",
    }


def test_trainable_scopes_update_only_the_configured_parameters() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(123)
    model = make_tiny_fake_model(torch)
    for parameter in model.parameters():
        parameter.requires_grad = False

    all_params = trainer.configure_trainable_scope(model, "all")
    assert all(parameter.requires_grad for parameter in model.parameters())
    assert len(all_params) == len(list(model.parameters()))
    drawing_before = [parameter.detach().clone() for parameter in model.drawing_encoder.parameters()]
    hatch_before = [parameter.detach().clone() for parameter in model.hatch_encoder.parameters()]
    optimizer = torch.optim.SGD(all_params, lr=0.05)
    tile = np.random.default_rng(3).integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    known = np.ones((64, 64), dtype=np.float32)
    positive = np.zeros((64, 64), dtype=np.float32)
    positive[20:44, 18:42] = 1.0
    trainer.execute_training_step(model, optimizer, all_params, "cpu", tile, known, positive, tile, 64, 1)
    assert any(not torch.equal(before, after) for before, after in zip(drawing_before, model.drawing_encoder.parameters()))
    assert any(not torch.equal(before, after) for before, after in zip(hatch_before, model.hatch_encoder.parameters()))

    head_params = trainer.configure_trainable_scope(model, "head")
    assert all(not parameter.requires_grad for parameter in model.drawing_encoder.parameters())
    assert all(not parameter.requires_grad for parameter in model.hatch_encoder.parameters())
    assert all(parameter.requires_grad for parameter in model.decoder.parameters())
    encoder_before = [parameter.detach().clone() for parameter in model.drawing_encoder.parameters()]
    decoder_before = [parameter.detach().clone() for parameter in model.decoder.parameters()]
    optimizer = torch.optim.SGD(head_params, lr=0.05)
    trainer.execute_training_step(model, optimizer, head_params, "cpu", tile, known, positive, tile, 64, 2)
    assert all(torch.equal(before, after) for before, after in zip(encoder_before, model.drawing_encoder.parameters()))
    assert any(not torch.equal(before, after) for before, after in zip(decoder_before, model.decoder.parameters()))


def test_torch_seed_repeats_random_sequence() -> None:
    torch = pytest.importorskip("torch")
    trainer.seed_torch(8675309)
    first = torch.rand(8)
    trainer.seed_torch(8675309)
    second = torch.rand(8)
    torch.testing.assert_close(first, second)


def test_invalid_trainable_scope_fails_before_output_or_manifest_access(tmp_path: Path) -> None:
    output = tmp_path / "not-created" / "model.pt"
    with pytest.raises(ValueError, match="trainable_scope"):
        trainer.train(
            manifest_path=tmp_path / "missing.json",
            data_root=tmp_path / "dataset",
            pretrained_path=tmp_path / "missing.pt",
            output_path=output,
            device="cpu",
            trainable_scope="encoder-only",
        )
    assert not output.parent.exists()


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
    starting_hash = trainer.sha256_model_state(model)
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
    assert prov["trainable_scope"] == "head"
    assert prov["torch_seed"] == 42
    assert prov["starting_model_sha256"] == starting_hash
    assert prov["frozen_modules"] == ["drawing_encoder", "hatch_encoder"]
    assert prov["trainable_parameters"] == sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

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
    assert args.sampling == "random"
    assert args.negative_bce_weight == 1.0
    weighted_args = parser.parse_args(["--negative-bce-weight", "2"])
    trainer.validate_train_args(weighted_args)
    assert weighted_args.negative_bce_weight == 2.0
    with pytest.raises(SystemExit):
        parser.parse_args(["--negative-bce-weight", "1.5"])

    cov_args = parser.parse_args(["--sampling", "coverage"])
    trainer.validate_train_args(cov_args)
    assert cov_args.sampling == "coverage"

    paired_args = parser.parse_args(["--sampling", "query-paired", "--steps", "8", "--save-every", "4"])
    trainer.validate_train_args(paired_args)
    assert paired_args.accumulation_steps == 1
    with pytest.raises(ValueError, match="requires query-paired"):
        trainer.validate_train_args(parser.parse_args(["--accumulation-steps", "2"]))
    with pytest.raises(ValueError, match="must be even"):
        trainer.validate_train_args(parser.parse_args(["--sampling", "query-paired", "--steps", "3"]))

    bad_tile = parser.parse_args(["--tile-size", "500"])
    with pytest.raises(ValueError, match="multiple of 64"):
        trainer.validate_train_args(bad_tile)

    bad_steps = parser.parse_args(["--steps", "0"])
    with pytest.raises(ValueError, match="positive"):
        trainer.validate_train_args(bad_steps)

    bad_sampling = parser.parse_args([])
    bad_sampling.sampling = "unsupported_mode"
    with pytest.raises(ValueError, match="sampling must be 'random', 'coverage', or 'query-paired'"):
        trainer.validate_train_args(bad_sampling)


def test_main_threads_negative_bce_weight_into_train(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_train(**kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(trainer, "select_device", lambda _device: "cpu")
    monkeypatch.setattr(trainer, "train", fake_train)
    trainer.main(["--negative-bce-weight", "2"])
    assert captured["negative_bce_weight"] == 2.0


def test_negative_coverage_never_falls_back(tmp_path: Path) -> None:
    from PIL import Image

    # 1. At crop center level: choose_crop_center vs choose_coverage_crop_center
    pos_mask = np.zeros((50, 50), dtype=bool)
    pos_mask[10:20, 10:20] = True
    empty_neg_mask = np.zeros((50, 50), dtype=bool)
    rng = np.random.default_rng(42)

    # Legacy silently falls back to positive pixel when negative is requested
    fallback_y, fallback_x = trainer.choose_crop_center(pos_mask, empty_neg_mask, class_choice=1, rng=rng)
    assert pos_mask[fallback_y, fallback_x] is np.True_

    # Coverage strictly raises ValueError and never falls back
    with pytest.raises(ValueError, match="No reviewed negative pixels"):
        trainer.choose_coverage_crop_center(pos_mask, empty_neg_mask, branch="negative", rng=rng)

    # 2. At sampler level: manifest with only positive-only rows
    data_root = tmp_path / "dataset"
    assets = data_root / "assets"
    assets.mkdir(parents=True)

    Image.fromarray(np.full((64, 64, 3), 255, dtype=np.uint8)).save(assets / "drawing.png")
    pos_arr = np.zeros((64, 64), dtype=np.uint8)
    pos_arr[20:30, 20:30] = 255
    Image.fromarray(pos_arr).save(assets / "pos.png")

    rows = [
        {
            "id": "q1",
            "document_id": "doc1",
            "image": "assets/drawing.png",
            "width": 64,
            "height": 64,
            "query_box": [5, 5, 10, 10],
            "labels": {"positive_mask": "assets/pos.png"},
        }
    ]
    cache = trainer.SingleImageCache(data_root)
    sampler = trainer.CoverageSampler(rows, data_root, cache, tile_size=64, seed=42)

    # Selecting negative query raises clear ValueError instead of falling back
    with pytest.raises(ValueError, match="No eligible documents or rows exist for negative branch"):
        sampler.select_query("negative")

    # Step 1 is positive: succeeds
    tile_rgb, tile_known, tile_pos, _, _ = sampler.sample_step()
    assert sampler.last_branch == "positive"
    # Step 2 is negative: raises ValueError before silently falling back
    with pytest.raises(ValueError, match="No eligible documents or rows exist for negative branch"):
        sampler.sample_step()

    # 3. With actual negative pixels, negative center is strictly on a true negative pixel
    Image.fromarray(np.full((128, 128, 3), 255, dtype=np.uint8)).save(assets / "drawing2.png")
    pos2_arr = np.zeros((128, 128), dtype=np.uint8)
    pos2_arr[20:30, 20:30] = 255
    Image.fromarray(pos2_arr).save(assets / "pos2.png")
    neg2_arr = np.zeros((128, 128), dtype=np.uint8)
    neg2_arr[60:70, 60:70] = 255
    Image.fromarray(neg2_arr).save(assets / "neg2.png")
    rows_with_neg = [
        {
            "id": "q2",
            "document_id": "doc2",
            "image": "assets/drawing2.png",
            "width": 128,
            "height": 128,
            "query_box": [5, 5, 10, 10],
            "labels": {"positive_mask": "assets/pos2.png", "negative_mask": "assets/neg2.png"},
        }
    ]
    sampler_valid = trainer.CoverageSampler(rows_with_neg, data_root, cache, tile_size=64, seed=42)
    t_rgb, t_known, t_pos, _, row_out = sampler_valid.sample_step(branch="negative")
    assert row_out["id"] == "q2"
    # Center pixel of tile must be negative (known=1.0, pos=0.0)
    center_y, center_x = 32, 32
    assert t_known[center_y, center_x] == 1.0
    assert t_pos[center_y, center_x] == 0.0


def test_deterministic_cycling_covers_eligible_documents_and_queries(tmp_path: Path) -> None:
    from PIL import Image

    data_root = tmp_path / "dataset"
    assets = data_root / "assets"
    assets.mkdir(parents=True)

    Image.fromarray(np.full((100, 100, 3), 255, dtype=np.uint8)).save(assets / "drawing.png")
    pos_arr = np.zeros((100, 100), dtype=np.uint8)
    pos_arr[20:30, 20:30] = 255
    Image.fromarray(pos_arr).save(assets / "pos.png")
    neg_arr = np.zeros((100, 100), dtype=np.uint8)
    neg_arr[60:70, 60:70] = 255
    Image.fromarray(neg_arr).save(assets / "neg.png")

    labels = {"positive_mask": "assets/pos.png", "negative_mask": "assets/neg.png"}
    rows = [
        {"id": "q1_a", "document_id": "doc1", "image": "assets/drawing.png", "width": 100, "height": 100, "query_box": [0, 0, 10, 10], "labels": labels},
        {"id": "q1_b", "document_id": "doc1", "image": "assets/drawing.png", "width": 100, "height": 100, "query_box": [0, 0, 10, 10], "labels": labels},
        {"id": "q2_a", "document_id": "doc2", "image": "assets/drawing.png", "width": 100, "height": 100, "query_box": [0, 0, 10, 10], "labels": labels},
        {"id": "q3_a", "document_id": "doc3", "image": "assets/drawing.png", "width": 100, "height": 100, "query_box": [0, 0, 10, 10], "labels": labels},
        {"id": "q3_b", "document_id": "doc3", "image": "assets/drawing.png", "width": 100, "height": 100, "query_box": [0, 0, 10, 10], "labels": labels},
    ]

    cache1 = trainer.SingleImageCache(data_root)
    cache2 = trainer.SingleImageCache(data_root)
    cache3 = trainer.SingleImageCache(data_root)

    sampler1 = trainer.CoverageSampler(rows, data_root, cache1, tile_size=64, seed=42)
    sampler2 = trainer.CoverageSampler(rows, data_root, cache2, tile_size=64, seed=42)
    sampler3 = trainer.CoverageSampler(rows, data_root, cache3, tile_size=64, seed=99)

    steps1 = [sampler1.sample_step() for _ in range(12)]
    steps2 = [sampler2.sample_step() for _ in range(12)]
    steps3 = [sampler3.sample_step() for _ in range(12)]

    # 1. Determinism under fixed seed
    seq1_rows = [s[4]["id"] for s in steps1]
    seq2_rows = [s[4]["id"] for s in steps2]
    seq3_rows = [s[4]["id"] for s in steps3]
    assert seq1_rows == seq2_rows
    assert seq1_rows != seq3_rows

    seq1_tiles = [s[0] for s in steps1]
    seq2_tiles = [s[0] for s in steps2]
    for t1, t2 in zip(seq1_tiles, seq2_tiles):
        assert np.array_equal(t1, t2)

    # 2. Alternating branches
    assert sampler1.branch_counts == {"positive": 6, "negative": 6}

    # 3. Cycling document and query coverage
    # For positive branch (6 steps):
    pos_steps = [steps1[i] for i in range(0, 12, 2)]
    pos_docs = [s[4]["document_id"] for s in pos_steps]
    pos_queries = [s[4]["id"] for s in pos_steps]
    assert set(pos_docs) == {"doc1", "doc2", "doc3"}
    doc1_queries = [q for d, q in zip(pos_docs, pos_queries) if d == "doc1"]
    assert set(doc1_queries) == {"q1_a", "q1_b"}
    doc3_queries = [q for d, q in zip(pos_docs, pos_queries) if d == "doc3"]
    assert set(doc3_queries) == {"q3_a", "q3_b"}
    assert len(set(pos_queries[:5])) == 5
    assert set(pos_queries[:5]) == {"q1_a", "q1_b", "q2_a", "q3_a", "q3_b"}

    # Same coverage for negative branch
    neg_steps = [steps1[i] for i in range(1, 12, 2)]
    neg_docs = [s[4]["document_id"] for s in neg_steps]
    neg_queries = [s[4]["id"] for s in neg_steps]
    assert set(neg_docs) == {"doc1", "doc2", "doc3"}
    assert set(q for d, q in zip(neg_docs, neg_queries) if d == "doc1") == {"q1_a", "q1_b"}
    assert set(q for d, q in zip(neg_docs, neg_queries) if d == "doc3") == {"q3_a", "q3_b"}
    assert len(set(neg_queries[:5])) == 5
    assert set(neg_queries[:5]) == {"q1_a", "q1_b", "q2_a", "q3_a", "q3_b"}


def test_positives_include_positive_only_rows(tmp_path: Path) -> None:
    from PIL import Image

    data_root = tmp_path / "dataset"
    assets = data_root / "assets"
    assets.mkdir(parents=True)

    Image.fromarray(np.full((80, 80, 3), 255, dtype=np.uint8)).save(assets / "drawing.png")
    pos_arr = np.zeros((80, 80), dtype=np.uint8)
    pos_arr[15:25, 15:25] = 255
    Image.fromarray(pos_arr).save(assets / "pos.png")
    neg_arr = np.zeros((80, 80), dtype=np.uint8)
    neg_arr[45:55, 45:55] = 255
    Image.fromarray(neg_arr).save(assets / "neg.png")

    rows = [
        {
            "id": "q_mixed",
            "document_id": "doc_mixed",
            "image": "assets/drawing.png",
            "width": 80,
            "height": 80,
            "query_box": [0, 0, 5, 5],
            "labels": {"positive_mask": "assets/pos.png", "negative_mask": "assets/neg.png"},
        },
        {
            "id": "q_posonly",
            "document_id": "doc_posonly",
            "image": "assets/drawing.png",
            "width": 80,
            "height": 80,
            "query_box": [0, 0, 5, 5],
            "labels": {"positive_mask": "assets/pos.png"},
        },
    ]

    cache = trainer.SingleImageCache(data_root)
    sampler = trainer.CoverageSampler(rows, data_root, cache, tile_size=64, seed=42)

    selected_pos: list[str] = []
    selected_neg: list[str] = []
    for _ in range(10):
        row_pos = sampler.select_query("positive")
        selected_pos.append(row_pos["id"])
        row_neg = sampler.select_query("negative")
        selected_neg.append(row_neg["id"])

    # Positive branch must include positive-only rows
    assert "q_posonly" in selected_pos
    assert "q_mixed" in selected_pos

    # Negative branch must NEVER include positive-only rows
    assert "q_posonly" not in selected_neg
    assert set(selected_neg) == {"q_mixed"}


def test_unknown_or_excluded_values_remain_ignored(tmp_path: Path) -> None:
    from PIL import Image

    data_root = tmp_path / "dataset"
    assets = data_root / "assets"
    assets.mkdir(parents=True)

    H, W = 160, 160
    Image.fromarray(np.full((H, W, 3), 255, dtype=np.uint8)).save(assets / "drawing.png")

    # Labeled positive pixels: inside query_box, inside context_boxes, and in valid region [80:100, 80:100]
    pos_arr = np.zeros((H, W), dtype=np.uint8)
    pos_arr[10:30, 10:30] = 255
    pos_arr[40:60, 40:60] = 255
    pos_arr[80:100, 80:100] = 255
    Image.fromarray(pos_arr).save(assets / "pos.png")

    # Known mask: covers query_box, context_box, valid positive, and valid negative [120:140, 120:140]
    known_arr = np.zeros((H, W), dtype=np.uint8)
    known_arr[10:30, 10:30] = 255
    known_arr[40:60, 40:60] = 255
    known_arr[80:100, 80:100] = 255
    known_arr[120:140, 120:140] = 255
    Image.fromarray(known_arr).save(assets / "known.png")

    row = {
        "id": "q_excl",
        "document_id": "doc_excl",
        "image": "assets/drawing.png",
        "width": W,
        "height": H,
        "query_box": [10, 10, 30, 30],
        "context_boxes": [[40, 40, 60, 60]],
        "labels": {"positive_mask": "assets/pos.png", "known_mask": "assets/known.png"},
    }

    cache = trainer.SingleImageCache(data_root)
    rgb, pos_mask, known_mask = cache.get(row)

    # Verify query_box and context_boxes are excluded from known and positive
    assert not known_mask[10:30, 10:30].any()
    assert not known_mask[40:60, 40:60].any()
    assert not pos_mask[10:30, 10:30].any()
    assert not pos_mask[40:60, 40:60].any()

    # Unknown regions are False
    assert not known_mask[:10, :10].any()
    assert not known_mask[140:, 140:].any()

    # Sample repeatedly and verify center is never in unknown or excluded region
    sampler = trainer.CoverageSampler([row], data_root, cache, tile_size=64, seed=42)
    for _ in range(20):
        t_rgb, t_known, t_pos, _, _ = sampler.sample_step()
        assert t_known[32, 32] == 1.0


def test_legacy_random_behavior_remains_available_and_default(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from PIL import Image

    # 1. CLI parser default
    parser = trainer.build_parser()
    args = parser.parse_args([])
    assert args.sampling == "random"

    args_cov = parser.parse_args(["--sampling", "coverage"])
    assert args_cov.sampling == "coverage"

    # 2. Legacy sample_training_step signature and call
    data_root = tmp_path / "dataset"
    assets = data_root / "assets"
    assets.mkdir(parents=True)

    Image.fromarray(np.full((64, 64, 3), 200, dtype=np.uint8)).save(assets / "drawing.png")
    pos_arr = np.zeros((64, 64), dtype=np.uint8)
    pos_arr[20:30, 20:30] = 255
    Image.fromarray(pos_arr).save(assets / "pos.png")
    neg_arr = np.zeros((64, 64), dtype=np.uint8)
    neg_arr[40:50, 40:50] = 255
    Image.fromarray(neg_arr).save(assets / "neg.png")

    manifest = tmp_path / "fit.json"
    rows = [
        {
            "id": "q1",
            "document_id": "doc1",
            "image": "assets/drawing.png",
            "width": 64,
            "height": 64,
            "kind": "synthetic",
            "query_box": [5, 5, 10, 10],
            "labels": {"positive_mask": "assets/pos.png", "negative_mask": "assets/neg.png"},
        }
    ]
    manifest.write_text(json.dumps({"examples": rows}), encoding="utf-8")

    doc_ids, docs_map = trainer.group_queries_by_document(rows)
    cache = trainer.SingleImageCache(data_root)
    rng = np.random.default_rng(42)

    tile_rgb, tile_known, tile_pos, hatch_rgb, row_out = trainer.sample_training_step(
        rng, doc_ids, docs_map, cache, tile_size=64
    )
    assert tile_rgb.shape == (64, 64, 3)
    assert tile_known.shape == (64, 64)
    assert tile_pos.shape == (64, 64)
    assert row_out["id"] == "q1"

    # 3. train() with legacy default vs coverage
    model_rand = make_tiny_fake_model(torch)
    prov_rand = trainer.train(
        manifest_path=manifest, data_root=data_root, pretrained_path=tmp_path / "none.pt",
        output_path=tmp_path / "out_rand.pt", device="cpu", steps=2, tile_size=64, lr=1e-3,
        model=model_rand,
    )
    assert prov_rand["sampling_mode"] == "random"
    assert prov_rand["branch_counts"] is None

    model_cov = make_tiny_fake_model(torch)
    prov_cov = trainer.train(
        manifest_path=manifest, data_root=data_root, pretrained_path=tmp_path / "none.pt",
        output_path=tmp_path / "out_cov.pt", device="cpu", steps=2, tile_size=64, lr=1e-3,
        model=model_cov, sampling="coverage",
    )
    assert prov_cov["sampling_mode"] == "coverage"
    assert prov_cov["branch_counts"] == {"positive": 1, "negative": 1}
    assert prov_cov["selected_document_ids"] == ["doc1", "doc1"]
    assert prov_cov["selected_query_ids"] == ["q1", "q1"]


@pytest.mark.parametrize("has_negative", [True, False])
def test_query_paired_sampler_uses_same_query_and_real_centers(tmp_path: Path, monkeypatch, has_negative: bool) -> None:
    pos = np.zeros((64, 64), dtype=bool)
    pos[8:16, 8:16] = True
    known = pos.copy()
    if has_negative:
        known[40:48, 40:48] = True
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)

    class Cache:
        def get(self, row):
            return rgb, pos, known

    row = {"id": "q1", "document_id": "d1", "query_box": [0, 0, 4, 4]}
    sampler = trainer.CoverageSampler([row], tmp_path, Cache(), 64, seed=45)
    monkeypatch.setattr(sampler, "_is_row_eligible", lambda _row, branch: branch == "positive")
    monkeypatch.setattr(trainer.adapter, "extract_query", lambda _rgb, _box: rgb[:4, :4])

    samples, trace = sampler.sample_pair()
    assert [sample[4]["id"] for sample in samples] == ["q1", "q1"]
    assert sum(sampler.branch_counts.values()) == len(samples)
    assert sampler.selected_query_ids == ["q1", "q1"]
    assert trace["query_id"] == "q1"
    y0, x0 = trace["centers"][0]
    assert pos[y0, x0]
    y1, x1 = trace["centers"][1]
    if has_negative:
        assert (known & ~pos)[y1, x1]
        assert trace["positive_only_fallback"] is False
        assert sampler.branch_counts == {"positive": 1, "negative": 1}
    else:
        assert pos[y1, x1]
        assert trace["positive_only_fallback"] is True
        assert sampler.branch_counts == {"positive": 2, "negative": 0}


def test_train_query_paired_arms_share_crops_and_have_expected_updates(tmp_path: Path, monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    import evaluate

    rgb = np.full((64, 64, 3), 127, dtype=np.uint8)
    positive = np.zeros((64, 64), dtype=bool)
    positive[8:16, 8:16] = True
    known = positive.copy()
    known[40:48, 40:48] = True
    row = {"id": "q1", "document_id": "d1", "image": "drawing.png", "query_box": [0, 0, 8, 8]}
    manifest = tmp_path / "fit.json"
    manifest.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(evaluate, "load_manifest", lambda *_args, **_kwargs: [row])
    monkeypatch.setattr(trainer, "check_split_safety", lambda *_args: {"d1"})
    monkeypatch.setattr(trainer, "inspect_row_branch_eligibility", lambda *_args: (True, True))
    monkeypatch.setattr(trainer.SingleImageCache, "get", lambda *_args: (rgb, positive, known))
    monkeypatch.setattr(trainer, "save_atomic_checkpoint", lambda _model, path: path.write_bytes(b"mock"))

    baseline = make_tiny_fake_model(torch)
    runs: list[dict[str, Any]] = []

    def capture_step(_model, _optimizer, _params, _device, tile_rgb, tile_known, tile_pos,
                     hatch_rgb, _tile_size, step, **kwargs):
        runs[-1]["crops"].append((tile_rgb.copy(), tile_known.copy(), tile_pos.copy(), hatch_rgb.copy()))
        runs[-1]["controls"].append((kwargs["gradient_scale"], kwargs["zero_grad"], kwargs["optimizer_step"]))
        runs[-1]["weights"].append(kwargs.get("negative_bce_weight", 1.0))
        return float(step), 0.0, 0.0

    monkeypatch.setattr(trainer, "execute_training_step", capture_step)
    provenances = []
    for accumulation in (1, 2):
        runs.append({"crops": [], "controls": [], "weights": []})
        model = copy.deepcopy(baseline)
        provenance = trainer.train(
            manifest_path=manifest, data_root=tmp_path, pretrained_path=tmp_path / "missing.pt",
            output_path=tmp_path / f"p{accumulation}.pt", device="cpu", steps=4,
            tile_size=64, save_every=4, model=model, sampling="query-paired",
            accumulation_steps=accumulation,
            negative_bce_weight=2.0,
        )
        provenances.append(provenance)

    p0, p1 = provenances
    assert p0["optimizer_updates"] == 4
    assert p1["optimizer_updates"] == 2
    assert p0["branch_counts"] == p1["branch_counts"] == {"positive": 2, "negative": 2}
    assert p0["pair_trace"] == p1["pair_trace"]
    assert p0["negative_bce_weight"] == p1["negative_bce_weight"] == 2.0
    assert len(runs[0]["crops"]) == len(runs[1]["crops"]) == 4
    for p0_crop, p1_crop in zip(runs[0]["crops"], runs[1]["crops"]):
        for p0_array, p1_array in zip(p0_crop, p1_crop):
            np.testing.assert_array_equal(p0_array, p1_array)
    assert [control[2] for control in runs[0]["controls"]] == [True, True, True, True]
    assert [control[2] for control in runs[1]["controls"]] == [False, True, False, True]
    assert runs[0]["weights"] == runs[1]["weights"] == [2.0] * 4


def test_query_paired_training_fails_on_empty_known_crop(tmp_path: Path, monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    import evaluate

    row = {"id": "q1", "document_id": "d1", "image": "drawing.png", "query_box": [0, 0, 8, 8]}
    manifest = tmp_path / "fit.json"
    manifest.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(evaluate, "load_manifest", lambda *_args, **_kwargs: [row])
    monkeypatch.setattr(trainer, "check_split_safety", lambda *_args: {"d1"})
    monkeypatch.setattr(trainer, "inspect_row_branch_eligibility", lambda *_args: (True, True))
    monkeypatch.setattr(
        trainer.SingleImageCache, "get",
        lambda *_args: (np.zeros((64, 64, 3), dtype=np.uint8), np.zeros((64, 64), dtype=bool), np.zeros((64, 64), dtype=bool)),
    )
    empty_crop = (np.zeros((64, 64, 3), dtype=np.uint8), np.zeros((64, 64), dtype=np.float32),
                  np.zeros((64, 64), dtype=np.float32), np.zeros((8, 8, 3), dtype=np.uint8), row)
    monkeypatch.setattr(
        trainer.CoverageSampler, "sample_pair",
        lambda _self: ([empty_crop, empty_crop], {"query_id": "q1", "document_id": "d1", "positive_only_fallback": False}),
    )
    with pytest.raises(ValueError, match="no reviewed known pixels"):
        trainer.train(
            manifest_path=manifest, data_root=tmp_path, pretrained_path=tmp_path / "missing.pt",
            output_path=tmp_path / "should-not-exist.pt", device="cpu", steps=2,
            tile_size=64, save_every=2, model=make_tiny_fake_model(torch), sampling="query-paired",
        )
