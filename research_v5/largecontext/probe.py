"""CPU memorization check. One synthetic sheet, no dataset files."""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import numpy as np
import torch

from .geom import area_half_rgb, cell_fractions, pack_hwc, square_window, tile_repeat
from .loss import masked_loss
from .model import HatchContextNet


def main() -> None:
    torch.manual_seed(1234)
    device = torch.device("cpu")
    batch = _batch(_scene())
    model = HatchContextNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    first = _train(model, opt, batch, device, steps=1)
    rest = _train(model, opt, batch, device, steps=39)
    gap = _gap(model, batch, device)
    print(
        f"first_loss {first['loss']:.4f} first_grad {first['grad']:.4f} "
        f"last_loss {rest['loss']:.4f} gap {gap['gap']:.4f} "
        f"pos {gap['pos']:.4f} neg {gap['neg']:.4f} known {gap['known']}",
        flush=True,
    )


def _scene():
    side = 128
    rgb = np.full((side, side, 3), 235, np.uint8)
    rgb[:, 0:64:4] = (15, 15, 15)
    rgb[0::4, 64:] = (210, 40, 40)
    known = np.ones((side, side), dtype=bool)
    positive = np.zeros((side, side), dtype=bool)
    positive[:, :64] = True
    box = (8, 8, 40, 40)
    known[box[1] : box[3], box[0] : box[2]] = False
    positive &= known
    return rgb, known, positive, box


def _batch(scene) -> dict[str, np.ndarray]:
    rgb, known, positive, box = scene
    window, known_c, pos_c = square_window(rgb, known, positive, 64, 32, 128)
    x0, y0, x1, y1 = box
    drawing = pack_hwc(area_half_rgb(window))
    query = pack_hwc(tile_repeat(area_half_rgb(rgb[y0:y1, x0:x1])))
    known_frac, pos_frac = cell_fractions(known_c, pos_c)
    return {
        "drawing": drawing[None],
        "query": query[None],
        "known": known_frac[None],
        "positive": pos_frac[None],
    }


def _train(model, opt, batch, device, steps: int) -> dict:
    last = {"loss": 0.0, "grad": 0.0}
    for _ in range(steps):
        last = _step(model, opt, batch, device)
    return last


def _step(model, opt, batch, device) -> dict:
    model.train()
    opt.zero_grad(set_to_none=True)
    loss, stats = masked_loss(_forward(model, batch, device), _tensor(batch["known"], device), _tensor(batch["positive"], device))
    loss.backward()
    grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
    opt.step()
    return {"loss": float(stats["loss"]), "grad": grad}


def _gap(model, batch, device) -> dict:
    model.eval()
    with torch.inference_mode():
        prob = torch.sigmoid(_forward(model, batch, device))[0, 0]
    known = torch.from_numpy(batch["known"][0] >= 0.75)
    positive = torch.from_numpy(batch["positive"][0] >= 0.5)
    pos = prob[known & positive]
    neg = prob[known & ~positive]
    return {
        "pos": float(pos.mean()),
        "neg": float(neg.mean()),
        "gap": float(pos.mean() - neg.mean()),
        "known": int(known.sum()),
    }


def _forward(model, batch, device) -> torch.Tensor:
    return model(_tensor(batch["drawing"], device), _tensor(batch["query"], device))


def _tensor(array: np.ndarray, device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(array)).to(device)


if __name__ == "__main__":
    main()
