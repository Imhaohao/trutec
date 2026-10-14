"""Score fit-trained embeddings on the development split and fuse with the HGB cache."""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = "0.4"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.7"

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_v5.devharness import HGB_CACHE, cached_scores, development_queries, load_rgb, report  # noqa: E402
from constants import HALO, TILE  # noqa: E402
from infer import embed_tiled, load_encoder, pick_device, prototype_from_embedding, rgb_tensor, scores_at_pixels  # noqa: E402


def main() -> None:
    args = parse_args()
    torch.set_num_threads(4)
    if not HGB_CACHE.is_dir():
        raise SystemExit(f"missing HGB cache: {HGB_CACHE}")
    device = pick_device(args.device)
    model = load_encoder(args.checkpoint, device)
    scores = _score_documents(model, args.tile, args.halo, device)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez(args.output / "scores.npz", **scores)
    hgb = cached_scores(HGB_CACHE)
    _same_layout(scores, hgb)
    written = {
        "contrastive": report("contrastive", scores, args.output / "contrastive.json"),
        "contrastive_x_hgb": report("contrastive_x_hgb", _combine(scores, hgb, _product), args.output / "contrastive_x_hgb.json"),
        "contrastive_min_hgb": report("contrastive_min_hgb", _combine(scores, hgb, _minimum), args.output / "contrastive_min_hgb.json"),
        "contrastive_geomean_hgb": report(
            "contrastive_geomean_hgb", _combine(scores, hgb, _geomean), args.output / "contrastive_geomean_hgb.json",
        ),
    }
    manifest = _manifest(args, written)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({name: item["selections"] for name, item in written.items()}, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tile", type=int, default=TILE)
    parser.add_argument("--halo", type=int, default=HALO)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    return parser.parse_args()


def _score_documents(model, tile: int, halo: int, device) -> dict[str, np.ndarray]:
    grouped: dict[str, list] = {}
    for query in development_queries():
        grouped.setdefault(query.image, []).append(query)
    scores = {}
    for image, group in grouped.items():
        started = time.perf_counter()
        embedding = embed_tiled(model, rgb_tensor(load_rgb(image)), tile, halo, device)
        for query in group:
            prototype = prototype_from_embedding(embedding, query.query_box)
            scores[query.id] = scores_at_pixels(embedding, prototype, query.y, query.x)
        del embedding
        print(f"embedded {image} queries {len(group)} sec {time.perf_counter() - started:.1f}", flush=True)
    return scores


def _combine(left, right, op) -> dict[str, np.ndarray]:
    return {key: op(left[key], right[key]).astype(np.float32) for key in left}


def _product(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return left * right


def _minimum(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.minimum(left, right)


def _geomean(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.sqrt(np.clip(left, 0.0, 1.0) * np.clip(right, 0.0, 1.0))


def _same_layout(scores, hgb) -> None:
    if set(scores) != set(hgb):
        raise RuntimeError("contrastive scores and the HGB cache do not cover the same queries")
    for key, value in scores.items():
        if value.shape != hgb[key].shape:
            raise RuntimeError(f"{key}: score length {value.shape} != HGB {hgb[key].shape}")


def _manifest(args, written) -> dict:
    return {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "tile": args.tile,
        "halo": args.halo,
        "device": args.device,
        "fusion": "development-selected among contrastive, product, minimum, and geometric mean with the frozen HGB cache",
        "selections": {name: item["selections"] for name, item in written.items()},
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
