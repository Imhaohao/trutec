"""Cache full-drawing texture probabilities for every development query.

Uses the crop-descriptor HistGradientBoosting model and only request fields.
"""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research_v5"))

from devharness import development_queries, load_gray  # noqa: E402
import hatch_texture  # noqa: E402
from inference_texture import predict_probability_grid, verify_model_metadata  # noqa: E402
from texture_learning import load_texture_model  # noqa: E402

MODEL = ROOT / "models/models/texture-hgb-crop-v1.joblib"
CACHE = Path(__file__).resolve().parent / "cache" / "probs"


def cache_path(query_id: str) -> Path:
    if not query_id or "/" in query_id or "\\" in query_id:
        raise ValueError(f"unsafe query id: {query_id}")
    return CACHE / f"{query_id}.npy"


def grouped_queries() -> dict[str, list]:
    groups: dict[str, list] = {}
    for query in development_queries():
        groups.setdefault(query.image, []).append(query)
    return groups


def _save(path: Path, probability: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npy")
    np.save(temporary, probability.astype(np.float16))
    temporary.replace(path)


def _predict_group(model, metadata: dict, gray: np.ndarray, queries: list) -> None:
    stride = int(metadata["stride"])
    source = metadata["query_descriptor_source"]
    features = hatch_texture.texture_features(gray, stride=stride)
    for query in queries:
        destination = cache_path(query.id)
        if destination.is_file():
            continue
        started = time.perf_counter()
        descriptor = hatch_texture.descriptor_for_query(
            gray, features, query.query_box, stride, source,
        )
        probability = predict_probability_grid(model, features, descriptor)
        _save(destination, probability)
        print(
            f"{query.id} shape={probability.shape} mean={float(probability.mean()):.4f} "
            f"sec={time.perf_counter() - started:.1f}",
            flush=True,
        )
    del features


def main() -> None:
    model, metadata = load_texture_model(MODEL)
    verify_model_metadata(metadata)
    if metadata.get("query_descriptor_source") != "crop" or int(metadata.get("stride", 0)) != 4:
        raise RuntimeError(f"unexpected model metadata: {metadata.get('query_descriptor_source')}")
    started = time.perf_counter()
    for image, queries in grouped_queries().items():
        if all(cache_path(query.id).is_file() for query in queries):
            print(f"cached {image}", flush=True)
            continue
        gray = load_gray(image)
        print(f"features {image} {gray.shape}", flush=True)
        _predict_group(model, metadata, gray, queries)
        del gray
    print(f"done sec={time.perf_counter() - started:.1f}", flush=True)


if __name__ == "__main__":
    main()
