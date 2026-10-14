"""Full-drawing texture probability grids for development queries.

Prediction uses inference_texture.predict_probability_grid and the query
descriptor stored in the model metadata. Reviewed coordinates are read only
when sampling those grids for the identity check and for later scoring.
"""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"

import gc
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research_v5"))

import hatch_texture  # noqa: E402
from devharness import DATA_ROOT, HGB_CACHE, development_queries, request_view  # noqa: E402
from evaluate_texture_sweep import remap_known_coordinates  # noqa: E402
from inference_texture import (  # noqa: E402
    native_grid_coordinates,
    predict_probability_grid,
    verify_model_metadata,
)
from texture_learning import grid_centers, load_texture_model  # noqa: E402

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
GRIDS = CACHE / "grids"
PIXELS = CACHE / "pixels"
BUILD_PATH = CACHE / "build.json"
FINDINGS = HERE / "findings.txt"
IDENTITY_LIMIT = 1e-5
STAMP = "sibling-grid-v1"


def model_path() -> Path:
    requested = ROOT / "models" / "texture-hgb-crop-v1.joblib"
    packaged = ROOT / "models" / "models" / "texture-hgb-crop-v1.joblib"
    if requested.is_file():
        return requested
    if packaged.is_file():
        return packaged
    raise FileNotFoundError(requested)


def load_inference_gray(relative_image: str, height: int, width: int) -> np.ndarray:
    """OpenCV grayscale, the loader inference.read_drawing uses.

    PIL ``convert('L')`` differs by up to one level and misses the HGB cache.
    """
    gray = cv2.imread(str(DATA_ROOT / relative_image), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(relative_image)
    if gray.shape != (height, width):
        raise ValueError(f"{relative_image}: gray shape {gray.shape} != {(height, width)}")
    return gray


def development_requests() -> list[dict]:
    manifest = json.loads((ROOT / "artifacts/splits/development-real.json").read_text(encoding="utf-8"))
    rows = manifest["examples"] if isinstance(manifest, dict) else manifest
    return [request_view(row) for row in rows]


def group_requests(requests: list[dict]) -> list[tuple[str, list[dict]]]:
    grouped: dict[str, list[dict]] = {}
    for request in requests:
        grouped.setdefault(request["image"], []).append(request)
    ordered = sorted(grouped.items(), key=lambda item: _area(item[1]))
    return ordered


def sample_probability(grid: np.ndarray, y: np.ndarray, x: np.ndarray, height: int, width: int, stride: int) -> np.ndarray:
    """Bilinear sample matching interpolate_and_threshold_mask, then clip like the cache."""
    gy, gx = native_grid_coordinates(height, width, stride)
    contiguous = np.ascontiguousarray(grid, dtype=np.float32)
    return remap_known_coordinates(contiguous, gx[x], gy[y])


def descriptor_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """One minus the mean symmetric relative gap used by the texture pair features."""
    gap = np.abs(left.astype(np.float64) - right.astype(np.float64))
    scale = np.abs(left.astype(np.float64)) + np.abs(right.astype(np.float64)) + 1e-4
    return float(1.0 - np.mean(gap / scale))


def median_in_box(grid: np.ndarray, height: int, width: int, stride: int, query_box: tuple[int, int, int, int]) -> float:
    rows, cols = _cells_in_box(height, width, stride, query_box)
    return float(np.median(grid[np.ix_(rows, cols)]))


def build_grids() -> dict:
    if _build_is_complete():
        print("grids already verified", flush=True)
        return json.loads(BUILD_PATH.read_text(encoding="utf-8"))
    model, metadata = load_texture_model(model_path())
    verify_model_metadata(metadata)
    stride = int(metadata["stride"])
    source = hatch_texture.descriptor_source_from_metadata(metadata)
    feature_set = hatch_texture.feature_set_from_metadata(metadata)
    coordinates = {query.id: query for query in development_queries()}
    payload = _resume_payload(metadata, source, feature_set, stride)
    for image, requests in group_requests(development_requests()):
        if _drawing_verified(payload, requests):
            print(f"cached {image}", flush=True)
            continue
        _drop_drawing(payload, requests)
        _process_drawing(payload, model, coordinates, image, requests, stride, source, feature_set)
        _write_build(payload)
        if any(not row["ok"] for row in payload["identity"]):
            _write_identity_failure(payload)
            raise SystemExit(2)
    payload["identity_ok"] = True
    payload["max_abs"] = max(row["max_abs"] for row in payload["identity"])
    _write_build(payload)
    print(f"identity max_abs={payload['max_abs']}", flush=True)
    return payload


def _process_drawing(payload, model, coordinates, image, requests, stride, source, feature_set) -> None:
    started = time.perf_counter()
    gray = load_inference_gray(image, int(requests[0]["height"]), int(requests[0]["width"]))
    print(f"features {image} {gray.shape}", flush=True)
    features = hatch_texture.texture_features(gray, stride=stride, feature_set=feature_set)
    predicted = []
    for request in requests:
        item = _predict_request(model, gray, features, request, stride, source, feature_set)
        _reject_if_identity_fails(payload, coordinates, item, stride)
        predicted.append(item)
    seconds = time.perf_counter() - started
    _store_drawing(payload, coordinates, image, requests, predicted, seconds, stride)
    del features, gray, predicted
    gc.collect()
    print(f"stored {image} sec={seconds:.1f}", flush=True)


def _predict_request(model, gray, features, request, stride, source, feature_set):
    started = time.perf_counter()
    box = tuple(int(value) for value in request["query_box"])
    descriptor = hatch_texture.descriptor_for_query(
        gray, features, box, stride, source, feature_set=feature_set,
    )
    grid = predict_probability_grid(model, features, descriptor)
    elapsed = time.perf_counter() - started
    print(f"  {request['id']} grid={grid.shape} sec={elapsed:.1f}", flush=True)
    return {"id": request["id"], "box": box, "descriptor": descriptor, "grid": grid, "seconds": elapsed}


def _store_drawing(payload, coordinates, image, requests, predicted, seconds, stride) -> None:
    banks = _sample_banks(coordinates, requests, predicted, stride)
    identity = [_identity_row(coordinates, item, banks) for item in predicted]
    payload["identity"].extend(identity)
    payload["drawings"].append({
        "image": image,
        "seconds": seconds,
        "query_ids": [request["id"] for request in requests],
        "query_seconds": {item["id"]: item["seconds"] for item in predicted},
    })
    payload["pairs"].extend(_pair_rows(predicted, requests, stride))
    for item in predicted:
        _save_grid(GRIDS / f"{item['id']}.npz", item["grid"])
        _save_bank(PIXELS / f"{item['id']}.npz", banks[item["id"]])
    failed = [row for row in identity if not row["ok"]]
    if failed:
        payload["identity_ok"] = False


def _sample_banks(coordinates, requests, predicted, stride) -> dict[str, dict[str, np.ndarray]]:
    banks = {}
    for request in requests:
        query = coordinates[request["id"]]
        _require_same_shape(request, query)
        banks[request["id"]] = {
            item["id"]: sample_probability(item["grid"], query.y, query.x, query.height, query.width, stride)
            for item in predicted
        }
    return banks


def _reject_if_identity_fails(payload, coordinates, item, stride) -> None:
    query = coordinates[item["id"]]
    sampled = sample_probability(item["grid"], query.y, query.x, query.height, query.width, stride)
    cached = np.load(HGB_CACHE / f"{item['id']}.npz")["probability"]
    difference = np.abs(sampled.astype(np.float64) - cached.astype(np.float64))
    gap = float(np.max(difference))
    print(f"  identity {item['id']} max_abs={gap:.3e} mean_abs={float(np.mean(difference)):.3e}", flush=True)
    if gap <= IDENTITY_LIMIT:
        return
    payload["identity"].append({
        "id": item["id"],
        "max_abs": gap,
        "mean_abs": float(np.mean(difference)),
        "ok": False,
    })
    _write_identity_failure(payload)
    raise SystemExit(2)


def _identity_row(coordinates, item, banks) -> dict:
    cached = np.load(HGB_CACHE / f"{item['id']}.npz")["probability"]
    sampled = banks[item["id"]][item["id"]]
    if sampled.shape != cached.shape:
        raise ValueError(f"{item['id']}: sampled {sampled.shape} != cache {cached.shape}")
    gap = float(np.max(np.abs(sampled.astype(np.float64) - cached.astype(np.float64))))
    mean_gap = float(np.mean(np.abs(sampled.astype(np.float64) - cached.astype(np.float64))))
    ok = gap <= IDENTITY_LIMIT
    print(f"  identity {item['id']} max_abs={gap:.3e} mean_abs={mean_gap:.3e} ok={ok}", flush=True)
    return {"id": item["id"], "max_abs": gap, "mean_abs": mean_gap, "ok": ok}


def _pair_rows(predicted, requests, stride) -> list[dict]:
    height = int(requests[0]["height"])
    width = int(requests[0]["width"])
    image = requests[0]["image"]
    rows = []
    for left in predicted:
        self_median = median_in_box(left["grid"], height, width, stride, left["box"])
        rows.extend(_pairs_from_left(left, predicted, self_median, height, width, stride, image))
    return rows


def _pairs_from_left(left, predicted, self_median, height, width, stride, image: str) -> list[dict]:
    rows = []
    for right in predicted:
        if right["id"] == left["id"]:
            continue
        rows.append({
            "a": left["id"],
            "b": right["id"],
            "image": image,
            "similarity": descriptor_similarity(left["descriptor"], right["descriptor"]),
            "cross_median": median_in_box(left["grid"], height, width, stride, right["box"]),
            "self_median": self_median,
        })
    return rows


def _cells_in_box(height: int, width: int, stride: int, query_box: tuple[int, int, int, int]):
    centers_y, centers_x = grid_centers(height, width, stride)
    x0, y0, x1, y1 = query_box
    rows = np.flatnonzero((centers_y >= y0) & (centers_y < y1))
    cols = np.flatnonzero((centers_x >= x0) & (centers_x < x1))
    if rows.size == 0 or cols.size == 0:
        return _nearest_cell(centers_y, centers_x, query_box)
    return rows, cols


def _nearest_cell(centers_y: np.ndarray, centers_x: np.ndarray, query_box: tuple[int, int, int, int]):
    cx = (query_box[0] + query_box[2] - 1) / 2.0
    cy = (query_box[1] + query_box[3] - 1) / 2.0
    return np.array([int(np.argmin(np.abs(centers_y - cy)))]), np.array([int(np.argmin(np.abs(centers_x - cx)))])


def _save_grid(path: Path, grid: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temporary, probability=np.ascontiguousarray(grid, dtype=np.float16))
    temporary.replace(path)


def _save_bank(path: Path, bank: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    source_ids = list(bank)
    stacked = np.stack([bank[source_id] for source_id in source_ids]).astype(np.float32)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temporary, source_ids=np.array(source_ids), probabilities=stacked)
    temporary.replace(path)


def _write_identity_failure(payload: dict) -> None:
    failed = [row for row in payload["identity"] if not row["ok"]]
    lines = [
        "identity check FAILED. Competition rules were not run.",
        "Sampling uses inference_texture.native_grid_coordinates and cv2.remap INTER_LINEAR "
        "with BORDER_REPLICATE, then clips to [0, 1], matching interpolate_and_threshold_mask "
        "and evaluate_texture_sweep.remap_known_coordinates.",
        f"model={payload['model']}",
        f"query_descriptor_source={payload['query_descriptor_source']} stride={payload['stride']} "
        f"feature_set={payload['feature_set']}",
    ]
    for row in failed:
        lines.append(
            f"FAIL {row['id']} max_abs={row['max_abs']:.8e} mean_abs={row['mean_abs']:.8e} limit={IDENTITY_LIMIT:.0e}"
        )
    lines.append(
        "The full-grid probabilities, sampled at the reviewed coordinates, do not match "
        "devharness.cached_scores(HGB_CACHE) within 1e-5."
    )
    FINDINGS.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


def _empty_payload(metadata, source, feature_set, stride) -> dict:
    return {
        "stamp": STAMP,
        "model": str(model_path()),
        "requested_model": str(ROOT / "models" / "texture-hgb-crop-v1.joblib"),
        "query_descriptor_source": source,
        "feature_set": feature_set,
        "stride": stride,
        "gray_loader": "cv2.IMREAD_GRAYSCALE",
        "similarity": "1 - mean(abs(qa-qb) / (abs(qa)+abs(qb)+1e-4))",
        "identity_limit": IDENTITY_LIMIT,
        "identity_ok": False,
        "identity": [],
        "drawings": [],
        "pairs": [],
        "training_wall_time_seconds": metadata.get("wall_time_seconds"),
    }


def _write_build(payload: dict) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    temporary = BUILD_PATH.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    temporary.replace(BUILD_PATH)


def _build_is_complete() -> bool:
    payload = _read_build()
    if payload is None or not payload.get("identity_ok"):
        return False
    return all(_query_cached(query.id) for query in development_queries())


def _resume_payload(metadata, source, feature_set, stride) -> dict:
    payload = _read_build()
    if payload is None:
        return _empty_payload(metadata, source, feature_set, stride)
    return payload


def _read_build() -> dict | None:
    if not BUILD_PATH.is_file():
        return None
    payload = json.loads(BUILD_PATH.read_text(encoding="utf-8"))
    if payload.get("stamp") != STAMP:
        return None
    return payload


def _drawing_verified(payload: dict, requests: list[dict]) -> bool:
    ok_ids = {row["id"] for row in payload["identity"] if row["ok"]}
    images = {row["image"] for row in payload["drawings"]}
    image = requests[0]["image"]
    if image not in images:
        return False
    return all(request["id"] in ok_ids and _query_cached(request["id"]) for request in requests)


def _drop_drawing(payload: dict, requests: list[dict]) -> None:
    ids = {request["id"] for request in requests}
    image = requests[0]["image"]
    payload["identity"] = [row for row in payload["identity"] if row["id"] not in ids]
    payload["drawings"] = [row for row in payload["drawings"] if row["image"] != image]
    payload["pairs"] = [row for row in payload["pairs"] if row["a"] not in ids and row["b"] not in ids]
    for query_id in ids:
        (GRIDS / f"{query_id}.npz").unlink(missing_ok=True)
        (PIXELS / f"{query_id}.npz").unlink(missing_ok=True)


def _query_cached(query_id: str) -> bool:
    return (GRIDS / f"{query_id}.npz").is_file() and (PIXELS / f"{query_id}.npz").is_file()


def _area(requests: list[dict]) -> tuple[int, str]:
    first = requests[0]
    return int(first["height"]) * int(first["width"]), first["image"]


def _require_same_shape(request: dict, query) -> None:
    if int(request["height"]) != int(query.height) or int(request["width"]) != int(query.width):
        raise ValueError(f"{request['id']}: request shape does not match the development query")
    if request["image"] != query.image:
        raise ValueError(f"{request['id']}: request image does not match the development query")


if __name__ == "__main__":
    build_grids()
