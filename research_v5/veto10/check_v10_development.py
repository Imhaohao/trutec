"""Native v10 scores sampled at development reviewed pixels must equal the sparse veto."""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import inference_v10 as v10  # noqa: E402

import json  # noqa: E402
from collections import defaultdict  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402

import devharness  # noqa: E402

SIBLING_CACHE = HERE.parent / "sibling7" / "cache"


def main() -> None:
    queries = {q.id: q for q in devharness.development_queries()}
    rows = json.loads(devharness.DEVELOPMENT_MANIFEST.read_text(encoding="utf-8"))["examples"]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["image"]].append(devharness.request_view(row))
    stage1, _metadata = v10.load_original()
    stage2 = joblib.load(v10.STRUCTURE_MODEL)["model"]
    native, sparse = {}, {}
    for image, requests in grouped.items():
        stored = np.load(SIBLING_CACHE / (Path(image).stem + ".npz"))
        rgb = devharness.load_rgb(image)
        maps = v10.build_image_color_cache(rgb, window_sizes=(v10.COLOR_WINDOW,))[v10.COLOR_WINDOW]

        def emit(request: dict, score: np.ndarray, stored=stored, rgb=rgb, maps=maps) -> None:
            query = queries[request["id"]]
            native[query.id] = score[query.y, query.x]
            signature = v10.extract_robust_query_signature(rgb, tuple(int(v) for v in request["query_box"]))
            mismatch = v10.compute_sample_mismatch(signature, maps, query.y, query.x)
            own = stored[f"{query.id}|at|{query.id}"]
            sparse[query.id] = np.where(mismatch > v10.VETO_LEVEL, 0.0, own).astype(np.float32)

        v10.predict_drawing(requests, stage1, stage2, devharness.DATA_ROOT, emit)
        print(f"checked {image}", flush=True)
    gap = max(float(np.max(np.abs(native[k] - sparse[k]))) for k in native)
    a = devharness.report("native", native)["points"]
    b = devharness.report("sparse", sparse)["points"]
    mismatches = [p["threshold"] for p, q in zip(a, b) if any(p[k] != q[k] for k in ("tp", "fp", "fn", "tn"))]
    summary = {"queries": len(native), "max_abs_gap": gap, "threshold_count_mismatches": mismatches,
               "identical": not mismatches}
    (HERE / "v10_native_check.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (HERE / "v10_dev_points.json").write_text(json.dumps(a, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
