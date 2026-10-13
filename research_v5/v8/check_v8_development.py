"""Check that native v8 scores, sampled at development reviewed pixels, equal the sparse rule."""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import inference_v8 as v8  # noqa: E402  (imports the combo bootstrap first)

import json  # noqa: E402
from collections import defaultdict  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402

import devharness  # noqa: E402

SIBLING_CACHE = HERE.parent / "sibling7" / "cache"


def sparse_scores(image: str, ids: list[str]) -> dict[str, np.ndarray]:
    stored = np.load(SIBLING_CACHE / (Path(image).stem + ".npz"))
    result = {}
    for target in ids:
        own = stored[f"{target}|at|{target}"]
        rivals = [
            stored[f"{s}|at|{target}"] for s in ids if s != target
            and float(stored[f"{target}|box|{s}"]) < v8.SAME_PATTERN
            and float(stored[f"{s}|box|{target}"]) < v8.SAME_PATTERN
        ]
        result[target] = v8.stolen_scores(own, rivals)
    return result


def main() -> None:
    queries = {q.id: q for q in devharness.development_queries()}
    rows = json.loads(devharness.DEVELOPMENT_MANIFEST.read_text(encoding="utf-8"))["examples"]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["image"]].append(devharness.request_view(row))
    stage1, _metadata = v8.load_original()
    stage2 = joblib.load(v8.STRUCTURE_MODEL)["model"]
    native, sparse, worst = {}, {}, 0.0
    for image, requests in grouped.items():
        scores = v8.final_scores(requests, v8.drawing_maps(requests, stage1, stage2, devharness.DATA_ROOT))
        expected = sparse_scores(image, [r["id"] for r in requests])
        for rid, score in scores.items():
            query = queries[rid]
            native[rid] = score[query.y, query.x]
            sparse[rid] = expected[rid]
            worst = max(worst, float(np.max(np.abs(native[rid] - sparse[rid]))))
        print(f"checked {image} max_abs_gap_so_far={worst}", flush=True)
    a = devharness.report("native", native)["points"]
    b = devharness.report("sparse", sparse)["points"]
    mismatches = [p["threshold"] for p, q in zip(a, b) if any(p[k] != q[k] for k in ("tp", "fp", "fn", "tn"))]
    summary = {"queries": len(native), "max_abs_gap": worst, "threshold_count_mismatches": mismatches,
               "identical": not mismatches}
    (HERE / "v8_native_check.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
