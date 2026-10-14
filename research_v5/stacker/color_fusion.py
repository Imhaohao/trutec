"""Apply the fixed color gate (w16, linear, strength 0.6) to the frozen stacker development scores."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "research_v5"))
sys.path.insert(0, str(ROOT))

import devharness  # noqa: E402
from color.color_features import (  # noqa: E402
    apply_gate,
    build_image_color_cache,
    compute_sample_mismatch,
    extract_robust_query_signature,
)


def main() -> None:
    queries = devharness.development_queries()
    stacker = {q.id: np.load(HERE / "cache" / "dev_scores" / f"{q.id}.npy") for q in queries}
    grouped = defaultdict(list)
    for query in queries:
        grouped[query.image].append(query)
    gated = {}
    for image, members in grouped.items():
        rgb = devharness.load_rgb(image)
        maps = build_image_color_cache(rgb, window_sizes=(16,))
        for query in members:
            signature = extract_robust_query_signature(rgb, query.query_box)
            mismatch = compute_sample_mismatch(signature, maps[16], query.y, query.x)
            gated[query.id] = apply_gate(stacker[query.id], mismatch, gate_type="linear", strength=0.6)
    result = devharness.report("stacker_times_color_w16_linear_0.6", gated, HERE / "dev_report_color.json")
    print(json.dumps(result["selections"], indent=1))


if __name__ == "__main__":
    main()
