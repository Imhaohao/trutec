"""v8 = frozen v7 scores plus guarded sibling stealing (batch-dependent).

Requests are grouped by drawing path. For query A, a sibling B on the same drawing is a
rival unless the two look like the same pattern (A's median score inside B's query box,
or B's median score inside A's box, is at least SAME_PATTERN). At a pixel, a rival takes
the pixel when its score is at least STEAL_LEVEL and exceeds A's score by STEAL_MARGIN;
A's score there is multiplied by STEAL_FACTOR. Only label-free request fields are read.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "combo"))
import bootstrap  # noqa: E402,F401

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from collections import defaultdict  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from bootstrap import STRUCTURE_MODEL  # noqa: E402
from evaluate_dev import load_original  # noqa: E402
from pipeline import cleaner_by_name, visit_maps  # noqa: E402

CLEANER = "g2"
SAME_PATTERN = 0.5
STEAL_LEVEL = 0.8
STEAL_MARGIN = 0.1
STEAL_FACTOR = 0.2
REQUEST_FIELDS = ("id", "image", "width", "height", "query_box", "context_boxes")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--threshold", required=True, type=float)
    return parser.parse_args()


def load_requests(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["examples"] if isinstance(payload, dict) else payload
    return [{key: row[key] for key in REQUEST_FIELDS if key in row} for row in rows]


def drawing_maps(requests: list[dict], stage1, stage2, data_root: Path) -> dict[str, np.ndarray]:
    maps: dict[str, np.ndarray] = {}

    def keep(request: dict, gated: np.ndarray) -> None:
        maps[request["id"]] = gated.astype(np.float32, copy=False)

    visit_maps(requests, stage1, stage2, cleaner_by_name(CLEANER), data_root, keep)
    return maps


def box_median(score: np.ndarray, box) -> float:
    x0, y0, x1, y1 = (int(v) for v in box)
    return float(np.median(score[y0:y1, x0:x1]))


def rivals_of(target: dict, requests: list[dict], maps: dict[str, np.ndarray]) -> list[str]:
    rivals = []
    for other in requests:
        if other["id"] == target["id"]:
            continue
        mine_in_theirs = box_median(maps[target["id"]], other["query_box"])
        theirs_in_mine = box_median(maps[other["id"]], target["query_box"])
        if mine_in_theirs < SAME_PATTERN and theirs_in_mine < SAME_PATTERN:
            rivals.append(other["id"])
    return rivals


def stolen_scores(own: np.ndarray, rival_maps: list[np.ndarray]) -> np.ndarray:
    if not rival_maps:
        return own
    strongest = rival_maps[0].copy()
    for rival in rival_maps[1:]:
        np.maximum(strongest, rival, out=strongest)
    steal = (strongest >= STEAL_LEVEL) & (strongest > own + STEAL_MARGIN)
    result = own.copy()
    result[steal] *= STEAL_FACTOR
    return result


def final_scores(requests: list[dict], maps: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    rivals = {r["id"]: rivals_of(r, requests, maps) for r in requests}
    return {rid: stolen_scores(maps[rid], [maps[s] for s in names]) for rid, names in rivals.items()}


def write_mask(path: Path, selected: np.ndarray) -> None:
    Image.fromarray(np.where(selected, 255, 0).astype(np.uint8), mode="L").save(path)


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage1, _metadata = load_original()
    stage2 = joblib.load(STRUCTURE_MODEL)["model"]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for request in load_requests(args.inputs):
        grouped[request["image"]].append(request)
    runtime = {"threshold": args.threshold, "drawings": []}
    for image, requests in grouped.items():
        drawing_started = time.perf_counter()
        scores = final_scores(requests, drawing_maps(requests, stage1, stage2, args.data_root))
        for rid, score in scores.items():
            write_mask(args.output_dir / f"{rid}.png", score >= args.threshold)
        runtime["drawings"].append({"image": image, "queries": len(requests),
                                    "seconds": time.perf_counter() - drawing_started})
        print(f"wrote {len(requests)} masks for {image}", flush=True)
        del scores
    runtime["elapsed_seconds"] = time.perf_counter() - started
    (args.output_dir / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
