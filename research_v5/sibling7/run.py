"""Sibling competition on top of the frozen v7 scores (development split, research only).

Every development drawing is scored once with the v7 pipeline. Each query's map is
sampled at the reviewed coordinates of every query on the same drawing, so a query's
siblings can compete for its pixels. Requests are grouped by drawing path only.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
sys.path.insert(0, str(RESEARCH / "combo"))
import bootstrap  # noqa: E402,F401  (combo bootstrap: thread limits and import paths)

import json  # noqa: E402
from collections import defaultdict  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402

import devharness  # noqa: E402
from bootstrap import STRUCTURE_MODEL  # noqa: E402
from evaluate_dev import load_original  # noqa: E402
from pipeline import cleaner_by_name, visit_maps  # noqa: E402

sys.path.insert(0, str(RESEARCH / "sibling"))
from rules import hard_winner, soft_margin, softmax_share  # noqa: E402

CACHE = HERE / "cache"
SAME_PATTERN_LEVEL = 0.5
BATCH_TRIALS = 20
BATCH_SEED = 20261013


def collect() -> dict:
    """Return {query_id: {"own": scores, "siblings": {sibling_id: scores}, "box_median": {...}}}."""
    CACHE.mkdir(exist_ok=True)
    rows = json.loads(devharness.DEVELOPMENT_MANIFEST.read_text(encoding="utf-8"))["examples"]
    queries = {query.id: query for query in devharness.development_queries()}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["image"]].append(devharness.request_view(row))
    stage1, _metadata = load_original()
    stage2 = joblib.load(STRUCTURE_MODEL)["model"]
    cleaner = cleaner_by_name("g2")
    for image, requests in grouped.items():
        path = CACHE / (Path(image).stem + ".npz")
        if not path.is_file():
            _collect_drawing(requests, queries, stage1, stage2, cleaner, path)
    return _load_all(grouped)


def _collect_drawing(requests, queries, stage1, stage2, cleaner, path: Path) -> None:
    sampled: dict[str, np.ndarray] = {}

    def visit(request: dict, gated: np.ndarray) -> None:
        source = request["id"]
        for target in requests:
            query = queries[target["id"]]
            sampled[f"{source}|at|{target['id']}"] = gated[query.y, query.x].astype(np.float32)
            x0, y0, x1, y1 = target["query_box"]
            sampled[f"{source}|box|{target['id']}"] = np.float32(np.median(gated[y0:y1, x0:x1]))

    visit_maps(requests, stage1, stage2, cleaner, devharness.DATA_ROOT, visit)
    np.savez_compressed(path, **sampled)
    print(f"collected {path.name} queries={len(requests)}", flush=True)


def _load_all(grouped) -> dict:
    data = {}
    for image, requests in grouped.items():
        stored = np.load(CACHE / (Path(image).stem + ".npz"))
        ids = [request["id"] for request in requests]
        for target in ids:
            data[target] = {
                "own": stored[f"{target}|at|{target}"],
                "siblings": {s: stored[f"{s}|at|{target}"] for s in ids if s != target},
                "box": {s: float(stored[f"{s}|box|{target}"]) for s in ids if s != target},
                "own_box_in": {s: float(stored[f"{target}|box|{s}"]) for s in ids if s != target},
            }
    return data


def competitors(entry: dict, mode: str) -> list[str]:
    """mode 'all': every sibling competes; 'distinct': drop siblings that mutually match."""
    if mode == "all":
        return list(entry["siblings"])
    return [
        s for s in entry["siblings"]
        if not (entry["box"][s] >= SAME_PATTERN_LEVEL and entry["own_box_in"][s] >= SAME_PATTERN_LEVEL)
    ]


RULES = {
    "r1_hard": lambda own, rivals: hard_winner(own, rivals),
    "r2_soft_t0.05": lambda own, rivals: soft_margin(own, rivals, 0.05),
    "r2_soft_t0.15": lambda own, rivals: soft_margin(own, rivals, 0.15),
    "r3_share_t0.1": lambda own, rivals: softmax_share(own, rivals, 0.1),
    "r3_share_t0.3": lambda own, rivals: softmax_share(own, rivals, 0.3),
}


def apply(data: dict, rule, mode: str, keep=None) -> dict[str, np.ndarray]:
    scores = {}
    for target, entry in data.items():
        names = competitors(entry, mode)
        if keep is not None:
            names = [s for s in names if keep(target, s)]
        scores[target] = rule(entry["own"], [entry["siblings"][s] for s in names])
    return scores


def summarize(result: dict) -> dict:
    points = result["points"]
    eligible = [p for p in points if (p["global_recall"] or 0) >= 0.95]
    at95 = max(eligible, key=lambda p: (p["document_macro_iou"], -p["fp"])) if eligible else None
    pick = lambda p: None if p is None else {k: p[k] for k in (  # noqa: E731
        "threshold", "document_macro_iou", "global_recall", "global_precision", "reviewed_negative_fpr")}
    selections = result["selections"]
    return {
        "best_macro": pick(selections["best_macro_iou"]),
        "recall_ge_0_95": pick(at95),
        "precision_at_recall_ge_0_98": pick(selections["best_precision_at_recall_ge_0_98"]),
    }


def batch_dependence(data: dict, rule, mode: str) -> dict:
    rng = np.random.default_rng(BATCH_SEED)
    macros, precisions = [], []
    for _trial in range(BATCH_TRIALS):
        kept = {(t, s): bool(rng.random() < 0.5) for t, e in data.items() for s in e["siblings"]}
        summary = summarize(devharness.report("trial", apply(data, rule, mode, lambda t, s: kept[(t, s)])))
        macros.append(summary["recall_ge_0_95"]["document_macro_iou"])
        precisions.append(summary["recall_ge_0_95"]["global_precision"])
    return {"trials": BATCH_TRIALS, "keep_probability": 0.5,
            "macro_at_recall_0_95": [min(macros), float(np.mean(macros)), max(macros)],
            "precision_at_recall_0_95": [min(precisions), float(np.mean(precisions)), max(precisions)]}


def main() -> None:
    data = collect()
    baseline = {t: e["own"] for t, e in data.items()}
    results = {"v7_no_siblings": summarize(devharness.report("v7", baseline))}
    for mode in ("all", "distinct"):
        for name, rule in RULES.items():
            results[f"{name}__{mode}"] = summarize(devharness.report(name, apply(data, rule, mode)))
    best = max((k for k in results if k != "v7_no_siblings"),
               key=lambda k: results[k]["recall_ge_0_95"]["document_macro_iou"])
    rule_name, mode = best.split("__")
    results["best_by_development"] = best
    results["batch_dependence"] = batch_dependence(data, RULES[rule_name], mode)
    (HERE / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
