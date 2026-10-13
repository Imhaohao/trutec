"""Select a guarded cleaner on the fit split, then score development."""
from __future__ import annotations

import json
import time
from collections import defaultdict

import bootstrap  # noqa: F401

import numpy as np

from bootstrap import (
    CHOSEN_PATH,
    FINDINGS_PATH,
    HERE,
    V6_DEV_SCORES,
    log,
    peak_gb,
)
from collect import collect_dev, collect_fit
from color.color_features import apply_gate, build_image_color_cache, compute_sample_mismatch, extract_robust_query_signature
from metrics import best_at_recall, best_macro_iou, best_precision_at_recall, choose_guard, query_iou, sweep_cached
from report_text import render

import devharness


def main() -> None:
    started = time.perf_counter()
    _run_tests()
    fit_jobs = collect_fit()
    fit_reports = _store_reports("fit", fit_jobs)
    chosen, point = choose_guard(fit_reports)
    _write_choice(chosen, point)
    log(f"selected {chosen} on fit")
    _run_tests()
    dev_jobs = collect_dev()
    gaps = _cache_gaps(dev_jobs)
    dev_reports = _store_reports("dev", dev_jobs)
    diffs = _development_diffs(dev_jobs, chosen)
    runtime = {
        "seconds": time.perf_counter() - started,
        "peak_gb": peak_gb(),
        "fit_queries": len(fit_jobs),
        "dev_queries": len(dev_jobs),
        **gaps,
    }
    FINDINGS_PATH.write_text(render(fit_reports, dev_reports, chosen, diffs, runtime), encoding="utf-8")
    log(f"wrote {FINDINGS_PATH}")
    log(FINDINGS_PATH.read_text(encoding="utf-8"))


def _run_tests() -> None:
    import test_guards

    test_guards.run_all()
    log("unit tests passed")


def _store_reports(split: str, jobs: list[dict]) -> dict:
    from guards import VARIANTS

    reports = {}
    for name in VARIANTS:
        points = sweep_cached(jobs, name)
        payload = {
            "split": split,
            "variant": name,
            "best_macro_iou": best_macro_iou(points),
            "recall_0_95": best_at_recall(points, 0.95),
            "precision_at_recall_0_98": best_precision_at_recall(points, 0.98),
            "points": points,
        }
        path = HERE / f"{split}_report_{name}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        reports[name] = payload
        log(f"{split} {name} macro={_short(payload['best_macro_iou'])} p95={_short(payload['recall_0_95'])}")
    return reports


def _write_choice(chosen: str, point: dict) -> None:
    CHOSEN_PATH.write_text(
        json.dumps(
            {
                "variant": chosen,
                "split": "fit",
                "rule": "max document_macro_iou at global_recall>=0.95 among g1,g2,g3; tie-break global_precision",
                "point": point,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _development_diffs(jobs: list[dict], chosen: str) -> list[dict]:
    rows = []
    for job in jobs:
        v6 = query_iou(job["cache_path"], "v6", 0.14)
        rule_d = query_iou(job["cache_path"], "rule_d", 0.14)
        selected = query_iou(job["cache_path"], chosen, 0.14)
        rows.append(
            {
                "id": job["id"],
                "document_id": job["document_id"],
                "iou_v6": v6,
                "iou_rule_d": rule_d,
                "iou_selected": selected,
                "delta": selected - v6,
            }
        )
    rows.sort(key=lambda item: abs(item["delta"]), reverse=True)
    return rows


def _cache_gaps(jobs: list[dict]) -> dict:
    v6_gap = _v6_gap(jobs)
    rule_gap = _rule_d_gap(jobs)
    log(f"v6 cache gap={v6_gap:.6g} rule_d cache gap={rule_gap:.6g}")
    if v6_gap > 1e-3 or rule_gap > 1e-3:
        raise RuntimeError(f"development cache gap is too large: v6={v6_gap} rule_d={rule_gap}")
    return {"v6_gap": v6_gap, "rule_d_gap": rule_gap}


def _v6_gap(jobs: list[dict]) -> float:
    queries = {query.id: query for query in devharness.development_queries()}
    worst = 0.0
    for image, group in _by_image(jobs).items():
        rgb = devharness.load_rgb(image)
        color_maps = build_image_color_cache(rgb, window_sizes=(16,))[16]
        for job in group:
            query = queries[job["id"]]
            raw = np.load(V6_DEV_SCORES / f"{query.id}.npy")
            signature = extract_robust_query_signature(rgb, query.query_box)
            mismatch = compute_sample_mismatch(signature, color_maps, query.y, query.x)
            gated = apply_gate(raw, mismatch, gate_type="linear", strength=0.6)
            with np.load(job["cache_path"]) as stored:
                mine = np.asarray(stored["v6"], dtype=np.float32)
            worst = max(worst, float(np.max(np.abs(mine - gated))))
        del rgb, color_maps
    return worst


def _rule_d_gap(jobs: list[dict]) -> float:
    cache = HERE.parent / "querycrop" / "cache" / "rule_d_combined"
    worst = 0.0
    for job in jobs:
        theirs = np.load(cache / f"{job['id']}.npy")
        with np.load(job["cache_path"]) as stored:
            mine = np.asarray(stored["rule_d"], dtype=np.float32)
        if mine.shape != theirs.shape:
            raise RuntimeError(f"rule d shape mismatch {job['id']}")
        worst = max(worst, float(np.max(np.abs(mine - theirs))))
    return worst


def _by_image(jobs: list[dict]) -> dict[str, list]:
    grouped = defaultdict(list)
    for job in jobs:
        grouped[job["image"]].append(job)
    return grouped


def _short(point) -> str:
    if point is None:
        return "none"
    return f"{point['document_macro_iou']:.4f}@{point['threshold']:.2f}"


if __name__ == "__main__":
    main()
