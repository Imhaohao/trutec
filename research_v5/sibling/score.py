"""Score sibling-competition rules on the development split."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research_v5"))

from color.color_features import (  # noqa: E402
    apply_gate,
    build_image_color_cache,
    compute_sample_mismatch,
    extract_robust_query_signature,
)
import devharness  # noqa: E402
from rules import (  # noqa: E402
    assert_unchanged_without_rivals,
    hard_winner,
    soft_margin,
    softmax_share,
)

HERE = Path(__file__).resolve().parent
PIXELS = HERE / "cache" / "pixels"
RESULTS = HERE / "results"
FINDINGS = HERE / "findings.txt"
BATCH_SEED = 20261012
BATCH_TRIALS = 20
CUTOFF_QUANTILES = (("q25", 0.25), ("q50", 0.50), ("q75", 0.75))


def evaluate(payload: dict) -> None:
    _check_subset_helper()
    banks = _load_banks()
    baseline = {query_id: banks[query_id][query_id] for query_id in banks}
    cache_scores = devharness.cached_scores(devharness.HGB_CACHE)
    query_ids = sorted(banks)
    cutoffs = choose_cutoffs(payload["pairs"])
    variants = _run_variants(banks, baseline, query_ids, cutoffs)
    cache_result = devharness.report("r0_hgb_cache", cache_scores, RESULTS / "r0_hgb_cache.json")
    best = _best_competition(variants)
    batch = batch_dependence(banks, best["rivals"], best["apply"])
    fused_scores = color_gate(best["scores"])
    fused = devharness.report(
        "best_times_color_w16_linear_alpha0.6",
        fused_scores,
        RESULTS / "best_times_color_w16_linear_alpha0.6.json",
    )
    _write_side_results(cutoffs, batch, best, fused)
    FINDINGS.write_text(
        _findings(payload, cache_result, variants, best, batch, fused),
        encoding="utf-8",
    )
    print(FINDINGS.read_text(encoding="utf-8"), flush=True)


def choose_cutoffs(pairs: list[dict]) -> list[dict]:
    """Three label-free cutoffs: paired 25/50/75 percentiles. No labels are read."""
    if not pairs:
        return [{"name": "q50", "quantile": 0.5, "sim_max": 0.0, "cross_max": 0.0}]
    similarity = np.array([pair["similarity"] for pair in pairs], dtype=np.float64)
    cross = np.array([pair["cross_median"] for pair in pairs], dtype=np.float64)
    return [_cutoff_at(name, quantile, similarity, cross) for name, quantile in CUTOFF_QUANTILES]


def batch_dependence(banks, rivals, apply) -> dict:
    rng = np.random.default_rng(BATCH_SEED)
    query_ids = sorted(banks)
    trials = [_one_trial(banks, rivals, apply, query_ids, rng, trial) for trial in range(BATCH_TRIALS)]
    return _batch_summary(trials)


def color_gate(scores: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Window-16 linear gate, strength 0.6, the and_fusion_w16_linear_alpha0.6 setting."""
    gated = {}
    grouped: dict[str, list] = {}
    for query in devharness.development_queries():
        grouped.setdefault(query.image, []).append(query)
    for image, queries in grouped.items():
        gated.update(_gate_image(image, queries, scores))
    return gated


def rival_map(query_ids: list[str], pairs: list[dict], sim_max: float, cross_max: float) -> dict[str, list[str]]:
    rivals = {query_id: [] for query_id in query_ids}
    for pair in pairs:
        different = pair["similarity"] < sim_max and pair["cross_median"] < cross_max
        if different:
            rivals[pair["a"]].append(pair["b"])
    return {query_id: sorted(set(names)) for query_id, names in rivals.items()}


def score_queries(banks, rivals, apply) -> dict[str, np.ndarray]:
    scores = {}
    baseline = {}
    for query_id in sorted(banks):
        own = banks[query_id][query_id]
        arrays = [banks[query_id][other] for other in rivals[query_id]]
        scores[query_id] = apply(own, arrays)
        baseline[query_id] = own
    assert_unchanged_without_rivals(scores, baseline, rivals)
    return scores


def summarize_subset(decisions: dict[str, np.ndarray], query_ids: set[str]) -> dict:
    kept = tuple(query for query in devharness.development_queries() if query.id in query_ids)
    if not kept:
        return {"empty": True, "query_count": 0}
    original = devharness.development_queries
    devharness.development_queries = lambda: kept
    try:
        summary = devharness.summarize_decisions(decisions)
    finally:
        devharness.development_queries = original
    summary["query_count"] = len(kept)
    summary["empty"] = False
    return summary


def _run_variants(banks, baseline, query_ids, cutoffs) -> list[dict]:
    none = {query_id: [] for query_id in query_ids}
    r0_apply = lambda own, rivals: own  # noqa: E731
    variants = [_variant("r0_baseline", "r0", None, {}, none, r0_apply, banks, baseline)]
    for setting in cutoffs:
        rivals = rival_map(query_ids, _pairs_of(banks), setting["sim_max"], setting["cross_max"])
        variants.extend(_cutoff_variants(setting, rivals, banks))
    return variants


def _pairs_of(banks) -> list[dict]:
    payload_path = HERE / "cache" / "build.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    known = set(banks)
    return [pair for pair in payload["pairs"] if pair["a"] in known and pair["b"] in known]


def _cutoff_variants(setting, rivals, banks) -> list[dict]:
    specs = (
        ("r1_hard", "r1", None, lambda own, other: hard_winner(own, other)),
        ("r2_soft_t0.05", "r2", 0.05, lambda own, other: soft_margin(own, other, 0.05)),
        ("r2_soft_t0.15", "r2", 0.15, lambda own, other: soft_margin(own, other, 0.15)),
        ("r3_softmax_t0.1", "r3", 0.1, lambda own, other: softmax_share(own, other, 0.1)),
        ("r3_softmax_t0.3", "r3", 0.3, lambda own, other: softmax_share(own, other, 0.3)),
    )
    return [
        _variant(name, kind, temperature, setting, rivals, apply, banks, None)
        for name, kind, temperature, apply in specs
    ]


def _variant(name, kind, temperature, setting, rivals, apply, banks, baseline_scores) -> dict:
    slug = name if kind == "r0" else f"{name}__{setting['name']}"
    scores = baseline_scores if kind == "r0" else score_queries(banks, rivals, apply)
    if kind == "r0":
        assert_unchanged_without_rivals(scores, scores, rivals)
    result = devharness.report(slug, scores, RESULTS / f"{slug}.json")
    with_rival = [query_id for query_id, names in rivals.items() if names]
    macro_threshold = result["selections"]["best_macro_iou"]["threshold"]
    precision_point = result["selections"]["best_precision_at_recall_ge_0_98"]
    print(f"scored {slug} macro={result['selections']['best_macro_iou']['document_macro_iou']}", flush=True)
    return {
        "slug": slug,
        "kind": kind,
        "temperature": temperature,
        "setting": setting,
        "rivals": rivals,
        "apply": apply,
        "scores": scores,
        "edges": sum(len(names) for names in rivals.values()),
        "queries_with_competitor": len(with_rival),
        "result": result,
        "subset_at_best_macro": _subset_point(scores, with_rival, macro_threshold),
        "subset_at_best_precision_recall": _subset_precision_point(scores, with_rival, precision_point),
    }


def _subset_point(scores, query_ids, threshold) -> dict:
    if not query_ids:
        return {"empty": True, "query_count": 0, "threshold": threshold}
    decisions = {query_id: scores[query_id] >= threshold for query_id in query_ids}
    summary = summarize_subset(decisions, set(query_ids))
    summary["threshold"] = float(threshold)
    return _public_summary(summary)


def _subset_precision_point(scores, query_ids, point) -> dict:
    if point is None:
        return {"empty": True, "reason": "no threshold reached recall 0.98 on the full set"}
    return _subset_point(scores, query_ids, point["threshold"])


def _one_trial(banks, rivals, apply, query_ids, rng, trial: int) -> dict:
    kept = {query_id: bool(rng.random() < 0.5) for query_id in query_ids}
    thinned = {query_id: [other for other in rivals[query_id] if kept[other]] for query_id in query_ids}
    scores = score_queries(banks, thinned, apply)
    result = devharness.report(f"batch_{trial}", scores, None)
    macro = result["selections"]["best_macro_iou"]
    precision = result["selections"]["best_precision_at_recall_ge_0_98"]
    return {
        "trial": trial,
        "best_macro_iou": macro["document_macro_iou"],
        "best_macro_threshold": macro["threshold"],
        "precision_at_recall_ge_0_98": None if precision is None else precision["global_precision"],
        "precision_threshold": None if precision is None else precision["threshold"],
    }


def _batch_summary(trials: list[dict]) -> dict:
    ious = np.array([trial["best_macro_iou"] for trial in trials], dtype=np.float64)
    precisions = [trial["precision_at_recall_ge_0_98"] for trial in trials]
    usable = np.array([value for value in precisions if value is not None], dtype=np.float64)
    return {
        "seed": BATCH_SEED,
        "trials": BATCH_TRIALS,
        "keep_probability": 0.5,
        "best_macro_iou_mean": float(ious.mean()),
        "best_macro_iou_min": float(ious.min()),
        "best_macro_iou_max": float(ious.max()),
        "precision_at_recall_ge_0_98_mean": None if usable.size == 0 else float(usable.mean()),
        "precision_at_recall_ge_0_98_min": None if usable.size == 0 else float(usable.min()),
        "precision_at_recall_ge_0_98_max": None if usable.size == 0 else float(usable.max()),
        "precision_missing_trials": int(sum(value is None for value in precisions)),
        "trial_records": trials,
    }


def _gate_image(image: str, queries, scores) -> dict[str, np.ndarray]:
    rgb = devharness.load_rgb(image)
    maps = build_image_color_cache(rgb, window_sizes=(16,))
    gated = {}
    for query in queries:
        signature = extract_robust_query_signature(rgb, query.query_box)
        mismatch = compute_sample_mismatch(signature, maps[16], query.y, query.x)
        gated[query.id] = apply_gate(scores[query.id], mismatch, gate_type="linear", strength=0.6)
    del rgb, maps
    return gated


def _best_competition(variants: list[dict]) -> dict:
    competition = [variant for variant in variants if variant["kind"] != "r0"]
    return max(competition, key=_rank_key)


def _rank_key(variant: dict) -> tuple:
    macro = variant["result"]["selections"]["best_macro_iou"]["document_macro_iou"] or 0.0
    precision = variant["result"]["selections"]["best_precision_at_recall_ge_0_98"]
    precision_value = 0.0 if precision is None else (precision["global_precision"] or 0.0)
    return (macro, precision_value)


def _cutoff_at(name: str, quantile: float, similarity: np.ndarray, cross: np.ndarray) -> dict:
    return {
        "name": name,
        "quantile": quantile,
        "sim_max": float(np.quantile(similarity, quantile)),
        "cross_max": float(np.quantile(cross, quantile)),
        "similarity_min": float(similarity.min()),
        "similarity_max": float(similarity.max()),
        "cross_min": float(cross.min()),
        "cross_max_observed": float(cross.max()),
        "pair_count": int(similarity.size),
    }


def _load_banks() -> dict[str, dict[str, np.ndarray]]:
    banks = {}
    for query in devharness.development_queries():
        banks[query.id] = _load_bank(PIXELS / f"{query.id}.npz")
        if query.id not in banks[query.id]:
            raise ValueError(f"{query.id} is missing its own probability in the pixel cache")
    return banks


def _load_bank(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        source_ids = [str(item) for item in data["source_ids"]]
        probabilities = np.array(data["probabilities"], dtype=np.float32, copy=True)
    return {source_id: probabilities[index] for index, source_id in enumerate(source_ids)}


def _check_subset_helper() -> None:
    queries = devharness.development_queries()
    decisions = {query.id: np.ones(query.positive.shape, dtype=bool) for query in queries}
    full = devharness.summarize_decisions(decisions)
    subset = summarize_subset(decisions, {query.id for query in queries})
    if full["tp"] != subset["tp"] or full["fp"] != subset["fp"]:
        raise AssertionError("summarize_subset does not match summarize_decisions on the full set")


def _write_side_results(cutoffs, batch, best, fused) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "cutoffs.json").write_text(json.dumps(cutoffs, indent=2), encoding="utf-8")
    (RESULTS / "batch_dependence.json").write_text(json.dumps(batch, indent=2), encoding="utf-8")
    (RESULTS / "best.json").write_text(json.dumps(_best_record(best, fused), indent=2), encoding="utf-8")


def _best_record(best, fused) -> dict:
    return {
        "slug": best["slug"],
        "kind": best["kind"],
        "temperature": best["temperature"],
        "setting": best["setting"],
        "edges": best["edges"],
        "queries_with_competitor": best["queries_with_competitor"],
        "selections": best["result"]["selections"],
        "subset_at_best_macro": best["subset_at_best_macro"],
        "subset_at_best_precision_recall": best["subset_at_best_precision_recall"],
        "color_selections": fused["selections"],
    }


def _findings(payload, cache_result, variants, best, batch, fused) -> str:
    lines = [_identity_line(payload), _cache_line(cache_result)]
    lines.extend(_variant_line(variant) for variant in variants)
    lines.append(_batch_line(best, batch))
    lines.append(_color_line(best, fused))
    lines.extend(_runtime_line(row) for row in payload["drawings"])
    lines.append(
        "development-selected best competition rule is the maximum full-set macro IoU "
        "among R1, R2, and R3; ties break toward higher precision at recall >= 0.98."
    )
    lines.append(
        "Competitor cutoffs are the paired 25th, 50th, and 75th percentiles of label-free "
        "descriptor similarity and of the cross median. B competes with A only when both are below the cutoff."
    )
    return "\n".join(lines) + "\n"


def _identity_line(payload: dict) -> str:
    gaps = [row["max_abs"] for row in payload["identity"]]
    first = payload["identity"][:3]
    first_text = ", ".join(f"{row['id']} max_abs={row['max_abs']:.3e}" for row in first)
    return (
        f"identity stamp={payload['stamp']} model={payload['model']} "
        f"gray_loader={payload.get('gray_loader')} "
        f"source={payload['query_descriptor_source']} feature_set={payload['feature_set']} "
        f"checked={len(gaps)} max_abs={max(gaps):.3e} limit=1e-5 first3=[{first_text}]"
    )


def _cache_line(result: dict) -> str:
    return "reference " + _selection_text("r0_hgb_cache", result)


def _variant_line(variant: dict) -> str:
    setting = variant["setting"]
    if variant["kind"] == "r0":
        settings = "score=p_A competitors=none"
    else:
        settings = (
            f"cutoff={setting['name']} sim_max={setting['sim_max']:.6f} cross_max={setting['cross_max']:.6f} "
            f"temperature={variant['temperature']} edges={variant['edges']} "
            f"queries_with_competitor={variant['queries_with_competitor']}"
        )
    subset = _subset_text(variant["subset_at_best_macro"])
    return f"{_selection_text(variant['slug'], variant['result'])} | {settings} | subset_at_best_macro {subset}"


def _selection_text(name: str, result: dict) -> str:
    macro = result["selections"]["best_macro_iou"]
    precision = result["selections"]["best_precision_at_recall_ge_0_98"]
    precision_text = "null" if precision is None else (
        f"precision={_num(precision['global_precision'])} threshold={_num(precision['threshold'])} "
        f"recall={_num(precision['global_recall'])} macro_iou={_num(precision['document_macro_iou'])}"
    )
    return (
        f"{name} best_macro_iou={_num(macro['document_macro_iou'])} threshold={_num(macro['threshold'])} "
        f"recall={_num(macro['global_recall'])} precision={_num(macro['global_precision'])} "
        f"best_precision_at_recall_ge_0.98=({precision_text}) "
        f"all_goals_met={result['selections']['all_goals_met']}"
    )


def _subset_text(summary: dict) -> str:
    if summary.get("empty"):
        return "empty"
    return (
        f"n={summary['query_count']} threshold={_num(summary['threshold'])} "
        f"macro_iou={_num(summary['document_macro_iou'])} recall={_num(summary['global_recall'])} "
        f"precision={_num(summary['global_precision'])}"
    )


def _batch_line(best, batch) -> str:
    return (
        f"batch_dependence rule={best['slug']} seed={batch['seed']} trials={batch['trials']} "
        f"keep_probability={batch['keep_probability']} "
        f"best_macro_iou mean={_num(batch['best_macro_iou_mean'])} "
        f"min={_num(batch['best_macro_iou_min'])} max={_num(batch['best_macro_iou_max'])} "
        f"precision_at_recall_ge_0.98 mean={_num(batch['precision_at_recall_ge_0_98_mean'])} "
        f"min={_num(batch['precision_at_recall_ge_0_98_min'])} max={_num(batch['precision_at_recall_ge_0_98_max'])}"
    )


def _color_line(best, fused) -> str:
    return (
        "color_gate setting=and_fusion_w16_linear_alpha0.6 applied as window=16 linear strength=0.6 "
        f"multiplied onto {best['slug']} | {_selection_text('best_times_color_w16_linear_alpha0.6', fused)}"
    )


def _runtime_line(row: dict) -> str:
    return f"runtime image={row['image']} seconds={row['seconds']:.3f} queries={len(row['query_ids'])}"


def _public_summary(summary: dict) -> dict:
    keys = (
        "document_macro_iou", "global_recall", "global_precision", "reviewed_negative_fpr",
        "tp", "fp", "fn", "tn", "query_count", "threshold", "empty",
    )
    return {key: _plain(summary[key]) for key in keys if key in summary}


def _plain(value):
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    return value


def _num(value) -> str:
    if value is None:
        return "null"
    return f"{float(value):.10g}"
