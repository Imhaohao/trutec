"""Document-grouped folds and fit-manifest order."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from constants import FOLD_COUNT, require


def group_by_image(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["image"]].append(row)
    return grouped


def indexed_training_order(rows: list[dict]) -> list[tuple[int, dict]]:
    """Match train_texture.collect_training_dataset's query counter."""
    ordered: list[tuple[int, dict]] = []
    counter = 0
    for group in group_by_image(rows).values():
        for row in group:
            ordered.append((counter, row))
            counter += 1
    return ordered


def assign_folds(rows: list[dict]) -> dict[str, int]:
    counts = _query_counts(rows)
    ordered = sorted(counts, key=lambda doc: (-counts[doc], doc))
    load = [0] * FOLD_COUNT
    assigned: dict[str, int] = {}
    for document in ordered:
        fold = _lightest(load)
        assigned[document] = fold
        load[fold] += counts[document]
    return assigned


def fold_payload(rows: list[dict], assigned: dict[str, int]) -> dict:
    counts = _query_counts(rows)
    folds = {str(fold): [] for fold in range(FOLD_COUNT)}
    queries = {str(fold): [] for fold in range(FOLD_COUNT)}
    for row in rows:
        fold = assigned[row["document_id"]]
        queries[str(fold)].append(row["id"])
    for document, fold in sorted(assigned.items()):
        folds[str(fold)].append(document)
    summary = {
        str(fold): {
            "documents": len(folds[str(fold)]),
            "queries": len(queries[str(fold)]),
        }
        for fold in range(FOLD_COUNT)
    }
    return {
        "fold_count": FOLD_COUNT,
        "rule": "greedy lightest-fold by query count; document_id breaks ties",
        "summary": summary,
        "document_fold": assigned,
        "query_counts": counts,
        "documents": folds,
        "query_ids": {key: sorted(value) for key, value in queries.items()},
    }


def save_folds(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_document_folds(path: Path) -> dict[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assigned = {str(key): int(value) for key, value in payload["document_fold"].items()}
    require(assigned, "fold assignment is empty")
    return assigned


def _query_counts(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row["document_id"]] += 1
    return dict(counts)


def _lightest(load: list[int]) -> int:
    return min(range(len(load)), key=lambda index: (load[index], index))
