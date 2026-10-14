"""Copy fit and development folds, then add synthetic documents round-robin."""
from __future__ import annotations

import json

import runtime  # noqa: F401

from manifests import combined_rows, development_rows, fit_rows, synthetic_rows
from runtime import ALLDATA12_FOLDS, FOLDS_PATH, inside, log, require, write_json

FOLD_COUNT = 4


def write_folds() -> dict:
    saved = _saved_folds()
    fit_assigned, development_assigned = _copied_assignment(saved)
    synthetic_assigned = _synthetic_assignment(fit_assigned, development_assigned)
    assigned = {**fit_assigned, **development_assigned, **synthetic_assigned}
    payload = _payload(assigned, fit_assigned, development_assigned, saved)
    inside(FOLDS_PATH)
    write_json(FOLDS_PATH, payload)
    _require_saved(saved, synthetic_assigned)
    log(f"folds documents={len(assigned)} synthetic={len(synthetic_assigned)}")
    return payload


def load_document_folds() -> dict[str, int]:
    payload = json.loads(FOLDS_PATH.read_text(encoding="utf-8"))
    assigned = {str(key): int(value) for key, value in payload["document_fold"].items()}
    require(len(assigned) == 717, f"fold documents {len(assigned)}")
    return assigned


def _saved_folds() -> dict:
    payload = json.loads(ALLDATA12_FOLDS.read_text(encoding="utf-8"))
    require(len(payload["document_fold"]) == 117, "alldata12 fold count")
    return payload


def _copied_assignment(saved: dict) -> tuple[dict[str, int], dict[str, int]]:
    fit_documents = {row["document_id"] for row in fit_rows()}
    development_documents = {row["document_id"] for row in development_rows()}
    fit_assigned = {}
    development_assigned = {}
    for document, fold in saved["document_fold"].items():
        _store_copied(str(document), int(fold), fit_documents, development_documents, fit_assigned, development_assigned)
    require(set(fit_assigned) == fit_documents, "fit folds were not copied")
    require(set(development_assigned) == development_documents, "development folds were not copied")
    return fit_assigned, development_assigned


def _store_copied(document, fold, fit_documents, development_documents, fit_assigned, development_assigned) -> None:
    require(fold in range(FOLD_COUNT), f"copied fold {document}")
    if document in fit_documents:
        fit_assigned[document] = fold
        return
    require(document in development_documents, f"unknown alldata12 document {document}")
    development_assigned[document] = fold


def _synthetic_assignment(fit_assigned: dict[str, int], development_assigned: dict[str, int]) -> dict[str, int]:
    documents = sorted({row["document_id"] for row in synthetic_rows()})
    require(len(documents) == 600, f"synthetic documents {len(documents)}")
    known = set(fit_assigned) | set(development_assigned)
    require(set(documents).isdisjoint(known), "synthetic document is already folded")
    return {document: index % FOLD_COUNT for index, document in enumerate(documents)}


def _payload(assigned, fit_assigned, development_assigned, saved) -> dict:
    rows = combined_rows()
    summary = {str(fold): _empty_summary() for fold in range(FOLD_COUNT)}
    _add_documents(summary, assigned, fit_assigned, development_assigned)
    _add_queries(summary, rows, fit_assigned, development_assigned)
    return {
        "fold_count": FOLD_COUNT,
        "rule": (
            "fit and development folds copied from research_v5/alldata12/folds.json; "
            "synthetic documents assigned round-robin by sorted document_id to folds 0-3"
        ),
        "development_rule": saved["development_rule"],
        "synthetic_rule": "fold = index % 4 over sorted synthetic document_id",
        "summary": summary,
        "document_fold": assigned,
        "documents": {key: value["documents"] for key, value in summary.items()},
        "query_ids": {key: value["query_ids"] for key, value in summary.items()},
    }


def _empty_summary() -> dict:
    return {
        "documents": [],
        "queries": 0,
        "fit_documents": [],
        "development_documents": [],
        "synthetic_documents": [],
        "fit_queries": [],
        "development_queries": [],
        "synthetic_queries": [],
        "query_ids": [],
    }


def _add_documents(summary, assigned, fit_assigned, development_assigned) -> None:
    for document, fold in sorted(assigned.items()):
        bucket = summary[str(fold)]
        bucket["documents"].append(document)
        bucket[_document_key(document, fit_assigned, development_assigned)].append(document)


def _document_key(document: str, fit_assigned, development_assigned) -> str:
    if document in fit_assigned:
        return "fit_documents"
    if document in development_assigned:
        return "development_documents"
    return "synthetic_documents"


def _add_queries(summary, rows, fit_assigned, development_assigned) -> None:
    lookup = _document_fold_lookup(summary)
    for row in rows:
        bucket = summary[str(lookup[row["document_id"]])]
        bucket["query_ids"].append(row["id"])
        bucket[_query_key(row, fit_assigned, development_assigned)].append(row["id"])
    for bucket in summary.values():
        _sort_bucket(bucket)


def _document_fold_lookup(summary: dict) -> dict[str, int]:
    lookup = {}
    for fold, bucket in summary.items():
        for document in bucket["documents"]:
            lookup[document] = int(fold)
    return lookup


def _query_key(row: dict, fit_assigned, development_assigned) -> str:
    if row["document_id"] in fit_assigned:
        return "fit_queries"
    if row["document_id"] in development_assigned:
        return "development_queries"
    return "synthetic_queries"


def _sort_bucket(bucket: dict) -> None:
    for key in ("query_ids", "fit_queries", "development_queries", "synthetic_queries"):
        bucket[key] = sorted(bucket[key])
    bucket["queries"] = len(bucket["query_ids"])


def _require_saved(saved: dict, synthetic_assigned: dict[str, int]) -> None:
    written = json.loads(FOLDS_PATH.read_text(encoding="utf-8"))
    for document, fold in saved["document_fold"].items():
        require(int(written["document_fold"][document]) == int(fold), f"copied fold changed {document}")
    ordered = sorted(synthetic_assigned)
    for index, document in enumerate(ordered):
        require(int(written["document_fold"][document]) == index % FOLD_COUNT, f"round-robin {document}")
    for fold in range(FOLD_COUNT):
        _require_fold_copy(saved, written, fold)


def _require_fold_copy(saved: dict, written: dict, fold: int) -> None:
    key = str(fold)
    previous = saved["summary"][key]
    current = written["summary"][key]
    require(current["fit_documents"] == previous["fit_documents"], f"fit documents changed in fold {fold}")
    require(current["fit_queries"] == previous["fit_queries"], f"fit queries changed in fold {fold}")
    require(current["development_documents"] == previous["development_documents"], f"development documents changed in fold {fold}")
    require(current["development_queries"] == previous["development_queries"], f"development queries changed in fold {fold}")
    require(len(current["synthetic_documents"]) == 150, f"synthetic documents in fold {fold}")
