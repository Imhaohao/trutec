"""Keep the four fit folds and add development documents round-robin."""
from __future__ import annotations

import json

import runtime  # noqa: F401

from manifests import combined_rows, development_rows, fit_rows
from runtime import FOLDS_PATH, STACKER_FOLDS, inside, log, require, write_json

FOLD_COUNT = 4


def write_folds() -> dict:
    fit = fit_rows()
    development = development_rows()
    fit_assigned = _fit_assignment()
    development_assigned = _development_assignment(development, fit_assigned)
    assigned = {**fit_assigned, **development_assigned}
    payload = _payload(fit, development, assigned, fit_assigned)
    inside(FOLDS_PATH)
    write_json(FOLDS_PATH, payload)
    _require_saved(fit_assigned, development_assigned)
    log(f"folds documents={len(assigned)} development={len(development_assigned)}")
    return payload


def load_document_folds() -> dict[str, int]:
    payload = json.loads(FOLDS_PATH.read_text(encoding="utf-8"))
    assigned = {str(key): int(value) for key, value in payload["document_fold"].items()}
    require(len(assigned) == 117, f"fold documents {len(assigned)}")
    return assigned


def _fit_assignment() -> dict[str, int]:
    payload = json.loads(STACKER_FOLDS.read_text(encoding="utf-8"))
    assigned = {str(key): int(value) for key, value in payload["document_fold"].items()}
    documents = {row["document_id"] for row in fit_rows()}
    require(set(assigned) == documents, "stacker folds do not match the fit manifest")
    require(set(assigned.values()) == set(range(FOLD_COUNT)), "fit folds are not 0-3")
    return assigned


def _development_assignment(rows: list[dict], fit_assigned: dict[str, int]) -> dict[str, int]:
    documents = sorted({row["document_id"] for row in rows})
    require(len(documents) == 12, f"development documents {len(documents)}")
    require(set(documents).isdisjoint(fit_assigned), "development document is already in a fit fold")
    return {document: index % FOLD_COUNT for index, document in enumerate(documents)}


def _payload(fit, development, assigned, fit_assigned) -> dict:
    rows = fit + development
    require(len(rows) == len(combined_rows()), "combined row count")
    summary = {str(fold): _empty_summary() for fold in range(FOLD_COUNT)}
    _add_documents(summary, assigned, fit_assigned)
    _add_queries(summary, rows, fit_assigned)
    return {
        "fold_count": FOLD_COUNT,
        "rule": (
            "fit folds copied from research_v5/stacker/folds.json; "
            "development documents assigned round-robin by sorted document_id to folds 0-3"
        ),
        "development_rule": "fold = index % 4 over sorted development document_id",
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
        "fit_queries": [],
        "development_queries": [],
        "query_ids": [],
    }


def _add_documents(summary: dict, assigned: dict[str, int], fit_assigned: dict[str, int]) -> None:
    for document, fold in sorted(assigned.items()):
        bucket = summary[str(fold)]
        bucket["documents"].append(document)
        key = "fit_documents" if document in fit_assigned else "development_documents"
        bucket[key].append(document)


def _add_queries(summary: dict, rows: list[dict], fit_assigned: dict[str, int]) -> None:
    for row in rows:
        bucket = summary[str(_fold_of(row, summary, fit_assigned))]
        bucket["query_ids"].append(row["id"])
        key = "fit_queries" if row["document_id"] in fit_assigned else "development_queries"
        bucket[key].append(row["id"])
    for bucket in summary.values():
        bucket["query_ids"] = sorted(bucket["query_ids"])
        bucket["fit_queries"] = sorted(bucket["fit_queries"])
        bucket["development_queries"] = sorted(bucket["development_queries"])
        bucket["queries"] = len(bucket["query_ids"])


def _fold_of(row: dict, summary: dict, fit_assigned: dict[str, int]) -> int:
    document = row["document_id"]
    for fold, bucket in summary.items():
        if document in bucket["documents"]:
            return int(fold)
    raise RuntimeError(f"document missing from folds {document} known_fit={document in fit_assigned}")


def _require_saved(fit_assigned: dict[str, int], development_assigned: dict[str, int]) -> None:
    saved = json.loads(FOLDS_PATH.read_text(encoding="utf-8"))
    for document, fold in fit_assigned.items():
        require(int(saved["document_fold"][document]) == fold, f"fit fold changed {document}")
    ordered = sorted(development_assigned)
    for index, document in enumerate(ordered):
        require(int(saved["document_fold"][document]) == index % FOLD_COUNT, f"round-robin {document}")
        require(len(saved["summary"][str(index % FOLD_COUNT)]["development_documents"]) == 3, "dev per fold")
