"""Fit rows followed by development rows, in manifest order."""
from __future__ import annotations

import json

import runtime  # noqa: F401

from folds import indexed_training_order
from runtime import DEVELOPMENT, FITS, require


def load_rows(path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["examples"] if isinstance(payload, dict) else payload
    require(isinstance(rows, list) and rows, f"empty manifest {path.name}")
    return rows


def fit_rows() -> list[dict]:
    rows = load_rows(FITS)
    require(len(rows) == 357, f"fit queries {len(rows)}")
    return rows


def development_rows() -> list[dict]:
    rows = load_rows(DEVELOPMENT)
    require(len(rows) == 55, f"development queries {len(rows)}")
    require(all("labels" in row for row in rows), "development row missing labels")
    return rows


def combined_rows() -> list[dict]:
    fit = fit_rows()
    development = development_rows()
    _require_separated(fit, development)
    rows = fit + development
    _require_stable_fit_index(fit, rows)
    _require_one_document_per_image(rows)
    return rows


def index_of(rows: list[dict]) -> dict[str, int]:
    assigned = {row["id"]: index for index, row in indexed_training_order(rows)}
    require(len(assigned) == len(rows), "duplicate query id")
    return assigned


def fit_ids() -> set[str]:
    return {row["id"] for row in fit_rows()}


def _require_separated(fit: list[dict], development: list[dict]) -> None:
    fit_documents = {row["document_id"] for row in fit}
    development_documents = {row["document_id"] for row in development}
    require(len(fit_documents) == 105, f"fit documents {len(fit_documents)}")
    require(len(development_documents) == 12, f"development documents {len(development_documents)}")
    require(fit_documents.isdisjoint(development_documents), "document overlap")
    fit_images = {row["image"] for row in fit}
    development_images = {row["image"] for row in development}
    require(fit_images.isdisjoint(development_images), "image overlap")


def _require_stable_fit_index(fit: list[dict], combined: list[dict]) -> None:
    fit_index = index_of(fit)
    combined_index = index_of(combined)
    for query_id, index in fit_index.items():
        require(combined_index[query_id] == index, f"fit index moved {query_id}")
    development_indexes = [combined_index[row["id"]] for row in combined if row["id"] not in fit_index]
    require(min(development_indexes) == len(fit), "development index does not follow fit")


def _require_one_document_per_image(rows: list[dict]) -> None:
    seen: dict[str, str] = {}
    for row in rows:
        document = seen.setdefault(row["image"], row["document_id"])
        require(document == row["document_id"], f"image spans documents {row['image']}")
