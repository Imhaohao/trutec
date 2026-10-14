"""Fit rows, then development rows, then synthetic rows."""
from __future__ import annotations

import json

import runtime  # noqa: F401

from folds import indexed_training_order
from runtime import (
    DATASET_ROOT,
    DEVELOPMENT,
    FITS,
    SYNTHETIC_MANIFEST,
    SYNTHETIC_ROOT,
    require,
)

_ROWS: dict[str, list[dict]] = {}
_INDEXES: dict[str, dict[str, int]] = {}


def load_rows(path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["examples"] if isinstance(payload, dict) else payload
    require(isinstance(rows, list) and rows, f"empty manifest {path.name}")
    return rows


def fit_rows() -> list[dict]:
    return _cached("fit", _load_fit)


def development_rows() -> list[dict]:
    return _cached("development", _load_development)


def synthetic_rows() -> list[dict]:
    return _cached("synthetic", _load_synthetic)


def combined_rows() -> list[dict]:
    return _cached("combined", _load_combined)


def data_root_for(row: dict):
    if row["id"] in synthetic_ids():
        return SYNTHETIC_ROOT
    return DATASET_ROOT


def index_of(rows: list[dict]) -> dict[str, int]:
    key = str(id(rows))
    cached = _INDEXES.get(key)
    if cached is not None:
        return cached
    assigned = {row["id"]: index for index, row in indexed_training_order(rows)}
    require(len(assigned) == len(rows), "duplicate query id")
    _INDEXES[key] = assigned
    return assigned


def cell_index_map() -> dict[str, int]:
    """Seed index of each query inside its own manifest."""
    cached = _INDEXES.get("cells")
    if cached is not None:
        return cached
    assigned = _own_manifest_indexes()
    _INDEXES["cells"] = assigned
    return assigned


def fit_ids() -> set[str]:
    return {row["id"] for row in fit_rows()}


def development_ids() -> set[str]:
    return {row["id"] for row in development_rows()}


def synthetic_ids() -> set[str]:
    return {row["id"] for row in synthetic_rows()}


def source_of(row: dict) -> str:
    if row["id"] in fit_ids():
        return "fit"
    if row["id"] in development_ids():
        return "development"
    return "synthetic"


def _cached(name: str, loader):
    if name not in _ROWS:
        _ROWS[name] = loader()
    return _ROWS[name]


def _load_fit() -> list[dict]:
    rows = load_rows(FITS)
    require(len(rows) == 357, f"fit queries {len(rows)}")
    return rows


def _load_development() -> list[dict]:
    rows = load_rows(DEVELOPMENT)
    require(len(rows) == 55, f"development queries {len(rows)}")
    require(all("labels" in row for row in rows), "development row missing labels")
    return rows


def _load_synthetic() -> list[dict]:
    rows = load_rows(SYNTHETIC_MANIFEST)
    require(len(rows) == 842, f"synthetic queries {len(rows)}")
    require(all("labels" in row for row in rows), "synthetic row missing labels")
    documents = {row["document_id"] for row in rows}
    require(len(documents) == 600, f"synthetic documents {len(documents)}")
    _require_one_image_per_document(rows)
    return rows


def _load_combined() -> list[dict]:
    fit = fit_rows()
    development = development_rows()
    synthetic = synthetic_rows()
    _require_separated(fit, development, synthetic)
    rows = fit + development + synthetic
    _require_offset_indexes(fit, development, synthetic, rows)
    _require_one_document_per_image(rows)
    return rows


def _own_manifest_indexes() -> dict[str, int]:
    pieces = (index_of(fit_rows()), index_of(development_rows()), index_of(synthetic_rows()))
    require(_disjoint_ids(pieces), "query id overlap in cell indexes")
    assigned: dict[str, int] = {}
    for piece in pieces:
        assigned.update(piece)
    require(len(assigned) == 357 + 55 + 842, f"cell indexes {len(assigned)}")
    return assigned


def _disjoint_ids(pieces: tuple[dict, ...]) -> bool:
    seen: set[str] = set()
    for piece in pieces:
        if seen & set(piece):
            return False
        seen.update(piece)
    return True


def _require_one_image_per_document(rows: list[dict]) -> None:
    images: dict[str, str] = {}
    for row in rows:
        image = images.setdefault(row["document_id"], row["image"])
        require(image == row["image"], f"document spans images {row['document_id']}")


def _require_separated(fit: list[dict], development: list[dict], synthetic: list[dict]) -> None:
    fit_documents = {row["document_id"] for row in fit}
    development_documents = {row["document_id"] for row in development}
    synthetic_documents = {row["document_id"] for row in synthetic}
    require(len(fit_documents) == 105, f"fit documents {len(fit_documents)}")
    require(len(development_documents) == 12, f"development documents {len(development_documents)}")
    require(fit_documents.isdisjoint(development_documents), "fit and development documents overlap")
    require(fit_documents.isdisjoint(synthetic_documents), "fit and synthetic documents overlap")
    require(development_documents.isdisjoint(synthetic_documents), "development and synthetic documents overlap")
    fit_images = {row["image"] for row in fit}
    development_images = {row["image"] for row in development}
    synthetic_images = {row["image"] for row in synthetic}
    require(fit_images.isdisjoint(development_images), "fit and development images overlap")
    require(fit_images.isdisjoint(synthetic_images), "fit and synthetic images overlap")
    require(development_images.isdisjoint(synthetic_images), "development and synthetic images overlap")
    fit_ids_ = {row["id"] for row in fit}
    development_ids_ = {row["id"] for row in development}
    synthetic_ids_ = {row["id"] for row in synthetic}
    require(fit_ids_.isdisjoint(development_ids_), "fit and development query ids overlap")
    require(fit_ids_.isdisjoint(synthetic_ids_), "fit and synthetic query ids overlap")
    require(development_ids_.isdisjoint(synthetic_ids_), "development and synthetic query ids overlap")


def _require_offset_indexes(fit, development, synthetic, combined) -> None:
    combined_index = index_of(combined)
    _require_offset(index_of(fit), combined_index, 0)
    _require_offset(index_of(development), combined_index, len(fit))
    _require_offset(index_of(synthetic), combined_index, len(fit) + len(development))


def _require_offset(local: dict[str, int], combined: dict[str, int], offset: int) -> None:
    for query_id, index in local.items():
        require(combined[query_id] == offset + index, f"combined index moved {query_id}")


def _require_one_document_per_image(rows: list[dict]) -> None:
    seen: dict[str, str] = {}
    for row in rows:
        document = seen.setdefault(row["image"], row["document_id"])
        require(document == row["document_id"], f"image spans documents {row['image']}")
