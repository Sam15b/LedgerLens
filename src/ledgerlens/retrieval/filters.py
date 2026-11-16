"""Turn :class:`~ledgerlens.retrieval.QueryFilters` into a Pinecone metadata filter.

Chunk vectors carry ``doc_type``, ``vendor_id``, ``doc_number`` and ``doc_date`` (a
``YYYYMMDD`` int, so ranges compare as numbers) -- see
:func:`ledgerlens.storage.pinecone_store.vector_metadata`. One value becomes ``$eq``,
several become ``$in``; the date range becomes ``$gte`` / ``$lte`` on ``doc_date``.
Different fields are separate top-level keys, which Pinecone ANDs together::

    QueryFilters(doc_types=("invoice",), vendor_ids=("v_acme",), date_from=date(2026, 1, 1))
    -> {"doc_type": {"$eq": "invoice"}, "vendor_id": {"$eq": "v_acme"},
        "doc_date": {"$gte": 20260101}}

Documents without a date are excluded by a date range (they have no ``doc_date``).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ledgerlens.retrieval import QueryFilters
from ledgerlens.storage.pinecone_store import date_filter


def _one_of(values: Sequence[str]) -> dict[str, Any]:
    unique = list(dict.fromkeys(values))
    return {"$eq": unique[0]} if len(unique) == 1 else {"$in": unique}


def to_pinecone_filter(filters: QueryFilters | None) -> dict[str, Any]:
    """The Pinecone metadata filter for ``filters`` (``{}`` when there is nothing to filter)."""
    if filters is None:
        return {}
    out: dict[str, Any] = {}
    if filters.doc_types:
        out["doc_type"] = _one_of([doc_type.value for doc_type in filters.doc_types])
    if filters.vendor_ids:
        out["vendor_id"] = _one_of(filters.vendor_ids)
    if filters.doc_numbers:
        out["doc_number"] = _one_of(filters.doc_numbers)
    out.update(date_filter(filters.date_from, filters.date_to))
    return out
