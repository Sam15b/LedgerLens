"""Stable identifiers shared across PostgreSQL, Pinecone and Neo4j.

Every document gets one ``document_id`` (UUID4). Its chunks are stored in Pinecone as
``{document_id}#c{n}`` so all vectors of a document can be found -- and deleted -- by prefix.
"""

from __future__ import annotations

import re
import uuid

CHUNK_SEPARATOR = "#c"

_VECTOR_ID = re.compile(
    r"^(?P<doc>[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})#c(?P<n>\d+)$"
)


def new_document_id() -> str:
    """Return a new random document id (lower-case UUID4 string)."""
    return str(uuid.uuid4())


def new_request_id() -> str:
    """Return a short, URL-safe id for correlating one request across services."""
    return uuid.uuid4().hex


def is_document_id(value: str) -> bool:
    try:
        return str(uuid.UUID(value)) == value
    except ValueError:
        return False


def chunk_vector_id(document_id: str, chunk_index: int) -> str:
    """Build the Pinecone vector id for chunk ``chunk_index`` of a document."""
    if not is_document_id(document_id):
        raise ValueError(f"not a canonical document id: {document_id!r}")
    if chunk_index < 0:
        raise ValueError(f"chunk_index must be >= 0, got {chunk_index}")
    return f"{document_id}{CHUNK_SEPARATOR}{chunk_index}"


def vector_id_prefix(document_id: str) -> str:
    """Prefix shared by every vector of a document (for list/delete by prefix)."""
    return f"{document_id}{CHUNK_SEPARATOR}"


def parse_vector_id(vector_id: str) -> tuple[str, int]:
    """Split ``{document_id}#c{n}`` into ``(document_id, n)``."""
    match = _VECTOR_ID.match(vector_id)
    if match is None:
        raise ValueError(f"not a chunk vector id: {vector_id!r}")
    return match["doc"], int(match["n"])
