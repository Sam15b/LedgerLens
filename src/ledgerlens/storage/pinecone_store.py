"""Pinecone vector store: chunk vectors in, hybrid hits out, plus index administration.

:class:`PineconeStore` is the chunk store (docs/layers/02-storage.md):

* ``upsert_chunks`` -- batches of ``upsert_batch_size`` (100); metadata is reduced to what
  queries filter on (``document_id``, ``doc_type``, ``vendor_id``, ``doc_number``,
  ``doc_date`` as a ``YYYYMMDD`` int, ``page``, ``chunk_index``) plus the chunk text cut to
  ``metadata_text_max_bytes`` -- the full text lives in Postgres.
* ``query`` -- hybrid dense + sparse: dense values are scaled by ``alpha`` and sparse values
  by ``1 - alpha`` (``hybrid_alpha``, 0.7), so one weight trades semantic against keyword
  match; metadata filters (``date_filter`` for date ranges) narrow the candidates.

Index administration has two implementations behind one protocol:

* :class:`CloudIndexAdmin` uses the Pinecone SDK against Pinecone cloud.
* :class:`LocalIndexAdmin` talks REST to the Pinecone Local emulator. Current SDKs speak a
  newer control-plane API version than the emulator understands, so creating/describing
  indexes there uses the older request format. Data-plane calls (upsert, query, stats) go
  through the SDK in both cases.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Any, Literal, Protocol

import httpx
from pinecone import NotFoundError, Pinecone, ServerlessSpec

from ledgerlens.core.config import PineconeConfig
from ledgerlens.core.schemas import (
    Contract,
    ExtractedRecord,
    Invoice,
    Payment,
    PurchaseOrder,
    Receipt,
)

if TYPE_CHECKING:
    from ledgerlens.ingestion.embedding import EmbeddedChunk, MetadataValue, SparseVector

# Pinecone Local accepts any API key; this placeholder is used when only a host is set.
LOCAL_API_KEY = "pclocal"
# Control-plane API version understood by the Pinecone Local emulator.
LOCAL_API_VERSION = "2025-10"


class PineconeNotConfiguredError(RuntimeError):
    """Raised when neither PINECONE_API_KEY nor PINECONE_HOST is set."""


class IndexMismatchError(RuntimeError):
    """An index with the requested name exists but has a different shape."""


def create_client(api_key: str, host: str = "", *, timeout_s: float = 30.0) -> Pinecone:
    """Create a client for Pinecone cloud, or for Pinecone Local when ``host`` is set."""
    if not api_key and not host:
        raise PineconeNotConfiguredError("set PINECONE_API_KEY (cloud) or PINECONE_HOST (local)")
    return Pinecone(api_key=api_key or LOCAL_API_KEY, host=host or None, timeout=timeout_s)


@dataclass(frozen=True)
class IndexInfo:
    name: str
    dimension: int | None
    metric: str
    host: str | None
    ready: bool


class IndexAdmin(Protocol):
    def describe(self, name: str) -> IndexInfo | None:
        """Return the index, or ``None`` if it does not exist."""
        ...

    def create(
        self, name: str, *, dimension: int, metric: str, cloud: str, region: str
    ) -> IndexInfo: ...

    def delete(self, name: str) -> None: ...


class CloudIndexAdmin:
    """Index administration on Pinecone cloud through the SDK."""

    def __init__(self, client: Pinecone, *, create_timeout_s: int = 300) -> None:
        self._client = client
        self._create_timeout_s = create_timeout_s

    def describe(self, name: str) -> IndexInfo | None:
        try:
            model = self._client.describe_index(name)
        except NotFoundError:
            return None
        return IndexInfo(
            name=model.name,
            dimension=model.dimension,
            metric=str(model.metric),
            host=model.host,
            ready=bool(model.status.ready),
        )

    def create(
        self, name: str, *, dimension: int, metric: str, cloud: str, region: str
    ) -> IndexInfo:
        self._client.create_index(
            name=name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud=cloud, region=region),
            timeout=self._create_timeout_s,  # wait until the index is ready
        )
        info = self.describe(name)
        if info is None:  # pragma: no cover - create succeeded, so this is a server bug
            raise RuntimeError(f"index {name!r} not found right after creation")
        return info

    def delete(self, name: str) -> None:
        self._client.delete_index(name)


class LocalIndexAdmin:
    """Index administration on the Pinecone Local emulator over REST."""

    def __init__(
        self,
        host: str,
        *,
        timeout_s: float = 10.0,
        transport: httpx.BaseTransport | None = None,  # injectable for tests
    ) -> None:
        self._http = httpx.Client(
            base_url=host.rstrip("/"),
            timeout=timeout_s,
            headers={"Api-Key": LOCAL_API_KEY, "X-Pinecone-Api-Version": LOCAL_API_VERSION},
            transport=transport,
        )

    @staticmethod
    def _to_info(body: dict[str, Any]) -> IndexInfo:
        host = body.get("host")
        if host and "://" not in host:
            host = f"http://{host}"  # the emulator serves plain HTTP
        return IndexInfo(
            name=body["name"],
            dimension=body.get("dimension"),
            metric=body.get("metric", ""),
            host=host,
            ready=bool(body.get("status", {}).get("ready", False)),
        )

    def describe(self, name: str) -> IndexInfo | None:
        response = self._http.get(f"/indexes/{name}")
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return self._to_info(response.json())

    def create(
        self, name: str, *, dimension: int, metric: str, cloud: str, region: str
    ) -> IndexInfo:
        response = self._http.post(
            "/indexes",
            json={
                "name": name,
                "dimension": dimension,
                "metric": metric,
                "spec": {"serverless": {"cloud": cloud, "region": region}},
            },
        )
        response.raise_for_status()
        return self._to_info(response.json())

    def delete(self, name: str) -> None:
        response = self._http.delete(f"/indexes/{name}")
        if response.status_code != 404:
            response.raise_for_status()


def create_index_admin(api_key: str, host: str = "", *, timeout_s: float = 30.0) -> IndexAdmin:
    """Pick the admin implementation: Pinecone Local when ``host`` is set, otherwise cloud."""
    if host:
        return LocalIndexAdmin(host, timeout_s=timeout_s)
    return CloudIndexAdmin(create_client(api_key, timeout_s=timeout_s))


def ensure_index(
    admin: IndexAdmin, name: str, *, dimension: int, metric: str, cloud: str, region: str
) -> tuple[Literal["created", "exists"], IndexInfo]:
    """Create the index if it is missing. Idempotent.

    Raises :class:`IndexMismatchError` if an index with this name already exists with a
    different dimension or metric -- recreating it would silently drop every vector.
    """
    existing = admin.describe(name)
    if existing is None:
        return "created", admin.create(
            name, dimension=dimension, metric=metric, cloud=cloud, region=region
        )
    if existing.dimension != dimension or existing.metric != metric:
        raise IndexMismatchError(
            f"index {name!r} exists with dimension={existing.dimension}, "
            f"metric={existing.metric!r}; expected dimension={dimension}, metric={metric!r}"
        )
    return "exists", existing


# --------------------------------------------------------------------------- chunk store


@dataclass(frozen=True)
class DocumentKeys:
    """Per-document values copied into every chunk's metadata, for filtering."""

    vendor_id: str | None = None
    doc_number: str | None = None
    doc_date: date | None = None

    @classmethod
    def from_record(
        cls, record: ExtractedRecord | None, *, vendor_id: str | None = None
    ) -> DocumentKeys:
        """The number and date that identify ``record`` (invoice number + invoice date, PO
        number + order date, payment reference + paid date, ...)."""
        number: str | None
        day: date | None
        if isinstance(record, Invoice):
            number, day = record.invoice_number, record.invoice_date
        elif isinstance(record, Receipt):
            number, day = record.receipt_number, record.receipt_date
        elif isinstance(record, PurchaseOrder):
            number, day = record.po_number, record.order_date
        elif isinstance(record, Payment):
            number, day = record.payment_reference, record.paid_at
        elif isinstance(record, Contract):
            number, day = record.contract_number, record.start_date
        else:
            number, day = None, None
        return cls(vendor_id=vendor_id, doc_number=number, doc_date=day)


NO_KEYS = DocumentKeys()


def date_key(day: date) -> int:
    """``YYYYMMDD`` as an int, so ``$gte`` / ``$lte`` filters work on dates."""
    return day.year * 10_000 + day.month * 100 + day.day


def truncate_utf8(text: str, max_bytes: int) -> str:
    """At most ``max_bytes`` of UTF-8, never cutting a character in half."""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


_COPIED = ("document_id", "doc_type", "kind", "chunk_index", "title")


def vector_metadata(
    chunk_metadata: Mapping[str, MetadataValue],
    keys: DocumentKeys | None = None,
    *,
    max_text_bytes: int,
) -> dict[str, MetadataValue]:
    """The metadata stored with a chunk vector (Pinecone rejects nulls, so unset values
    are left out)."""
    metadata: dict[str, MetadataValue] = {
        k: chunk_metadata[k] for k in _COPIED if chunk_metadata.get(k) is not None
    }
    keys = keys or NO_KEYS
    pages = chunk_metadata.get("pages")
    if isinstance(pages, list) and pages:
        metadata["page"] = int(pages[0])
    if keys.vendor_id:
        metadata["vendor_id"] = keys.vendor_id
    if keys.doc_number:
        metadata["doc_number"] = keys.doc_number
    if keys.doc_date is not None:
        metadata["doc_date"] = date_key(keys.doc_date)
    text = chunk_metadata.get("text")
    if isinstance(text, str):
        metadata["text"] = truncate_utf8(text, max_text_bytes)
    return metadata


class VectorIndex(Protocol):
    """The data-plane calls the store makes (``pinecone.Index`` or a test fake)."""

    def upsert(self, *, vectors: list[dict[str, Any]], namespace: str) -> Any: ...

    def query(self, **kwargs: Any) -> Any: ...

    def delete(self, *, ids: Sequence[str], namespace: str) -> Any: ...

    def list(self, *, prefix: str, namespace: str) -> Iterator[Any]: ...

    def describe_index_stats(self) -> Any: ...


@dataclass(frozen=True)
class VectorHit:
    """One query match: the chunk's vector id, its hybrid score and stored metadata."""

    id: str
    score: float
    metadata: dict[str, Any]

    @property
    def document_id(self) -> str:
        return str(self.metadata.get("document_id") or self.id.split("#", 1)[0])

    @property
    def text(self) -> str:
        """The truncated chunk text kept in metadata (full text: Postgres ``chunks``)."""
        return str(self.metadata.get("text", ""))


def hybrid_scale(
    dense: Sequence[float], sparse: SparseVector | None, alpha: float
) -> tuple[list[float], dict[str, list[Any]] | None]:
    """Weight dense by ``alpha`` and sparse by ``1 - alpha`` (convex combination).

    Without sparse terms the dense vector is returned unscaled -- scaling it alone would
    change scores but not the ranking. ``alpha == 1`` drops the sparse part.
    """
    if not 0 <= alpha <= 1:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if sparse is None or not sparse.indices or alpha == 1:
        return list(dense), None
    return [v * alpha for v in dense], {
        "indices": list(sparse.indices),
        "values": [v * (1 - alpha) for v in sparse.values],
    }


def date_filter(start: date | None = None, end: date | None = None) -> dict[str, Any]:
    """A metadata filter on ``doc_date`` between ``start`` and ``end`` (inclusive)."""
    bounds: dict[str, int] = {}
    if start is not None:
        bounds["$gte"] = date_key(start)
    if end is not None:
        bounds["$lte"] = date_key(end)
    return {"doc_date": bounds} if bounds else {}


def _matches(response: Any) -> list[Any]:
    matches = getattr(response, "matches", None)
    if matches is None and isinstance(response, Mapping):
        matches = response.get("matches")
    return list(matches or [])


def batched(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


class PineconeStore:
    """Chunk vectors in one index, isolated by namespace (``dev`` / ``eval`` / ``prod``)."""

    def __init__(
        self, index: VectorIndex, *, namespace: str, config: PineconeConfig | None = None
    ) -> None:
        self.index = index
        self.config = config or PineconeConfig()
        self.namespace = namespace

    def upsert_chunks(
        self,
        chunks: Sequence[EmbeddedChunk],
        *,
        keys: DocumentKeys | None = None,
        namespace: str | None = None,
    ) -> int:
        """Upsert ``chunks`` in batches; ``keys`` adds vendor, number and date to each
        chunk's metadata. Returns how many vectors were written."""
        records = [self._record(chunk, keys) for chunk in chunks]
        target = namespace or self.namespace
        for batch in batched(records, self.config.upsert_batch_size):
            self.index.upsert(vectors=list(batch), namespace=target)
        return len(records)

    def _record(self, chunk: EmbeddedChunk, keys: DocumentKeys | None) -> dict[str, Any]:
        if len(chunk.values) != self.config.dimension:
            raise ValueError(
                f"{chunk.id}: {len(chunk.values)}-dim vector, index is {self.config.dimension}"
            )
        record = chunk.to_pinecone()
        record["metadata"] = vector_metadata(
            chunk.metadata, keys, max_text_bytes=self.config.metadata_text_max_bytes
        )
        return record

    def query(
        self,
        dense: Sequence[float],
        sparse: SparseVector | None = None,
        *,
        top_k: int | None = None,
        filter: Mapping[str, Any] | None = None,
        namespace: str | None = None,
        alpha: float | None = None,
    ) -> list[VectorHit]:
        """Hybrid nearest neighbours of a query, best first.

        ``alpha`` defaults to ``hybrid_alpha``; ``top_k`` to ``query_top_k``. ``filter`` is a
        Pinecone metadata filter, e.g. ``{"doc_type": "invoice", **date_filter(start)}``.
        """
        if len(dense) != self.config.dimension:
            raise ValueError(f"query has {len(dense)} dims, index is {self.config.dimension}")
        weight = self.config.hybrid_alpha if alpha is None else alpha
        values, sparse_values = hybrid_scale(dense, sparse, weight)
        kwargs: dict[str, Any] = {
            "vector": values,
            "top_k": top_k or self.config.query_top_k,
            "namespace": namespace or self.namespace,
            "include_metadata": True,
        }
        if sparse_values is not None:
            kwargs["sparse_vector"] = sparse_values
        if filter:
            kwargs["filter"] = dict(filter)
        return [
            VectorHit(id=m.id, score=float(m.score), metadata=dict(m.metadata or {}))
            for m in _matches(self.index.query(**kwargs))
        ]
