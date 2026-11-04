"""Pinecone vector store: client factory and index administration.

The chunk store itself (upsert / hybrid query / delete-by-document) comes with the storage
layer. See docs/layers/02-storage.md.

Index administration has two implementations behind one protocol:

* :class:`CloudIndexAdmin` uses the Pinecone SDK against Pinecone cloud.
* :class:`LocalIndexAdmin` talks REST to the Pinecone Local emulator. Current SDKs speak a
  newer control-plane API version than the emulator understands, so creating/describing
  indexes there uses the older request format. Data-plane calls (upsert, query, stats) go
  through the SDK in both cases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

import httpx
from pinecone import NotFoundError, Pinecone, ServerlessSpec

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
