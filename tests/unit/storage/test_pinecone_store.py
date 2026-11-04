import json

import httpx
import pytest

from ledgerlens.storage.pinecone_store import (
    LOCAL_API_VERSION,
    CloudIndexAdmin,
    IndexInfo,
    IndexMismatchError,
    LocalIndexAdmin,
    PineconeNotConfiguredError,
    create_client,
    create_index_admin,
    ensure_index,
)


class FakeAdmin:
    def __init__(self, existing: IndexInfo | None = None) -> None:
        self.indexes = {existing.name: existing} if existing else {}
        self.created: list[str] = []

    def describe(self, name: str) -> IndexInfo | None:
        return self.indexes.get(name)

    def create(
        self, name: str, *, dimension: int, metric: str, cloud: str, region: str
    ) -> IndexInfo:
        info = IndexInfo(name, dimension, metric, f"http://{name}", ready=True)
        self.indexes[name] = info
        self.created.append(name)
        return info

    def delete(self, name: str) -> None:
        self.indexes.pop(name, None)


def _ensure(admin: FakeAdmin, name: str = "chunks") -> tuple[str, IndexInfo]:
    return ensure_index(
        admin, name, dimension=768, metric="dotproduct", cloud="aws", region="us-east-1"
    )


def test_ensure_index_creates_missing_index() -> None:
    admin = FakeAdmin()

    outcome, info = _ensure(admin)

    assert outcome == "created"
    assert (info.dimension, info.metric) == (768, "dotproduct")
    assert admin.created == ["chunks"]


def test_ensure_index_is_idempotent() -> None:
    admin = FakeAdmin()

    _ensure(admin)
    outcome, _ = _ensure(admin)

    assert outcome == "exists"
    assert admin.created == ["chunks"]


@pytest.mark.parametrize(("dimension", "metric"), [(384, "dotproduct"), (768, "cosine")])
def test_ensure_index_refuses_shape_mismatch(dimension: int, metric: str) -> None:
    admin = FakeAdmin(IndexInfo("chunks", dimension, metric, "http://chunks", ready=True))

    with pytest.raises(IndexMismatchError, match="chunks"):
        _ensure(admin)


def test_create_client_requires_key_or_host() -> None:
    with pytest.raises(PineconeNotConfiguredError):
        create_client("", "")


def test_admin_factory_picks_local_when_host_is_set() -> None:
    assert isinstance(create_index_admin("", "http://localhost:5080"), LocalIndexAdmin)
    assert isinstance(create_index_admin("key"), CloudIndexAdmin)


def _local_admin(handler: httpx.MockTransport) -> LocalIndexAdmin:
    return LocalIndexAdmin("http://localhost:5080/", transport=handler)


def test_local_admin_describe_and_create() -> None:
    seen: list[httpx.Request] = []
    store: dict[str, dict[str, object]] = {}

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.method == "GET":
            name = request.url.path.rsplit("/", 1)[-1]
            return httpx.Response(200, json=store[name]) if name in store else httpx.Response(404)
        body = json.loads(request.content)
        store[body["name"]] = body | {"host": "localhost:5081", "status": {"ready": True}}
        return httpx.Response(201, json=store[body["name"]])

    admin = _local_admin(httpx.MockTransport(handle))

    assert admin.describe("chunks") is None
    created = admin.create("chunks", dimension=768, metric="dotproduct", cloud="aws", region="x")

    assert created == IndexInfo("chunks", 768, "dotproduct", "http://localhost:5081", True)
    assert admin.describe("chunks") == created
    assert all(r.headers["X-Pinecone-Api-Version"] == LOCAL_API_VERSION for r in seen)
    assert json.loads(seen[1].content)["spec"] == {"serverless": {"cloud": "aws", "region": "x"}}


def test_local_admin_delete_ignores_missing_index() -> None:
    admin = _local_admin(httpx.MockTransport(lambda _: httpx.Response(404)))

    admin.delete("missing")  # no exception


def test_local_admin_raises_on_server_error() -> None:
    admin = _local_admin(httpx.MockTransport(lambda _: httpx.Response(500)))

    with pytest.raises(httpx.HTTPStatusError):
        admin.describe("chunks")
