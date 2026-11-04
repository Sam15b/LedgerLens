import threading
from collections.abc import Callable, Iterator

import pytest
from fastapi.testclient import TestClient

from ledgerlens.api import deps, health_checks
from ledgerlens.api.health_checks import HealthCheck, default_checks, run_checks
from ledgerlens.api.main import create_app
from ledgerlens.api.routers.health import get_health_checks
from ledgerlens.storage.pinecone_store import IndexInfo


def ok(detail: str | None = None) -> Callable[[], str | None]:
    return lambda: detail


def fail(message: str) -> Callable[[], str | None]:
    def check() -> str | None:
        raise ConnectionError(message)

    return check


@pytest.fixture
def make_client() -> Iterator[Callable[[list[HealthCheck]], TestClient]]:
    clients: list[TestClient] = []

    def factory(checks: list[HealthCheck]) -> TestClient:
        app = create_app()
        app.dependency_overrides[get_health_checks] = lambda: checks
        client = TestClient(app)
        clients.append(client)
        return client

    yield factory
    for client in clients:
        client.close()


# --- endpoints ---------------------------------------------------------------------------


def test_health_ok_when_all_dependencies_up(make_client: Callable[..., TestClient]) -> None:
    client = make_client([HealthCheck("postgres", ok()), HealthCheck("neo4j", ok("v5"))])

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["dependencies"]["postgres"]["status"] == "ok"
    assert body["dependencies"]["neo4j"]["detail"] == "v5"
    assert body["dependencies"]["neo4j"]["latency_ms"] >= 0
    assert {"version", "env"} <= body.keys()


def test_health_degraded_but_200_when_required_dependency_down(
    make_client: Callable[..., TestClient],
) -> None:
    client = make_client([HealthCheck("postgres", ok()), HealthCheck("neo4j", fail("refused"))])

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["dependencies"]["neo4j"] == body["dependencies"]["neo4j"] | {
        "status": "down",
        "required": True,
        "error": "ConnectionError: refused",
    }


def test_ready_returns_503_when_required_dependency_down(
    make_client: Callable[..., TestClient],
) -> None:
    client = make_client([HealthCheck("redis", fail("refused"))])

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "degraded"


def test_ready_ignores_optional_dependencies(make_client: Callable[..., TestClient]) -> None:
    client = make_client(
        [HealthCheck("postgres", ok()), HealthCheck("ollama", fail("refused"), required=False)]
    )

    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["dependencies"]["ollama"]["status"] == "down"


# --- run_checks ------------------------------------------------------------------------


def test_run_checks_times_out_hanging_checks() -> None:
    release = threading.Event()

    def hang() -> str | None:
        release.wait(5)
        return None

    try:
        results = run_checks([HealthCheck("slow", hang), HealthCheck("fast", ok())], timeout_s=0.2)
    finally:
        release.set()

    assert results["fast"].status == "ok"
    assert results["slow"].status == "down"
    assert results["slow"].error is not None and "timed out" in results["slow"].error


def test_run_checks_truncates_long_errors() -> None:
    results = run_checks([HealthCheck("x", fail("boom " * 200))], timeout_s=1)

    error = results["x"].error
    assert error is not None and len(error) <= 300


def test_run_checks_with_no_checks() -> None:
    assert run_checks([], timeout_s=1) == {}


def test_default_checks_cover_every_store() -> None:
    checks = {c.name: c for c in default_checks()}

    assert set(checks) == {"postgres", "neo4j", "redis", "pinecone", "ollama"}
    assert not checks["ollama"].required
    assert all(c.required for name, c in checks.items() if name != "ollama")


# --- individual checks -----------------------------------------------------------------


class FakeAdmin:
    def __init__(self, info: IndexInfo | None) -> None:
        self.info = info

    def describe(self, name: str) -> IndexInfo | None:
        return self.info


@pytest.mark.parametrize(
    ("info", "expected"),
    [
        (None, "missing"),
        (IndexInfo("ledgerlens-chunks", 768, "dotproduct", "h", ready=True), "ready"),
        (IndexInfo("ledgerlens-chunks", 768, "dotproduct", None, ready=False), "initializing"),
    ],
)
def test_check_pinecone_reports_index_state(
    monkeypatch: pytest.MonkeyPatch, info: IndexInfo | None, expected: str
) -> None:
    monkeypatch.setattr(deps, "get_pinecone_admin", lambda: FakeAdmin(info))

    detail = health_checks.check_pinecone()

    assert detail is not None and expected in detail


def test_check_redis_propagates_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    class DownRedis:
        def ping(self) -> bool:
            raise ConnectionError("refused")

    monkeypatch.setattr(deps, "get_redis", DownRedis)

    with pytest.raises(ConnectionError):
        health_checks.check_redis()
