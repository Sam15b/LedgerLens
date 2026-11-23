"""The API's protections: API keys (401), the chat rate limit (429), request timeouts
(504) and the chat run deadline, all with the one error shape."""

import asyncio
import time
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from starlette.requests import Request
from tests.unit.api.test_chat import parse_sse

from ledgerlens.agents.graph import AgentNodes, build_graph
from ledgerlens.agents.nodes.router import Router
from ledgerlens.agents.state import LedgerState
from ledgerlens.api.agent import AgentProvider, AgentRuntime, AgentUnavailableError
from ledgerlens.api.deps import key_id
from ledgerlens.api.main import create_app
from ledgerlens.api.ratelimit import Limit, SlidingWindowLimiter, client_key, parse_limit
from ledgerlens.api.routers.chat import with_deadline
from ledgerlens.core.config import get_settings

KEYS = "key-one-0123456789,key-two-9876543210"


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[pytest.MonkeyPatch]:
    """Set environment variables, then rebuild the cached settings; undone afterwards."""
    get_settings.cache_clear()
    yield monkeypatch
    get_settings.cache_clear()


def configure(settings: pytest.MonkeyPatch, **env: str) -> None:
    for name, value in env.items():
        settings.setenv(name, value)
    get_settings.cache_clear()


async def no_agents(stack: Any) -> AgentRuntime:
    """The agents never come up (fast, unlike a real connection attempt)."""
    raise AgentUnavailableError("no agents in this test")


def client(agents: AgentProvider | None = None) -> TestClient:
    app = create_app(agents or AgentProvider(no_agents))
    return TestClient(app, raise_server_exceptions=False)


# --------------------------------------------------------------------------- API keys


def test_without_keys_configured_auth_is_off(settings: pytest.MonkeyPatch) -> None:
    configure(settings, API_KEYS="")
    response = client().get("/eval/runs", params={"limit": 0})
    assert response.status_code == 422  # reached the endpoint's validation


@pytest.mark.parametrize(
    "headers",
    [{}, {"X-API-Key": "wrong"}, {"Authorization": "Bearer wrong"}, {"Authorization": "Basic x"}],
)
def test_a_missing_or_wrong_key_is_401(
    settings: pytest.MonkeyPatch, headers: dict[str, str]
) -> None:
    configure(settings, API_KEYS=KEYS)
    response = client().get("/documents", headers=headers)
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "ApiKey"
    error = response.json()["error"]
    assert error["code"] == "unauthorized" and "X-API-Key" in error["message"]
    assert error["request_id"] == response.headers["x-request-id"]


@pytest.mark.parametrize(
    "headers",
    [
        {"X-API-Key": "key-one-0123456789"},
        {"X-API-Key": "key-two-9876543210"},
        {"Authorization": "Bearer key-two-9876543210"},
    ],
)
def test_any_configured_key_is_accepted(
    settings: pytest.MonkeyPatch, headers: dict[str, str]
) -> None:
    configure(settings, API_KEYS=KEYS)
    response = client().get("/eval/runs", params={"limit": 0}, headers=headers)
    assert response.status_code == 422  # past auth, into validation


def test_the_key_is_checked_before_the_body(settings: pytest.MonkeyPatch) -> None:
    configure(settings, API_KEYS=KEYS)
    response = client().post("/chat", json={"not": "a chat request"})
    assert response.status_code == 401


def test_health_and_cors_preflight_need_no_key(settings: pytest.MonkeyPatch) -> None:
    configure(settings, API_KEYS=KEYS)
    api = client()
    assert api.get("/health").status_code == 200
    preflight = api.options(
        "/chat",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "x-api-key,content-type",
        },
    )
    assert preflight.status_code == 200
    assert "x-api-key" in preflight.headers["access-control-allow-headers"].lower()


def test_every_route_but_health_is_keyed(settings: pytest.MonkeyPatch) -> None:
    configure(settings, API_KEYS=KEYS)
    api = client()  # 401 comes before any store is touched
    open_paths = set()
    app = api.app
    assert isinstance(app, FastAPI)
    for path, methods in app.openapi()["paths"].items():
        if "get" in methods and "{" not in path and api.get(path).status_code != 401:
            open_paths.add(path)
    assert open_paths == {"/health", "/ready"}


# --------------------------------------------------------------------------- rate limit


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("20/minute", Limit(20, 60)),
        ("5/10s", Limit(5, 10)),
        ("100 / hour", Limit(100, 3600)),
        ("3/second", Limit(3, 1)),
        ("2/mins", Limit(2, 60)),
        ("", None),
        ("0/minute", None),
    ],
)
def test_parse_limit(spec: str, expected: Limit | None) -> None:
    assert parse_limit(spec) == expected


@pytest.mark.parametrize("spec", ["twenty/minute", "5/fortnight", "5"])
def test_parse_limit_refuses_nonsense(spec: str) -> None:
    with pytest.raises(ValueError, match="not a rate limit"):
        parse_limit(spec)


def test_the_window_slides() -> None:
    now = [100.0]
    limiter = SlidingWindowLimiter(Limit(2, 10), clock=lambda: now[0])
    assert limiter.hit("a") is None
    now[0] = 104.0
    assert limiter.hit("a") is None
    assert limiter.hit("a") == pytest.approx(6.0)  # the first hit leaves the window at 110
    assert limiter.hit("b") is None  # another client has its own count
    now[0] = 110.0
    assert limiter.hit("a") is None
    assert limiter.hit("a") == pytest.approx(4.0)


def test_idle_clients_are_forgotten() -> None:
    now = [0.0]
    limiter = SlidingWindowLimiter(Limit(1, 10), clock=lambda: now[0], max_keys=3)
    for n in range(3):
        limiter.hit(f"old-{n}")
    now[0] = 20.0
    limiter.hit("new-1")
    assert set(limiter._hits) == {"new-1"}


def request_from(host: str, headers: dict[str, str], key: str | None = None) -> Request:
    scope = {
        "type": "http",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
        "client": (host, 50000),
        "state": {"api_key_id": key} if key else {},
    }
    return Request(scope)


def test_forwarded_addresses_count_only_from_a_keyed_caller() -> None:
    spoofed = request_from("10.0.0.9", {"X-Forwarded-For": "1.2.3.4"})
    assert client_key(spoofed) == "10.0.0.9"
    dashboard = request_from("10.0.0.2", {"X-Forwarded-For": "1.2.3.4, 10.0.0.1"}, key="k1")
    assert client_key(dashboard) == "k1:1.2.3.4"
    direct = request_from("10.0.0.2", {}, key="k1")
    assert client_key(direct) == "k1:10.0.0.2"


def test_chat_over_the_limit_is_429_with_retry_after(settings: pytest.MonkeyPatch) -> None:
    configure(settings, API_KEYS=KEYS, CHAT_RATE_LIMIT="2/minute")
    api = client()
    key = {"X-API-Key": "key-one-0123456789"}
    first = [api.post("/chat", json={"message": "hi"}, headers=key) for _ in range(2)]
    assert [r.status_code for r in first] == [503, 503]  # allowed through (no agents here)
    third = api.post("/chat", json={"message": "hi"}, headers=key)
    assert third.status_code == 429
    assert 1 <= int(third.headers["retry-after"]) <= 60
    error = third.json()["error"]
    assert error["code"] == "too_many_requests" and "2 per 60 s" in error["message"]

    other = {**key, "X-Forwarded-For": "203.0.113.7"}
    assert api.post("/chat", json={"message": "hi"}, headers=other).status_code == 503
    counted = set(api.app.state.chat_limiter._hits)  # type: ignore[attr-defined]
    caller = key_id("key-one-0123456789")
    assert counted == {f"{caller}:testclient", f"{caller}:203.0.113.7"}


def test_no_chat_limit_when_turned_off(settings: pytest.MonkeyPatch) -> None:
    configure(settings, CHAT_RATE_LIMIT="")
    api = client()
    assert api.app.state.chat_limiter is None  # type: ignore[attr-defined]
    assert {api.post("/chat", json={"message": "hi"}).status_code for _ in range(5)} == {503}


# --------------------------------------------------------------------------- timeouts


def test_a_slow_request_is_504_with_the_error_shape(settings: pytest.MonkeyPatch) -> None:
    configure(settings, REQUEST_TIMEOUT_S="0.05")
    api = client()

    app = api.app
    assert isinstance(app, FastAPI)

    @app.get("/slow")
    async def slow() -> dict[str, str]:
        await asyncio.sleep(2)
        return {"never": "sent"}

    started = time.monotonic()
    response = api.get("/slow", headers={"X-Request-ID": "slow-1"})
    assert time.monotonic() - started < 1.5
    assert response.status_code == 504
    assert response.json() == {
        "error": {
            "code": "timeout",
            "message": "the request took longer than 0.05 s; try again",
            "request_id": "slow-1",
        }
    }
    assert response.headers["x-request-id"] == "slow-1"


async def numbers(delay: float) -> AsyncIterator[str]:
    for n in range(3):
        await asyncio.sleep(delay)
        yield str(n)


def collect(stream: AsyncIterator[str], seconds: float) -> tuple[list[str], bool]:
    async def run() -> tuple[list[str], bool]:
        out: list[str] = []
        try:
            async for item in with_deadline(stream, seconds):
                out.append(item)
        except TimeoutError:
            return out, True
        return out, False

    return asyncio.run(run())


def test_with_deadline_passes_a_fast_stream_through() -> None:
    assert collect(numbers(0), 5) == (["0", "1", "2"], False)


def test_with_deadline_stops_a_slow_stream() -> None:
    items, timed_out = collect(numbers(0.2), 0.3)
    assert timed_out and items == ["0"]


def slow_qa(state: LedgerState) -> dict[str, Any]:
    time.sleep(1.0)
    return {"messages": [AIMessage("too late", name="qa")], "step_count": 1}


def test_a_chat_run_past_its_deadline_is_stopped(settings: pytest.MonkeyPatch) -> None:
    configure(settings, CHAT_TIMEOUT_S="0.2")
    graph = build_graph(AgentNodes(router=Router(None), qa=slow_qa), checkpointer=InMemorySaver())

    async def factory(stack: Any) -> AgentRuntime:
        return AgentRuntime(graph)

    api = client(AgentProvider(factory))
    with api.stream("POST", "/chat", json={"message": "Which invoices are unpaid?"}) as response:
        events = parse_sse(response.read().decode())
    names = [name for name, _ in events]
    assert names[-2:] == ["error", "done"]
    assert "longer than 0.2 s" in events[-2][1]["message"]
    assert events[-1][1]["status"] == "completed"  # stopped, not waiting for a person
    thread = events[-1][1]["thread_id"]

    time.sleep(1.0)  # let the abandoned node finish in its thread
    again = api.post("/chat", json={"message": "and now?", "thread_id": thread})
    assert again.status_code == 200  # the thread is not stuck in a fake "interrupted"
