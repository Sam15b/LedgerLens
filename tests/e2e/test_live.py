"""Smoke suite against a deployed LedgerLens: what a fresh visitor and the operator see.

Runs only when pointed at an environment, and never by default (``live`` marker)::

    LEDGERLENS_LIVE_URL=https://d111111abcdef8.cloudfront.net \\
    LEDGERLENS_LIVE_SESSION_SECRET=... \\
    pytest -m live tests/e2e/test_live.py

* ``LEDGERLENS_LIVE_URL``: the public address (``terraform output url``). Unset: skipped.
* ``LEDGERLENS_LIVE_SESSION_SECRET``: the dashboard's ``DASHBOARD_SESSION_SECRET`` (from
  the app secret). With it the suite signs a session cookie the way the dashboard does
  (``web/lib/session.ts``) and checks the pages behind the login; without it those skip.
* ``LEDGERLENS_LIVE_ENV``: the ``APP_ENV`` the API should report (default ``prod``).
* ``LEDGERLENS_LIVE_SEEDED=1``: also require the demo data (``seed_demo_data.py``).

Everything goes through the public URL, as a browser would: CloudFront, the load
balancer, the dashboard and its ``/api`` forwarder. Nothing here writes.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import time
from collections.abc import Iterator
from typing import Any

import httpx
import pytest

URL = os.environ.get("LEDGERLENS_LIVE_URL", "").rstrip("/")
SECRET = os.environ.get("LEDGERLENS_LIVE_SESSION_SECRET", "")
ENV = os.environ.get("LEDGERLENS_LIVE_ENV", "prod")
SEEDED = os.environ.get("LEDGERLENS_LIVE_SEEDED") == "1"
SESSION_COOKIE = "ledgerlens_session"

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not URL, reason="set LEDGERLENS_LIVE_URL to the deployed address"),
]
needs_session = pytest.mark.skipif(
    not SECRET, reason="set LEDGERLENS_LIVE_SESSION_SECRET to check the pages behind the login"
)


def session_cookie(secret: str, *, now_ms: int | None = None, hours: int = 1) -> str:
    """``<expiry>.<signature>``: HMAC-SHA256 of ``v1.<expiry>``, base64url without padding,
    exactly as ``createSession`` in ``web/lib/session.ts`` signs it."""
    expires = (now_ms if now_ms is not None else int(time.time() * 1000)) + hours * 3_600_000
    digest = hmac.new(secret.encode(), f"v1.{expires}".encode(), hashlib.sha256).digest()
    return f"{expires}.{base64.urlsafe_b64encode(digest).decode().rstrip('=')}"


@pytest.fixture(scope="module")
def visitor() -> Iterator[httpx.Client]:
    """A browser with no session; redirects are checked, not followed."""
    with httpx.Client(base_url=URL, timeout=30, follow_redirects=False) as client:
        yield client


@pytest.fixture(scope="module")
def signed_in() -> Iterator[httpx.Client]:
    with httpx.Client(
        base_url=URL,
        timeout=60,
        follow_redirects=False,
        cookies={SESSION_COOKIE: session_cookie(SECRET)},
    ) as client:
        yield client


# --- a fresh visitor ---------------------------------------------------------------------


def test_api_health_reports_every_store_ok(visitor: httpx.Client) -> None:
    response = visitor.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["env"] == ENV
    down = {k: v for k, v in body["dependencies"].items() if v["required"] and v["status"] != "ok"}
    assert body["status"] == "ok", f"required stores down: {down}"


def test_dashboard_liveness_probe(visitor: httpx.Client) -> None:
    response = visitor.get("/healthz")

    assert response.status_code == 200


def test_plain_http_is_redirected_to_https(visitor: httpx.Client) -> None:
    if not URL.startswith("https://"):
        pytest.skip("the deployment is not served over HTTPS")
    response = visitor.get("http://" + URL.removeprefix("https://") + "/healthz")

    assert response.status_code in (301, 308)
    assert response.headers["location"].startswith("https://")


def test_pages_need_a_login_and_keep_the_scheme(visitor: httpx.Client) -> None:
    response = visitor.get("/anomalies")

    assert response.status_code == 307
    # Next sends a same-host redirect as a relative Location; either way it must land on
    # the scheme the visitor used (behind CloudFront the load balancer sees plain HTTP)
    location = httpx.URL(URL).join(response.headers["location"])
    assert location.path == "/login"
    assert location.params.get("next") == "/anomalies"
    assert location.scheme == httpx.URL(URL).scheme
    assert visitor.get("/login").status_code == 200


def test_data_is_not_reachable_without_a_session(visitor: httpx.Client) -> None:
    assert visitor.get("/api/stats/overview").status_code == 401  # the dashboard's forwarder
    # the API itself is not public: only /health is routed to it
    assert visitor.get("/stats/overview").status_code in (307, 401, 404)


def test_forged_session_is_refused(visitor: httpx.Client) -> None:
    forged = session_cookie("not-the-secret")
    response = visitor.get("/api/health", cookies={SESSION_COOKIE: forged})

    assert response.status_code == 401


# --- signed in ---------------------------------------------------------------------------


@needs_session
def test_overview_renders(signed_in: httpx.Client) -> None:
    response = signed_in.get("/")

    assert response.status_code == 200
    assert "Documents ingested" in response.text


@needs_session
@pytest.mark.parametrize("path", ["/documents", "/anomalies", "/reminders", "/eval", "/chat"])
def test_every_page_renders(signed_in: httpx.Client, path: str) -> None:
    assert signed_in.get(path).status_code == 200


@needs_session
def test_api_answers_through_the_forwarder(signed_in: httpx.Client) -> None:
    overview = signed_in.get("/api/stats/overview")
    assert overview.status_code == 200
    assert {"documents", "unpaid"} <= overview.json().keys()

    for path in ("/api/documents?limit=5", "/api/anomalies?limit=5", "/api/eval/runs?limit=1"):
        response = signed_in.get(path)
        assert response.status_code == 200, path
        assert isinstance(response.json(), list), path


@needs_session
@pytest.mark.skipif(not SEEDED, reason="set LEDGERLENS_LIVE_SEEDED=1 once the demo is seeded")
def test_demo_data_is_there(signed_in: httpx.Client) -> None:
    documents: dict[str, Any] = signed_in.get("/api/stats/overview").json()["documents"]
    assert documents["ready"] >= 28  # the demo bundle's rendered documents

    anomalies = signed_in.get("/api/anomalies?limit=50").json()
    assert anomalies, "the demo includes one invoice per anomaly kind"
