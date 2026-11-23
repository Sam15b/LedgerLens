"""FastAPI application factory.

Every request gets a request id (:class:`RequestIdMiddleware`): the caller's
``X-Request-ID`` if it is a sane token, else a new one. It is bound to every log line of
the request (``core.logging``), returned in the ``X-Request-ID`` header, and included in
every error body, so a user's report can be matched to the logs.

A request that runs past ``REQUEST_TIMEOUT_S`` is answered 504 (``timeout``,
:class:`TimeoutMiddleware`); a chat run past ``CHAT_TIMEOUT_S`` is stopped with an
``error`` event.

Errors have one shape (:func:`error_body`)::

    {"error": {"code": "not_found", "message": "...", "request_id": "..."}}

for HTTP errors (``code`` from the status), request validation (``invalid_request``, with
the field ``details``) and anything unexpected (``internal_error``, 500, logged with its
traceback; the message does not leak internals).

``POST /chat`` is rate-limited per client (``CHAT_RATE_LIMIT``, :mod:`.ratelimit`): over
it, 429 with ``Retry-After``.

With ``API_KEYS`` set, every route but ``/health`` and ``/ready`` needs one of them in
``X-API-Key`` (:func:`ledgerlens.api.deps.require_api_key`); 401 otherwise.

Browsers may call it from ``CORS_ORIGINS`` (the dashboard, ``http://localhost:3000`` by
default); the request id and chat thread id headers are exposed to them.

Run it on a selector event loop (``make api`` does)::

    uvicorn ledgerlens.api.main:app --port 8000 --loop asyncio:SelectorEventLoop

The chat agents keep their state in Postgres through psycopg's async mode, which does
not run on the Proactor loop uvicorn picks on Windows.
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.datastructures import MutableHeaders
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from ledgerlens import __version__
from ledgerlens.api.agent import AgentProvider
from ledgerlens.api.deps import API_KEY_HEADER, require_api_key
from ledgerlens.api.ratelimit import SlidingWindowLimiter, parse_limit
from ledgerlens.api.routers import anomalies, chat, documents, health, reminders, stats
from ledgerlens.api.routers import eval as eval_runs
from ledgerlens.core.config import get_settings
from ledgerlens.core.logging import configure_logging, reset_request_id, set_request_id
from ledgerlens.observability.tracing import get_tracer

logger = logging.getLogger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"
_REQUEST_ID = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")
CODES = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    413: "too_large",
    415: "unsupported_media_type",
    422: "invalid_request",
    429: "too_many_requests",
    502: "bad_gateway",
    503: "unavailable",
    504: "timeout",
}


class RequestIdMiddleware:
    """Pure ASGI (so streamed responses keep the id in their context too)."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        given = dict(scope.get("headers") or []).get(REQUEST_ID_HEADER.lower().encode(), b"")
        text = given.decode("latin-1")
        request_id = text if _REQUEST_ID.match(text) else uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id

        async def send_with_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message)[REQUEST_ID_HEADER] = request_id
            await send(message)

        token = set_request_id(request_id)
        try:
            await self.app(scope, receive, send_with_id)
        finally:
            reset_request_id(token)


class TimeoutMiddleware:
    """Answers 504 (``timeout``) when a request runs past ``REQUEST_TIMEOUT_S``.

    Pure ASGI, inside :class:`RequestIdMiddleware` so the error carries the request id.
    ``/chat`` is left alone: it streams for as long as the run takes, under its own
    deadline (``CHAT_TIMEOUT_S``). A response that has already started cannot become a
    504; it is cut off and logged. Work handed to a thread pool finishes in the
    background, but the client gets its answer on time.
    """

    def __init__(self, app: ASGIApp, exempt: frozenset[str] = frozenset({"/chat"})) -> None:
        self.app = app
        self.exempt = exempt

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] in self.exempt:
            await self.app(scope, receive, send)
            return
        seconds = get_settings().request_timeout_s
        started = False

        async def tracked(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            async with asyncio.timeout(seconds):
                await self.app(scope, receive, tracked)
        except TimeoutError:
            logger.warning("%s %s ran past %gs", scope["method"], scope["path"], seconds)
            if started:
                return
            request_id = (scope.get("state") or {}).get("request_id")
            body = {
                "error": {
                    "code": "timeout",
                    "message": f"the request took longer than {seconds:g} s; try again",
                    "request_id": request_id,
                }
            }
            response = JSONResponse(body, status_code=504)
            await response(scope, receive, send)


def request_id_of(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def error_body(request: Request, code: str, message: str, details: Any = None) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message, "request_id": request_id_of(request)}
    if details is not None:
        error["details"] = details
    return {"error": error}


def _with_id(request: Request, headers: Mapping[str, str] | None = None) -> dict[str, str]:
    out = dict(headers or {})
    if (request_id := request_id_of(request)) is not None:
        out[REQUEST_ID_HEADER] = request_id
    return out


async def http_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)
    code = CODES.get(exc.status_code, f"http_{exc.status_code}")
    return JSONResponse(
        error_body(request, code, str(exc.detail)),
        status_code=exc.status_code,
        headers=_with_id(request, exc.headers),
    )


async def validation_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    details = jsonable_encoder(
        [{k: e[k] for k in ("loc", "msg", "type") if k in e} for e in exc.errors()]
    )
    first = details[0] if details else {}
    where = ".".join(str(part) for part in first.get("loc", []))
    message = f"{where}: {first.get('msg')}" if first else "invalid request"
    return JSONResponse(
        error_body(request, "invalid_request", message, details),
        status_code=422,
        headers=_with_id(request),
    )


async def unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        error_body(
            request,
            "internal_error",
            "Something went wrong on our side. Quote the request id if you report it.",
        ),
        status_code=500,
        headers=_with_id(request),
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    yield
    await app.state.agents.aclose()
    get_tracer().flush()  # spans still queued for the tracing backend


def create_app(agents: AgentProvider | None = None) -> FastAPI:
    """The app; ``agents`` builds the agent graph (default: the live one, on first use)."""
    configure_logging(get_settings().log_level)
    app = FastAPI(
        lifespan=lifespan,
        title="LedgerLens API",
        version=__version__,
        description="Agentic GraphRAG over invoices, purchase orders and contracts.",
    )
    app.state.agents = agents or AgentProvider()
    chat_limit = parse_limit(get_settings().chat_rate_limit)
    app.state.chat_limiter = SlidingWindowLimiter(chat_limit) if chat_limit else None
    app.add_middleware(TimeoutMiddleware)  # innermost of these: it sees the request id
    app.add_middleware(RequestIdMiddleware)
    app.add_middleware(  # added last, so it is outermost: 4xx error bodies get CORS headers too
        CORSMiddleware,
        allow_origins=get_settings().cors_origin_list,
        allow_methods=["GET", "POST", "PATCH"],
        allow_headers=["Content-Type", "Authorization", API_KEY_HEADER, REQUEST_ID_HEADER],
        expose_headers=[REQUEST_ID_HEADER, "X-Thread-Id"],
    )
    app.add_exception_handler(StarletteHTTPException, http_error)
    app.add_exception_handler(RequestValidationError, validation_error)
    app.add_exception_handler(Exception, unexpected_error)
    app.include_router(health.router)  # open: load balancers and uptime checks call it
    keyed = [Depends(require_api_key)]
    for module in (chat, reminders, documents, anomalies, stats, eval_runs):
        app.include_router(module.router, dependencies=keyed)
    if not get_settings().api_key_list:
        logger.warning("API_KEYS is empty: the API accepts requests without a key")
    return app


app = create_app()
