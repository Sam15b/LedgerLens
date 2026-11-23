"""FastAPI application factory.

Every request gets a request id (:class:`RequestIdMiddleware`): the caller's
``X-Request-ID`` if it is a sane token, else a new one. It is bound to every log line of
the request (``core.logging``), returned in the ``X-Request-ID`` header, and included in
every error body, so a user's report can be matched to the logs.

Errors have one shape (:func:`error_body`)::

    {"error": {"code": "not_found", "message": "...", "request_id": "..."}}

for HTTP errors (``code`` from the status), request validation (``invalid_request``, with
the field ``details``) and anything unexpected (``internal_error``, 500, logged with its
traceback; the message does not leak internals).

Browsers may call it from ``CORS_ORIGINS`` (the dashboard, ``http://localhost:3000`` by
default); the request id and chat thread id headers are exposed to them.

Run it on a selector event loop (``make api`` does)::

    uvicorn ledgerlens.api.main:app --port 8000 --loop asyncio:SelectorEventLoop

The chat agents keep their state in Postgres through psycopg's async mode, which does
not run on the Proactor loop uvicorn picks on Windows.
"""

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.datastructures import MutableHeaders
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from ledgerlens import __version__
from ledgerlens.api.agent import AgentProvider
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
    503: "unavailable",
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
    app.add_middleware(RequestIdMiddleware)
    app.add_middleware(  # added last, so it is outermost: 4xx error bodies get CORS headers too
        CORSMiddleware,
        allow_origins=get_settings().cors_origin_list,
        allow_methods=["GET", "POST", "PATCH"],
        allow_headers=["Content-Type", REQUEST_ID_HEADER],
        expose_headers=[REQUEST_ID_HEADER, "X-Thread-Id"],
    )
    app.add_exception_handler(StarletteHTTPException, http_error)
    app.add_exception_handler(RequestValidationError, validation_error)
    app.add_exception_handler(Exception, unexpected_error)
    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(reminders.router)
    app.include_router(documents.router)
    app.include_router(anomalies.router)
    app.include_router(stats.router)
    app.include_router(eval_runs.router)
    return app


app = create_app()
