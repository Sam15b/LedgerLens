"""A sliding-window rate limiter, per client, in this process's memory.

``CHAT_RATE_LIMIT`` (``20/minute`` by default) caps how often one client may start a chat
run: each run can call LLMs and cost money, so this is the API's spending brake. A client
is the API key's id plus the end user's address. The dashboard forwards that address in
``X-Forwarded-For``, which is believed only from a caller with a valid key; anyone else is
counted by the connection's own address.

Counts live in memory, per API process: with several processes, the effective limit is
the limit times the number of processes. Put a shared limit at the load balancer for that.
"""

from __future__ import annotations

import math
import re
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from fastapi import HTTPException, Request, status

_SPEC = re.compile(r"^\s*(\d+)\s*/\s*(\d*)\s*(s|sec|second|m|min|minute|h|hour)s?\s*$", re.I)
_UNIT_S = {"s": 1, "sec": 1, "second": 1, "m": 60, "min": 60, "minute": 60, "h": 3600, "hour": 3600}


@dataclass(frozen=True)
class Limit:
    requests: int
    window_s: float

    def __str__(self) -> str:
        return f"{self.requests} per {self.window_s:g} s"


def parse_limit(spec: str) -> Limit | None:
    """``20/minute``, ``5/10s``, ``100/hour`` -> a limit; empty or ``0/...`` -> no limit."""
    if not spec.strip():
        return None
    match = _SPEC.match(spec)
    if match is None:
        raise ValueError(f"not a rate limit: {spec!r} (expected e.g. 20/minute or 5/10s)")
    count, multiple, unit = match.groups()
    if int(count) == 0:
        return None
    return Limit(int(count), float(multiple or 1) * _UNIT_S[unit.lower()])


class SlidingWindowLimiter:
    """At most ``limit.requests`` hits per key in any ``limit.window_s`` seconds."""

    def __init__(
        self,
        limit: Limit,
        clock: Callable[[], float] = time.monotonic,
        max_keys: int = 50_000,
    ) -> None:
        self.limit = limit
        self.clock = clock
        self.max_keys = max_keys
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def hit(self, key: str) -> float | None:
        """Count a request; ``None`` if allowed, else the seconds until one would be."""
        now = self.clock()
        start = now - self.limit.window_s
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and hits[0] <= start:
                hits.popleft()
            if len(hits) >= self.limit.requests:
                return hits[0] + self.limit.window_s - now
            hits.append(now)
            if len(self._hits) > self.max_keys:
                self._forget(start)
            return None

    def _forget(self, start: float) -> None:
        """Drop clients with no hit in the window (bounds memory under many addresses)."""
        for key in [k for k, hits in self._hits.items() if not hits or hits[-1] <= start]:
            del self._hits[key]


def client_key(request: Request) -> str:
    """Who is asking: the API key's id and the end user's address (see the module doc)."""
    caller = getattr(request.state, "api_key_id", None)
    address = request.client.host if request.client else "unknown"
    if caller:
        forwarded = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
        return f"{caller}:{forwarded or address}"
    return address


def enforce(limiter: SlidingWindowLimiter | None, request: Request, what: str) -> None:
    """Raise 429 (with ``Retry-After``) when the client is over the limit."""
    if limiter is None:
        return
    wait = limiter.hit(client_key(request))
    if wait is not None:
        seconds = max(1, math.ceil(wait))
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            f"too many {what}: the limit is {limiter.limit}; try again in {seconds} s",
            headers={"Retry-After": str(seconds)},
        )
