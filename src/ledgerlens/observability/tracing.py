"""Tracing: every ``/chat`` request becomes one trace of nested spans.

    with span("chat", inputs={"message": text}) as root:
        ...                                  # router -> agent -> tools -> LLM calls nest here
        root.set_outputs(answer=reply)

A span is one timed step -- an agent node, a tool call, an LLM call -- with inputs, outputs,
attributes and an error, if it failed. The span opened inside another becomes its child
(through a context variable, so it follows ``await`` and ``asyncio.to_thread``), which
gives the tree ``chat -> router -> qa -> tool search_documents -> llm smart``.

Finished spans go to one backend (``TRACING_BACKEND``):

* ``langsmith`` -- LangSmith runs in project ``LANGSMITH_PROJECT`` (needs
  ``LANGSMITH_API_KEY``); the tree is in the LangSmith UI;
* ``jsonl`` -- one JSON line per span in ``TRACING_PATH``, for a machine with no LangSmith
  key; ``python -m ledgerlens.observability.tracing show`` prints the last trace as a tree.

With no backend the tracer is off: :func:`span` still runs the code inside it and hands
out a no-op span, so callers never check whether tracing is on. A backend error is logged
and dropped; it never reaches the traced code.
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from ledgerlens.core.config import Settings, get_settings, repo_path

logger = logging.getLogger(__name__)

SpanKind = Literal["chain", "llm", "tool", "retriever"]
"""LangSmith's run types for what LedgerLens traces."""

MAX_TEXT = 4_000
"""Longer strings in span inputs and outputs are cut to this many characters."""
MAX_ITEMS = 50
"""Longer lists in span inputs and outputs keep this many items."""


def compact(value: Any, *, depth: int = 0) -> Any:
    """``value`` as JSON-safe data of bounded size (long strings cut, long lists capped)."""
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        return value if len(value) <= MAX_TEXT else value[:MAX_TEXT] + "...[truncated]"
    if depth >= 6:
        return compact(repr(value), depth=depth)
    if isinstance(value, Mapping):
        return {str(k): compact(v, depth=depth + 1) for k, v in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        items = list(value)
        out = [compact(v, depth=depth + 1) for v in items[:MAX_ITEMS]]
        if len(items) > MAX_ITEMS:
            out.append(f"...[{len(items) - MAX_ITEMS} more]")
        return out
    dump = getattr(value, "model_dump", None)
    if callable(dump):  # pydantic models
        return compact(dump(mode="json"), depth=depth)
    return compact(str(value), depth=depth)


@dataclass
class SpanRecord:
    """One span as the backends see it."""

    name: str
    kind: SpanKind
    trace_id: str
    span_id: str
    parent_id: str | None
    start_time: datetime
    end_time: datetime | None = None
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    attributes: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def duration_ms(self) -> float | None:
        if self.end_time is None:
            return None
        return round((self.end_time - self.start_time).total_seconds() * 1000, 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "attributes": self.attributes,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SpanRecord:
        end = data.get("end_time")
        return cls(
            name=data["name"],
            kind=data["kind"],
            trace_id=data["trace_id"],
            span_id=data["span_id"],
            parent_id=data.get("parent_id"),
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=datetime.fromisoformat(end) if end else None,
            inputs=dict(data.get("inputs") or {}),
            outputs=dict(data.get("outputs") or {}),
            attributes=dict(data.get("attributes") or {}),
            error=data.get("error"),
        )


# --------------------------------------------------------------------------- backends


class TraceBackend(Protocol):
    """Where spans go. ``start`` returns a handle that ``end`` gets back (its parent's
    handle comes with ``start``); either may raise -- the tracer logs and carries on."""

    def start(self, record: SpanRecord, parent: Any) -> Any: ...

    def end(self, record: SpanRecord, handle: Any) -> None: ...

    def flush(self) -> None: ...


class MemoryBackend:
    """Keeps finished spans in memory (tests, and checking a trace in-process)."""

    def __init__(self) -> None:
        self.records: list[SpanRecord] = []
        self._lock = threading.Lock()

    def start(self, record: SpanRecord, parent: Any) -> Any:
        return None

    def end(self, record: SpanRecord, handle: Any) -> None:
        with self._lock:
            self.records.append(record)

    def flush(self) -> None:
        pass

    def by_name(self, name: str) -> list[SpanRecord]:
        return [r for r in self.records if r.name == name]


class JsonlBackend:
    """Appends each finished span to ``path`` as one JSON line."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def start(self, record: SpanRecord, parent: Any) -> Any:
        return None

    def end(self, record: SpanRecord, handle: Any) -> None:
        line = json.dumps(record.to_dict(), default=str) + "\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line)

    def flush(self) -> None:
        pass


class LangSmithBackend:
    """Each span is a LangSmith run (its parent's child), posted when it ends.

    The LangSmith client sends runs in batches from a background thread; :meth:`flush`
    waits for what is queued (the API calls it on shutdown).
    """

    def __init__(self, project: str, *, api_key: str | None = None, client: Any = None) -> None:
        from langsmith import Client

        self.project = project
        self.client = client if client is not None else Client(api_key=api_key or None)

    def start(self, record: SpanRecord, parent: Any) -> Any:
        from langsmith.run_trees import RunTree

        if parent is not None:
            return parent.create_child(
                name=record.name,
                run_type=record.kind,
                run_id=uuid.UUID(record.span_id),
                inputs=record.inputs,
                start_time=record.start_time,
            )
        return RunTree(
            name=record.name,
            run_type=record.kind,
            id=uuid.UUID(record.span_id),
            trace_id=uuid.UUID(record.trace_id),
            inputs=record.inputs,
            start_time=record.start_time,
            project_name=self.project,
            ls_client=self.client,
        )

    def end(self, record: SpanRecord, handle: Any) -> None:
        if handle is None:
            return
        handle.end(
            outputs=record.outputs,
            error=record.error,
            end_time=record.end_time,
            metadata=record.attributes,
        )
        handle.post()

    def flush(self) -> None:
        self.client.flush()


# --------------------------------------------------------------------------- spans


class Span:
    """A span being recorded; :meth:`end` (or leaving :func:`span`) sends it."""

    recording = True

    def __init__(
        self, tracer: Tracer, record: SpanRecord, parent: Span | None, handle: Any
    ) -> None:
        self.tracer = tracer
        self.record = record
        self.parent = parent
        self.handle = handle
        self._ended = False

    @property
    def trace_id(self) -> str:
        return self.record.trace_id

    def set(self, key: str, value: Any) -> None:
        self.record.attributes[key] = compact(value)

    def set_attributes(self, attributes: Mapping[str, Any]) -> None:
        for key, value in attributes.items():
            self.set(key, value)

    def set_outputs(self, **outputs: Any) -> None:
        self.record.outputs.update(compact(outputs))

    def record_error(self, exc: BaseException | str) -> None:
        self.record.error = exc if isinstance(exc, str) else f"{type(exc).__name__}: {exc}"

    def end(self) -> None:
        if self._ended:
            return
        self._ended = True
        self.record.end_time = datetime.now(UTC)
        self.tracer.export(self)


class NoopSpan(Span):
    """What tracing off hands out: every method does nothing."""

    recording = False

    def __init__(self) -> None:
        pass

    @property
    def trace_id(self) -> str:
        return ""

    def set(self, key: str, value: Any) -> None:
        pass

    def set_outputs(self, **outputs: Any) -> None:
        pass

    def record_error(self, exc: BaseException | str) -> None:
        pass

    def end(self) -> None:
        pass


NOOP_SPAN = NoopSpan()

_current: ContextVar[Span | None] = ContextVar("ledgerlens_span", default=None)


def current_span() -> Span | None:
    """The innermost open span in this context (``None`` outside any)."""
    return _current.get()


class Tracer:
    """Makes spans and hands finished ones to the backend (``None``: tracing off)."""

    def __init__(self, backend: TraceBackend | None = None) -> None:
        self.backend = backend
        self._warned = False

    @property
    def enabled(self) -> bool:
        return self.backend is not None

    def start_span(
        self,
        name: str,
        *,
        kind: SpanKind = "chain",
        inputs: Mapping[str, Any] | None = None,
        attributes: Mapping[str, Any] | None = None,
        parent: Span | None = None,
    ) -> Span:
        """A started span, the child of ``parent`` (default: :func:`current_span`). It is
        not made current: :func:`span` does that."""
        if self.backend is None:
            return NOOP_SPAN
        parent = parent if parent is not None else current_span()
        if parent is not None and not parent.recording:
            return NOOP_SPAN
        span_id = str(uuid.uuid4())
        record = SpanRecord(
            name=name,
            kind=kind,
            trace_id=parent.trace_id if parent is not None else span_id,
            span_id=span_id,
            parent_id=parent.record.span_id if parent is not None else None,
            start_time=datetime.now(UTC),
            inputs=compact(dict(inputs or {})),
            attributes=compact(dict(attributes or {})),
        )
        parent_handle = parent.handle if parent is not None else None
        handle = self._safely("start", self.backend.start, record, parent_handle)
        return Span(self, record, parent, handle)

    def export(self, span: Span) -> None:
        if self.backend is not None:
            self._safely("end", self.backend.end, span.record, span.handle)

    def flush(self) -> None:
        if self.backend is not None:
            self._safely("flush", self.backend.flush)

    def _safely(self, what: str, call: Any, *args: Any) -> Any:
        """Run a backend call; an error is logged (in full once) and swallowed."""
        try:
            return call(*args)
        except Exception as exc:
            if not self._warned:
                self._warned = True
                logger.warning(
                    "tracing backend %s failed on %s; spans may be missing: %s",
                    type(self.backend).__name__,
                    what,
                    exc,
                    exc_info=True,
                )
            else:
                logger.debug("tracing backend failed on %s: %s", what, exc)
            return None


def build_tracer(settings: Settings | None = None) -> Tracer:
    """The tracer the settings ask for: ``jsonl``, or ``langsmith`` when a LangSmith key
    is set; otherwise tracing is off."""
    settings = settings or get_settings()
    if settings.tracing_backend == "jsonl":
        return Tracer(JsonlBackend(repo_path(settings.tracing_path)))
    if settings.langsmith_api_key:
        return Tracer(
            LangSmithBackend(settings.langsmith_project, api_key=settings.langsmith_api_key)
        )
    return Tracer()


_tracer: Tracer | None = None
_tracer_lock = threading.Lock()


def get_tracer() -> Tracer:
    """The process's tracer, built from the settings on first use."""
    global _tracer
    if _tracer is None:
        with _tracer_lock:
            if _tracer is None:
                _tracer = build_tracer()
    return _tracer


def set_tracer(tracer: Tracer | None) -> None:
    """Use ``tracer`` from now on (``None``: rebuild from the settings on next use)."""
    global _tracer
    with _tracer_lock:
        _tracer = tracer


@contextmanager
def span(
    name: str,
    *,
    kind: SpanKind = "chain",
    inputs: Mapping[str, Any] | None = None,
    attributes: Mapping[str, Any] | None = None,
    expected: tuple[type[BaseException], ...] = (),
) -> Iterator[Span]:
    """Open a span for the block, current for the spans opened inside it.

    An exception leaving the block is recorded as the span's error and re-raised;
    ``expected`` exceptions (control flow, such as a graph interrupt) end it cleanly.
    """
    opened = get_tracer().start_span(name, kind=kind, inputs=inputs, attributes=attributes)
    token = _current.set(opened)
    try:
        yield opened
    except GeneratorExit:  # the consumer stopped early (a client went away)
        opened.set("cancelled", True)
        raise
    except BaseException as exc:
        if not isinstance(exc, expected):
            opened.record_error(exc)
        raise
    finally:
        try:
            _current.reset(token)
        except ValueError:  # ended in another context (an async generator moved tasks)
            _current.set(opened.parent if opened.recording else None)
        opened.end()


# --------------------------------------------------------------------------- viewing


def read_spans(path: Path) -> list[SpanRecord]:
    """Spans a :class:`JsonlBackend` wrote, in the order they ended."""
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [SpanRecord.from_dict(json.loads(line)) for line in lines if line.strip()]


def format_tree(records: Sequence[SpanRecord]) -> str:
    """One trace's spans as an indented tree, children in start order."""
    children: dict[str | None, list[SpanRecord]] = {}
    ids = {r.span_id for r in records}
    for record in sorted(records, key=lambda r: r.start_time):
        parent = record.parent_id if record.parent_id in ids else None
        children.setdefault(parent, []).append(record)
    lines: list[str] = []

    def walk(parent: str | None, depth: int) -> None:
        for record in children.get(parent, []):
            took = f"{record.duration_ms:.0f} ms" if record.duration_ms is not None else "open"
            parts = [f"{record.name} [{record.kind}] {took}"]
            parts += [f"{k}={v}" for k, v in record.attributes.items()]
            if record.error:
                parts.append(f"ERROR {record.error}")
            lines.append("  " * depth + " ".join(parts))
            walk(record.span_id, depth + 1)

    walk(None, 0)
    return "\n".join(line.rstrip() for line in lines)


def last_trace(records: Sequence[SpanRecord], trace_id: str | None = None) -> list[SpanRecord]:
    """The spans of ``trace_id`` (default: the trace whose root ended last)."""
    if trace_id is None:
        roots = [r for r in records if r.parent_id is None]
        if not roots:
            return []
        trace_id = roots[-1].trace_id
    return [r for r in records if r.trace_id == trace_id]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Show traces the jsonl backend wrote.")
    sub = parser.add_subparsers(dest="command", required=True)
    show = sub.add_parser("show", help="print one trace as a tree (default: the last)")
    show.add_argument("--trace", help="trace id (default: the last finished trace)")
    show.add_argument("--path", type=Path, help="spans file (default: TRACING_PATH)")
    args = parser.parse_args(argv)
    path = args.path or repo_path(get_settings().tracing_path)
    spans = last_trace(read_spans(path), args.trace)
    if not spans:
        print(f"no trace found in {path}")
        return 1
    print(f"trace {spans[0].trace_id} ({len(spans)} spans)")
    print(format_tree(spans))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
