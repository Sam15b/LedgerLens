import json
import logging
from collections.abc import Iterator

import pytest

from ledgerlens.core.logging import (
    JsonFormatter,
    RequestIdFilter,
    configure_logging,
    get_request_id,
    request_context,
)


@pytest.fixture(autouse=True)
def _restore_root_logger() -> Iterator[None]:
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)


def _record(msg: str = "hello", **extra: object) -> logging.LogRecord:
    record = logging.LogRecord("ledgerlens.test", logging.INFO, __file__, 1, msg, None, None)
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_request_context_sets_and_resets() -> None:
    assert get_request_id() is None
    with request_context("req-1"):
        assert get_request_id() == "req-1"
        with request_context("req-2"):
            assert get_request_id() == "req-2"
        assert get_request_id() == "req-1"
    assert get_request_id() is None


def test_filter_adds_request_id_or_dash() -> None:
    record = _record()
    RequestIdFilter().filter(record)
    assert record.__dict__["request_id"] == "-"

    with request_context("abc"):
        record = _record()
        RequestIdFilter().filter(record)
    assert record.__dict__["request_id"] == "abc"


def test_json_formatter_includes_core_fields_and_extras() -> None:
    with request_context("req-9"):
        record = _record("ingested", document_id="d1", pages=3)
        RequestIdFilter().filter(record)
        payload = json.loads(JsonFormatter().format(record))

    assert payload["message"] == "ingested"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "ledgerlens.test"
    assert payload["request_id"] == "req-9"
    assert payload["document_id"] == "d1"
    assert payload["pages"] == 3


def test_json_formatter_serializes_exceptions() -> None:
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        import sys

        record = _record("failed")
        record.exc_info = sys.exc_info()

    payload = json.loads(JsonFormatter().format(record))

    assert "RuntimeError: boom" in payload["exc_info"]


def test_configure_logging_is_idempotent() -> None:
    configure_logging("DEBUG", json_logs=True)
    configure_logging("WARNING", json_logs=False)

    root = logging.getLogger()
    ours = [h for h in root.handlers if h.get_name() == "ledgerlens"]

    assert len(ours) == 1
    assert root.level == logging.WARNING
    assert not isinstance(ours[0].formatter, JsonFormatter)


def test_configure_logging_emits_json(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO", json_logs=True)

    with request_context("req-out"):
        logging.getLogger("ledgerlens.x").info("done", extra={"count": 2})

    line = capsys.readouterr().err.strip().splitlines()[-1]
    payload = json.loads(line)
    assert payload == payload | {"message": "done", "request_id": "req-out", "count": 2}
