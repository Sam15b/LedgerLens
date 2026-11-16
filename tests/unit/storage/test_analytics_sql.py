"""analytics.sql is split into named queries (the queries themselves run in
tests/integration/storage/test_analytics.py)."""

import pytest

from ledgerlens.storage.postgres import analytics
from ledgerlens.storage.postgres.analytics import load_queries, parse_queries


def test_named_blocks_are_split_and_the_header_dropped() -> None:
    script = """-- header comment
-- name: first
SELECT 1;

-- name: second
-- what it does
SELECT :x
FROM t;
"""
    assert parse_queries(script) == {
        "first": "SELECT 1",
        "second": "-- what it does\nSELECT :x\nFROM t",
    }


def test_a_name_used_twice_is_an_error() -> None:
    with pytest.raises(ValueError, match="defined twice"):
        parse_queries("-- name: q\nSELECT 1;\n-- name: q\nSELECT 2;")


def test_every_packaged_query_has_a_typed_wrapper() -> None:
    queries = load_queries()
    assert queries, "analytics.sql has no named queries"
    for name, sql in queries.items():
        assert callable(getattr(analytics, name, None)), f"no wrapper for {name}"
        assert ";" not in sql, f"{name} holds more than one statement"
