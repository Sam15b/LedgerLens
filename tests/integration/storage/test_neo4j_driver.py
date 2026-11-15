"""Neo4j sessions and the read-only guard against the compose Neo4j."""

import uuid
from collections.abc import Iterator

import pytest
from neo4j import Driver

from ledgerlens.storage.neo4j.driver import (
    WriteQueryError,
    get_driver,
    query_type,
    run_read,
    run_read_only,
    run_write,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def driver() -> Iterator[Driver]:
    driver = get_driver()
    yield driver
    run_write(driver, "MATCH (n:ItProbe) DETACH DELETE n")


def test_write_then_read_returns_plain_dicts(driver: Driver) -> None:
    key = uuid.uuid4().hex
    run_write(driver, "CREATE (:ItProbe {key: $key, n: 1})", {"key": key})

    rows = run_read(
        driver, "MATCH (p:ItProbe {key: $key}) RETURN p.key AS key, p.n AS n", {"key": key}
    )

    assert rows == [{"key": key, "n": 1}]


@pytest.mark.parametrize(
    ("cypher", "kind"),
    [
        ("MATCH (n:ItProbe) RETURN count(n)", "r"),
        ("CREATE (:ItProbe)", "w"),
        ("MATCH (n:ItProbe) SET n.x = 1 RETURN n", "rw"),
    ],
)
def test_query_type_classifies_without_running(driver: Driver, cypher: str, kind: str) -> None:
    assert query_type(driver, cypher) == kind
    assert run_read(driver, "MATCH (n:ItProbe) RETURN count(n) AS n") == [{"n": 0}]


def test_read_only_runs_reads(driver: Driver) -> None:
    assert run_read_only(driver, "RETURN $x + 1 AS y", {"x": 41}) == [{"y": 42}]


@pytest.mark.parametrize(
    "cypher",
    [
        "CREATE (:ItProbe {sneaky: true})",
        "MATCH (n) DETACH DELETE n",
        "MERGE (:ItProbe {k: 1}) RETURN 1",
        "MATCH (n:ItProbe) SET n.x = 1 RETURN n",
    ],
)
def test_read_only_refuses_writes_before_running_them(driver: Driver, cypher: str) -> None:
    with pytest.raises(WriteQueryError, match="only read-only Cypher"):
        run_read_only(driver, cypher)

    assert run_read(driver, "MATCH (n:ItProbe) RETURN count(n) AS n") == [{"n": 0}]
