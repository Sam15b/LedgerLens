"""CypherGuard against the compose Neo4j: writes never execute, totals come out right.

A tiny ledger is created per test for vendor ``V-<tag>`` and deleted afterwards:

    INV-1  100.00 GBP  paid 100.00      (2026-01-10)
    INV-2  250.00 GBP  paid  50.00      (2026-02-10)
    INV-3   80.00 EUR  no payment       (2026-02-20)
"""

import uuid
from collections.abc import Iterator

import pytest
from neo4j import Driver
from neo4j.exceptions import ClientError, CypherSyntaxError

from ledgerlens.retrieval import text2cypher
from ledgerlens.retrieval.text2cypher import CypherGuard, CypherRejectedError
from ledgerlens.storage.neo4j.driver import get_driver, run_read, run_write

pytestmark = pytest.mark.integration

SEED = """
CREATE (v:Vendor {id: $vendor, name: 'Guard Test Ltd'})
CREATE (v)-[:ISSUED]->(i1:Invoice {id: $tag + '-1', invoice_number: 'INV-1-' + $tag,
  vendor_id: $vendor, currency: 'GBP', total: 100.0, invoice_date: date('2026-01-10')})
CREATE (v)-[:ISSUED]->(i2:Invoice {id: $tag + '-2', invoice_number: 'INV-2-' + $tag,
  vendor_id: $vendor, currency: 'GBP', total: 250.0, invoice_date: date('2026-02-10')})
CREATE (v)-[:ISSUED]->(:Invoice {id: $tag + '-3', invoice_number: 'INV-3-' + $tag,
  vendor_id: $vendor, currency: 'EUR', total: 80.0, invoice_date: date('2026-02-20')})
CREATE (:Payment {id: $tag + '-p1', vendor_id: $vendor})-[:SETTLES {amount: 100.0}]->(i1)
CREATE (:Payment {id: $tag + '-p2', vendor_id: $vendor})-[:SETTLES {amount: 50.0}]->(i2)
"""


@pytest.fixture(scope="module")
def driver() -> Driver:
    return get_driver()


@pytest.fixture
def vendor(driver: Driver) -> Iterator[str]:
    tag = f"g{uuid.uuid4().hex[:10]}"
    vendor_id = f"V-{tag}"
    run_write(driver, SEED, {"tag": tag, "vendor": vendor_id})
    yield vendor_id
    run_write(driver, "MATCH (n) WHERE n.id CONTAINS $tag DETACH DELETE n", {"tag": tag})


@pytest.fixture
def guard(driver: Driver) -> CypherGuard:
    return CypherGuard(driver, timeout_s=5.0)


def snapshot(driver: Driver, vendor: str) -> list[dict[str, object]]:
    return run_read(
        driver,
        "MATCH (n) WHERE n.id STARTS WITH $prefix OR n.vendor_id = $vendor "
        "RETURN n.id AS id, properties(n) AS props ORDER BY id",
        {"prefix": vendor.removeprefix("V-"), "vendor": vendor},
    )


WRITES = [
    "MATCH (i:Invoice {vendor_id: '{v}'}) SET i.total = 0 RETURN i",
    "MATCH (i:Invoice {vendor_id: '{v}'}) DETACH DELETE i",
    "MATCH (v:Vendor {id: '{v}'}) CREATE (v)-[:ISSUED]->(:Invoice {id: 'evil'}) RETURN v",
    "MERGE (v:Vendor {id: '{v}'}) SET v.name = 'pwned' RETURN v",
    "MATCH (v:Vendor {id: '{v}'}) REMOVE v.name RETURN v",
    "MATCH (i:Invoice {vendor_id: '{v}'}) RETURN i; MATCH (n {vendor_id: '{v}'}) DETACH DELETE n",
    "CALL apoc.periodic.iterate('MATCH (n) RETURN n', 'DETACH DELETE n', {})",
    "MATCH (i:Invoice {vendor_id: '{v}'}) CALL (i) { SET i.total = 0 } RETURN i",
]


@pytest.mark.parametrize("template", WRITES)
def test_no_write_query_executes(
    driver: Driver, guard: CypherGuard, vendor: str, template: str
) -> None:
    before = snapshot(driver, vendor)

    with pytest.raises(CypherRejectedError):
        guard.run(template.replace("{v}", vendor))

    assert snapshot(driver, vendor) == before


def test_neo4j_classification_stops_a_write_the_scan_missed(
    driver: Driver, guard: CypherGuard, vendor: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Second line of defence: with the keyword scan switched off, EXPLAIN still refuses."""
    monkeypatch.setattr(text2cypher, "check_cypher", lambda cypher, **_: cypher)
    before = snapshot(driver, vendor)

    with pytest.raises(CypherRejectedError, match="not read-only"):
        guard.run(f"MATCH (v:Vendor {{id: '{vendor}'}}) SET v.name = 'pwned' RETURN v")

    assert snapshot(driver, vendor) == before


def test_aggregation_totals_are_correct(guard: CypherGuard, vendor: str) -> None:
    cypher = f"""
    MATCH (v:Vendor {{id: '{vendor}'}})-[:ISSUED]->(i:Invoice)
    WHERE i.invoice_date >= date('2026-01-01') AND i.invoice_date <= date('2026-03-31')
    OPTIONAL MATCH (:Payment)-[s:SETTLES]->(i)
    WITH i, i.total - coalesce(sum(s.amount), 0.0) AS open
    WHERE open >= 0.01
    RETURN i.currency AS currency, count(i) AS unpaid_invoices, round(sum(open), 2) AS unpaid
    ORDER BY currency
    """

    run = guard.run(cypher)

    assert run.cypher.endswith("LIMIT 50")
    assert run.rows == [
        {"currency": "EUR", "unpaid_invoices": 1, "unpaid": 80.0},
        {"currency": "GBP", "unpaid_invoices": 1, "unpaid": 200.0},
    ]


def test_row_cap_applies(driver: Driver, vendor: str) -> None:
    guard = CypherGuard(driver, max_rows=2)

    run = guard.run(f"MATCH (i:Invoice {{vendor_id: '{vendor}'}}) RETURN i.id AS id LIMIT 40")

    assert run.cypher.endswith("LIMIT 2") and len(run.rows) == 2


def test_slow_queries_time_out(driver: Driver) -> None:
    guard = CypherGuard(driver, timeout_s=0.5)

    with pytest.raises(ClientError, match="(?i)timed ?out|terminated"):
        guard.run("UNWIND range(1, 500000000) AS x WITH x WHERE x % 7 = 0 RETURN count(x) AS n")


def test_syntax_errors_surface_for_the_retry(guard: CypherGuard) -> None:
    with pytest.raises(CypherSyntaxError):
        guard.run("MATCH (n:Invoice) WHERE n.total > RETURN n")
