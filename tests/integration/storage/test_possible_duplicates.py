"""Near-duplicate hits become POSSIBLE_DUPLICATE edges in Neo4j (tagged test nodes only)."""

import uuid
from collections.abc import Iterator

import pytest
from neo4j import Driver

from ledgerlens.anomaly.rules import RuleHit, link_possible_duplicates
from ledgerlens.storage.neo4j.driver import apply_schema, get_driver, run_read, run_write

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def driver() -> Driver:
    driver = get_driver()
    apply_schema(driver)
    return driver


@pytest.fixture
def tag(driver: Driver) -> Iterator[str]:
    tag = f"t{uuid.uuid4().hex[:10]}"
    yield tag
    run_write(driver, "MATCH (n) WHERE n.id CONTAINS $tag DETACH DELETE n", {"tag": tag})


def test_near_duplicate_hits_become_edges_once(driver: Driver, tag: str) -> None:
    run_write(
        driver,
        "UNWIND $ids AS id CREATE (:Invoice {id: id})",
        {"ids": [f"{tag}-a", f"{tag}-b"]},
    )
    hit = RuleHit(
        rule="near_duplicate",
        invoice_id=f"{tag}-b",
        reason="looks like INV-1 re-sent",
        related_ids=[f"{tag}-a", f"{tag}-missing"],
        details={
            "other_number": "INV-1",
            "amount_gap": 0.0,
            "days_apart": 2,
            "number_similarity": 0.9,
        },
    )
    other = RuleHit(
        rule="exact_duplicate", invoice_id=f"{tag}-b", reason="x", related_ids=[f"{tag}-a"]
    )

    assert link_possible_duplicates(driver, [hit, other]) == 1  # the missing node is skipped
    assert link_possible_duplicates(driver, [hit]) == 1  # MERGE: still one edge

    edges = run_read(
        driver,
        "MATCH (a:Invoice)-[r:POSSIBLE_DUPLICATE]->(b:Invoice) WHERE a.id CONTAINS $tag "
        "RETURN a.id AS a, b.id AS b, r.days_apart AS days, r.reason AS reason",
        {"tag": tag},
    )
    assert edges == [
        {"a": f"{tag}-b", "b": f"{tag}-a", "days": 2, "reason": "looks like INV-1 re-sent"}
    ]
    assert link_possible_duplicates(driver, [other]) == 0
