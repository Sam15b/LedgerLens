"""Graph expansion: from seed documents to the facts around them in Neo4j.

Vector search finds the documents that *read* like the question; the answer to a
relationship question ("which payment settled INV-889?", "Acme invoices whose PO has no
payment") lies in documents linked to those. :class:`GraphExpander` takes seeds --
document ids (R3's top chunks) and document numbers from the question -- and runs a
parameterized template from :mod:`ledgerlens.storage.neo4j.queries` to collect them::

    expander = GraphExpander(get_driver())
    rows = expander.expand(["INV-889", "5a0aef53-..."])     # invoices within two hops
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from neo4j import Driver

from ledgerlens.storage.neo4j.driver import run_read
from ledgerlens.storage.neo4j.queries import INVOICE_NEIGHBORHOOD, CypherTemplate

Row = dict[str, Any]

DEFAULT_PER_SEED = 10
DEFAULT_LIMIT = 30


def native(value: Any) -> Any:
    """``value`` with Neo4j temporal types (``neo4j.time.Date``, ...) turned into their
    ``datetime`` equivalents, inside lists and maps too."""
    if isinstance(value, Mapping):
        return {key: native(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [native(item) for item in value]
    to_native = getattr(value, "to_native", None)
    return to_native() if callable(to_native) else value


class TemplateRunner(Protocol):
    """Runs a Cypher template (the Neo4j driver in production, a stub in unit tests)."""

    def run(self, template: CypherTemplate, params: Mapping[str, Any]) -> list[Row]: ...


class Neo4jRunner:
    """:class:`TemplateRunner` on a Neo4j driver: a read transaction per query."""

    def __init__(self, driver: Driver, *, database: str | None = None) -> None:
        self.driver = driver
        self.database = database

    def run(self, template: CypherTemplate, params: Mapping[str, Any]) -> list[Row]:
        rows = run_read(self.driver, template.cypher, params, database=self.database)
        return [native(row) for row in rows]


def unique(values: Sequence[str | None]) -> list[str]:
    """``values`` without blanks and repeats, in order."""
    return list(dict.fromkeys(value for value in values if value))


class GraphExpander:
    """Expands seed documents through the knowledge graph."""

    def __init__(
        self,
        driver: Driver | None = None,
        *,
        runner: TemplateRunner | None = None,
        database: str | None = None,
        per_seed: int = DEFAULT_PER_SEED,
        limit: int = DEFAULT_LIMIT,
    ) -> None:
        if runner is None:
            if driver is None:
                raise ValueError("GraphExpander needs a Neo4j driver or a runner")
            runner = Neo4jRunner(driver, database=database)
        self.runner = runner
        self.per_seed = per_seed
        self.limit = limit

    def rows(self, template: CypherTemplate, **params: Any) -> list[Row]:
        """Run one template (unknown or missing parameters are an error)."""
        return self.runner.run(template, template.bind(**params))

    def expand(self, seeds: Sequence[str | None]) -> list[Row]:
        """Invoices within two hops of the seeds -- document ids or document numbers, best
        first -- with their vendor, PO, contract and payments (``invoice_neighborhood``)."""
        wanted = unique(seeds)
        if not wanted:
            return []
        return self.rows(
            INVOICE_NEIGHBORHOOD, seeds=wanted, per_seed=self.per_seed, limit=self.limit
        )
