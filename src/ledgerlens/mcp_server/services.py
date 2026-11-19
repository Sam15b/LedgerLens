"""What the MCP tools read from: the live stores, built on first use and shared by every
tool call of one server.

Nothing connects at import or when the server starts: the retriever loads its models on
the first search, the Neo4j driver opens on the first graph query. Tests hand
:class:`Services` stand-ins instead (``Services(retriever=..., graph=...)``).
"""

from __future__ import annotations

from datetime import date
from functools import cached_property
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sqlalchemy.orm import Session, sessionmaker

    from ledgerlens.llm.router import LLMClient
    from ledgerlens.retrieval import Retriever
    from ledgerlens.retrieval.analyze import VendorFinder
    from ledgerlens.retrieval.graph import TemplateRunner
    from ledgerlens.retrieval.text2cypher import CypherGenerator, CypherGuard


class Services:
    """Lazily built clients for the tools; pass any of them to use it instead."""

    def __init__(
        self,
        *,
        retriever: Retriever | None = None,
        graph: TemplateRunner | None = None,
        cypher_generator: CypherGenerator | None = None,
        cypher_guard: CypherGuard | None = None,
        sessions: sessionmaker[Session] | None = None,
        llm: LLMClient | None = None,
        vendors: VendorFinder | None = None,
        today: date | None = None,
    ) -> None:
        given: dict[str, Any] = {
            "retriever": retriever,
            "graph": graph,
            "cypher_generator": cypher_generator,
            "cypher_guard": cypher_guard,
            "sessions": sessions,
            "llm": llm,
            "vendors": vendors,
        }
        for name, value in given.items():
            if value is not None:
                self.__dict__[name] = value  # pre-fills the cached_property
        self._today = today

    @property
    def today(self) -> date:
        return self._today or date.today()

    @cached_property
    def retriever(self) -> Retriever:
        """``retrieval.default`` (R4, GraphRAG) over the live stores."""
        from ledgerlens.retrieval.factory import build_retriever

        return build_retriever()

    @cached_property
    def driver(self) -> Any:
        from ledgerlens.storage.neo4j.driver import get_driver

        return get_driver()

    @cached_property
    def graph(self) -> TemplateRunner:
        from ledgerlens.retrieval.graph import Neo4jRunner

        return Neo4jRunner(self.driver)

    @cached_property
    def cypher_guard(self) -> CypherGuard:
        from ledgerlens.retrieval.text2cypher import CypherGuard

        return CypherGuard(self.driver)

    @cached_property
    def llm(self) -> LLMClient:
        from ledgerlens.llm.router import LLMClient

        return LLMClient()

    @cached_property
    def cypher_generator(self) -> CypherGenerator:
        from ledgerlens.retrieval.text2cypher import CypherGenerator

        return CypherGenerator(self.llm, today=self._today)

    @cached_property
    def sessions(self) -> sessionmaker[Session]:
        from ledgerlens.storage.postgres.db import get_session_factory

        return get_session_factory()

    @cached_property
    def vendors(self) -> VendorFinder:
        """The vendor master's names and aliases, for vendor arguments given by name."""
        from ledgerlens.retrieval.analyze import QueryAnalyzer

        return QueryAnalyzer.from_postgres().finder
