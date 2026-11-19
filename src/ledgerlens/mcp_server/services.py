"""What the MCP tools read from: the live stores, built on first use and shared by every
tool call of one server.

Nothing connects at import: the retriever loads its models on the first search, the
Neo4j driver opens on the first graph query. Loading the embedding and reranking models
takes tens of seconds on a CPU, longer than some clients wait for a tool call, so the
server calls :meth:`Services.warm_up` when it starts: one search in a background thread,
which the first real search waits for instead of loading the models a second time. Tests
hand :class:`Services` stand-ins instead (``Services(retriever=..., graph=...)``).
"""

from __future__ import annotations

import logging
import threading
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


logger = logging.getLogger(__name__)

WARM_UP_TIMEOUT_S = 300.0


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
        self._warm = threading.Event()
        self._warm.set()  # nothing to wait for until warm_up() starts

    def warm_up(self) -> threading.Thread:
        """Build the retriever and run one search in a background thread, so the
        models are loaded before the first tool call needs them."""
        self._warm.clear()

        def run() -> None:
            try:
                self.retriever.retrieve("warm-up", k=1)
                logger.info("retriever warmed up")
            except Exception:
                logger.warning(
                    "retriever warm-up failed; the first search will retry", exc_info=True
                )
            finally:
                self._warm.set()

        thread = threading.Thread(target=run, name="mcp-warm-up", daemon=True)
        thread.start()
        return thread

    def ready_retriever(self) -> Retriever:
        """The retriever, once a warm-up in progress has finished."""
        self._warm.wait(WARM_UP_TIMEOUT_S)
        return self.retriever

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
