"""Text2Cypher: an LLM writes Cypher for questions no template covers.

Aggregations ("total unpaid by vendor for Q1") and unusual multi-hop questions have too
many shapes for the parameterized templates in :mod:`ledgerlens.storage.neo4j.queries`.
:class:`CypherGenerator` asks a model (the ``smart`` alias) for one read-only query, with
the graph schema and few-shot examples from the ``text2cypher`` prompt and the vendor ids
the query analyzer resolved as hints::

    generator = CypherGenerator(LLMClient())
    cypher = generator.generate("Total unpaid by vendor for Q1 2026?", analysis)

The generated text is untrusted: it only runs through :class:`CypherGuard`, which

1. **rejects** it before it reaches Neo4j if, outside string literals and comments, it has
   a write or admin clause (``CREATE``, ``MERGE``, ``SET``, ``DELETE``, ``REMOVE``,
   ``DROP``, ``FOREACH``, ``LOAD CSV``, ``... IN TRANSACTIONS``, ...), a procedure
   ``CALL`` (``CALL dbms...``, ``CALL db...``, ``CALL apoc...`` -- ``CALL { }``
   subqueries are fine), any ``apoc.`` / ``dbms.`` function, or a second statement;
2. **caps** it with ``LIMIT`` (adds one, or lowers a larger one, to ``max_rows``);
3. **EXPLAIN**s it and refuses anything Neo4j itself does not classify as read-only;
4. **runs** it in a READ transaction that the server aborts after ``timeout_s`` (5 s).

Any failure raises :class:`CypherRejectedError` (refused before running) or a Neo4j error
(syntax, timeout), whose message is what the generator is shown on its one retry.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from neo4j import Driver
from pydantic import BaseModel, ConfigDict, Field

from ledgerlens.core.prompts import load_prompt
from ledgerlens.llm.router import LLMClient
from ledgerlens.llm.types import Message
from ledgerlens.retrieval.analyze import QueryAnalysis
from ledgerlens.retrieval.graph import native
from ledgerlens.storage.neo4j.driver import query_type, run_read

PROMPT = "text2cypher"


class GeneratedCypher(BaseModel):
    """The model's answer (structured output)."""

    model_config = ConfigDict(extra="forbid")

    cypher: str = Field(description="One read-only Cypher query that answers the question")


def hints(analysis: QueryAnalysis | None) -> str:
    """What the analyzer already knows, for the prompt: vendor ids, document numbers and
    the date range (``"none"`` when nothing)."""
    if analysis is None:
        return "none"
    lines = [
        f"{name} = vendor id {vendor_id}"
        for vendor_id, name in zip(analysis.vendor_ids, analysis.vendor_names, strict=False)
    ]
    if analysis.doc_numbers:
        lines.append("document numbers: " + ", ".join(analysis.doc_numbers))
    if analysis.date_from or analysis.date_to:
        start = analysis.date_from.isoformat() if analysis.date_from else "the beginning"
        end = analysis.date_to.isoformat() if analysis.date_to else "today"
        lines.append(f"period: {start} to {end}")
    return "; ".join(lines) or "none"


def strip_fences(text: str) -> str:
    """The query without Markdown code fences or a trailing semicolon."""
    body = text.strip()
    if body.startswith("```"):
        body = body.split("\n", 1)[1] if "\n" in body else ""
        body = body.rsplit("```", 1)[0]
    return body.strip().rstrip(";").strip()


class CypherGenerator:
    """Writes Cypher for a question with an LLM; can be shown its last error to retry."""

    def __init__(self, llm: LLMClient, *, model: str = "smart", today: date | None = None) -> None:
        self.llm = llm
        self.model = model
        self._today = today
        self.prompt = load_prompt(PROMPT)

    @property
    def today(self) -> date:
        return self._today or date.today()

    def messages(self, question: str, analysis: QueryAnalysis | None = None) -> list[Message]:
        text = self.prompt.render(
            today=self.today.isoformat(), question=question, hints=hints(analysis)
        )
        return [Message.user(text)]

    def generate(
        self,
        question: str,
        analysis: QueryAnalysis | None = None,
        *,
        previous: str | None = None,
        error: str | None = None,
    ) -> str:
        """A Cypher query for ``question``. With ``previous`` and ``error``, the model sees
        the query it wrote and why it failed, and is asked for a corrected one."""
        messages = self.messages(question, analysis)
        if previous is not None and error is not None:
            messages += [
                Message.assistant(GeneratedCypher(cypher=previous).model_dump_json()),
                Message.user(
                    f"That query failed: {error}\n"
                    "Write a corrected query that follows every rule above."
                ),
            ]
        answer = self.llm.structured(messages, GeneratedCypher, model=self.model)
        return strip_fences(answer.cypher)


# --------------------------------------------------------------------------- guardrails

DEFAULT_MAX_ROWS = 50
DEFAULT_TIMEOUT_S = 5.0


class CypherRejectedError(ValueError):
    """Generated Cypher that may not run (a write, a procedure call, no RETURN, ...)."""


_LEXEME = re.compile(
    r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"|`[^`]*`|//[^\n]*|/\*.*?\*/",
    re.DOTALL,
)
"""String literals, backtick-quoted names and comments, in one pass (so ``'http://x'``
is one string, not a string cut short by a comment)."""
_FORBIDDEN = re.compile(
    r"\b(CREATE|MERGE|SET|DELETE|DETACH|REMOVE|DROP|FOREACH|LOAD\s+CSV|"
    r"IN\s+TRANSACTIONS|PERIODIC\s+COMMIT|GRANT|DENY|REVOKE|ALTER|RENAME|"
    r"START\s+DATABASE|STOP\s+DATABASE|TERMINATE|USE)\b",
    re.IGNORECASE,
)
_PROCEDURE = re.compile(r"\bCALL\s+(?![{(])", re.IGNORECASE)
_NAMESPACE = re.compile(r"\b(apoc|dbms|gds)\s*\.", re.IGNORECASE)
_RETURN = re.compile(r"\bRETURN\b", re.IGNORECASE)
_TRAILING_LIMIT = re.compile(r"\bLIMIT\s+(\S+)\s*$", re.IGNORECASE)


def _blank(match: re.Match[str]) -> str:
    lexeme = match[0]
    if lexeme[0] in "'\"":
        return "''"
    return "``" if lexeme[0] == "`" else " "


def code_only(cypher: str) -> str:
    """``cypher`` with comments, string literals and backtick-quoted names blanked out, so
    a vendor called "Set Solutions" or ``// delete`` cannot trip (or hide from) the scan."""
    return _LEXEME.sub(_blank, cypher)


def check_cypher(cypher: str, *, max_rows: int = DEFAULT_MAX_ROWS) -> str:
    """The query, safe to EXPLAIN and capped at ``max_rows`` rows; raises
    :class:`CypherRejectedError` naming the first problem."""
    query = strip_fences(cypher)
    if not query:
        raise CypherRejectedError("the query is empty")
    code = code_only(query)
    if ";" in code:
        raise CypherRejectedError("only one statement is allowed (found ';')")
    if match := _FORBIDDEN.search(code):
        raise CypherRejectedError(f"{match[1].upper()} is not allowed: the graph is read-only")
    if _PROCEDURE.search(code):
        raise CypherRejectedError("procedure calls are not allowed (CALL { } subqueries are)")
    if match := _NAMESPACE.search(code):
        raise CypherRejectedError(f"{match[1]}.* functions are not allowed")
    if not _RETURN.search(code):
        raise CypherRejectedError("the query must RETURN something")
    if limit := _TRAILING_LIMIT.search(query):
        value = limit[1]
        if value.isdigit() and int(value) <= max_rows:
            return query
        return f"{query[: limit.start()]}LIMIT {max_rows}"
    return f"{query}\nLIMIT {max_rows}"


@dataclass
class CypherRun:
    """A guarded run: the query as executed and its rows."""

    cypher: str
    rows: list[dict[str, Any]] = field(default_factory=list)


class CypherGuard:
    """Runs untrusted Cypher read-only, capped and with a timeout."""

    def __init__(
        self,
        driver: Driver,
        *,
        database: str | None = None,
        max_rows: int = DEFAULT_MAX_ROWS,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self.driver = driver
        self.database = database
        self.max_rows = max_rows
        self.timeout_s = timeout_s

    def run(self, cypher: str) -> CypherRun:
        query = check_cypher(cypher, max_rows=self.max_rows)
        kind = query_type(self.driver, query, database=self.database)  # EXPLAIN: plans only
        if kind != "r":
            raise CypherRejectedError(f"Neo4j classifies the query as {kind!r}, not read-only")
        rows = run_read(self.driver, query, database=self.database, timeout_s=self.timeout_s)
        return CypherRun(query, [native(row) for row in rows[: self.max_rows]])
