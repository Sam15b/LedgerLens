"""Text2Cypher: an LLM writes Cypher for questions no template covers.

Aggregations ("total unpaid by vendor for Q1") and unusual multi-hop questions have too
many shapes for the parameterized templates in :mod:`ledgerlens.storage.neo4j.queries`.
:class:`CypherGenerator` asks a model (the ``smart`` alias) for one read-only query, with
the graph schema and few-shot examples from the ``text2cypher`` prompt and the vendor ids
the query analyzer resolved as hints::

    generator = CypherGenerator(LLMClient())
    cypher = generator.generate("Total unpaid by vendor for Q1 2026?", analysis)

The generated text is untrusted: it only runs through :class:`CypherGuard` below.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field

from ledgerlens.core.prompts import load_prompt
from ledgerlens.llm.router import LLMClient
from ledgerlens.llm.types import Message
from ledgerlens.retrieval.analyze import QueryAnalysis

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
