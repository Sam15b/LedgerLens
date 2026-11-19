"""``graph_query``: read the knowledge graph, by named template or by question.

* ``template`` + ``params`` runs one of the reviewed read-only Cypher templates
  (:data:`ledgerlens.storage.neo4j.queries.TEMPLATES`): ``invoice_neighborhood``,
  ``unpaid_with_po``, ``vendor_ledger``, ``contract_for_po``. Parameters a template does
  not get fall back to its defaults (and ``as_of`` to today). ``vendor_ids`` may hold
  vendor names; they are resolved like ``search_documents``' ``vendor``. Rows come back
  with their facts rendered as text and citation tags.
* ``question`` asks Text2Cypher (the ``smart`` model) to write the query. It runs through
  :class:`~ledgerlens.retrieval.text2cypher.CypherGuard` (read only, row cap, timeout); if
  it is rejected or fails, the model sees the error and gets one retry.

Exactly one of the two is required.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from pydantic import BaseModel, Field

from ledgerlens.llm.types import LLMError
from ledgerlens.mcp_server.services import Services
from ledgerlens.mcp_server.tools import READ_ONLY, in_thread, jsonable
from ledgerlens.mcp_server.tools.search import resolve_vendor
from ledgerlens.retrieval.graph import render
from ledgerlens.storage.neo4j.queries import TEMPLATES, CypherTemplate

NUMBER_KEYS = ("invoice_number", "po_number", "contract_number", "payment_reference")


class GraphQueryResult(BaseModel):
    source: str
    """``template:<name>`` or ``text2cypher``."""
    cypher: str | None = None
    """The query Text2Cypher wrote and ran (``None`` for templates)."""
    params: dict[str, Any] = Field(default_factory=dict)
    rows: list[dict[str, Any]] = Field(default_factory=list)
    facts: list[str] = Field(default_factory=list)
    """Template rows rendered as one-line facts."""
    citations: dict[str, str] = Field(default_factory=dict)
    """Citation tag -> document id, for documents the rows name."""


def describe_templates() -> str:
    return "\n".join(
        f"- {t.name}({', '.join(t.params)}): {t.description}" for t in TEMPLATES.values()
    )


def row_citations(rows: Any) -> dict[str, str]:
    """``doc:<number>`` -> document id for every map in ``rows`` (at any depth) with a
    ``document_id`` and a document number."""
    found: dict[str, str] = {}

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            document_id = value.get("document_id")
            if document_id:
                number = next((value[k] for k in NUMBER_KEYS if value.get(k)), None)
                found.setdefault(f"doc:{number or str(document_id)[:8]}", str(document_id))
            for item in value.values():
                walk(item)
        elif isinstance(value, list | tuple):
            for item in value:
                walk(item)

    walk(rows)
    return found


def run_template(services: Services, name: str, params: Mapping[str, Any]) -> GraphQueryResult:
    template: CypherTemplate | None = TEMPLATES.get(name)
    if template is None:
        raise ToolError(f"unknown template {name!r}; choose from {', '.join(TEMPLATES)}")
    values = dict(params)
    if "as_of" in template.params and "as_of" not in values:
        values["as_of"] = services.today.isoformat()
    if values.get("vendor_ids"):
        values["vendor_ids"] = [
            vendor_id
            for given in values["vendor_ids"]
            for vendor_id in resolve_vendor(services, given)
        ]
    try:
        bound = template.bind(**values)
    except ValueError as exc:
        raise ToolError(str(exc)) from exc
    rows = services.graph.run(template, bound)
    facts = render(template, rows)
    citations = row_citations(rows)
    for fact in facts:
        citations.update(fact.citations)
    return GraphQueryResult(
        source=f"template:{template.name}",
        params=jsonable(bound),
        rows=jsonable(rows),
        facts=[fact.text for fact in facts],
        citations=citations,
    )


def run_question(services: Services, question: str) -> GraphQueryResult:
    from ledgerlens.retrieval.text2cypher import CypherRejectedError

    try:
        cypher = services.cypher_generator.generate(question)
        try:
            run = services.cypher_guard.run(cypher)
        except Exception as first:  # rejected, or Neo4j refused it: one corrected retry
            error = f"{type(first).__name__}: {first}"
            cypher = services.cypher_generator.generate(question, previous=cypher, error=error)
            run = services.cypher_guard.run(cypher)
    except LLMError as exc:
        raise ToolError(f"Text2Cypher is unavailable ({exc}); use a template") from exc
    except CypherRejectedError as exc:
        raise ToolError(f"the generated query was rejected: {exc}") from exc
    except Exception as exc:
        raise ToolError(f"the generated query failed: {type(exc).__name__}: {exc}") from exc
    rows = jsonable(run.rows)
    return GraphQueryResult(
        source="text2cypher", cypher=run.cypher, rows=rows, citations=row_citations(rows)
    )


def graph_query(
    services: Services,
    *,
    template: str | None = None,
    params: Mapping[str, Any] | None = None,
    question: str | None = None,
) -> GraphQueryResult:
    if (template is None) == (question is None):
        raise ToolError("give either a template (with params) or a question, not both")
    if template is not None:
        return run_template(services, template, params or {})
    assert question is not None
    if params:
        raise ToolError("params go with a template, not with a question")
    return run_question(services, question)


def register(server: FastMCP, services: Services) -> None:
    @server.tool(
        name="graph_query",
        annotations=READ_ONLY,
        description=(
            "Read the accounts-payable knowledge graph (vendors, invoices, POs, contracts, "
            "payments and how they link). Either run a named template with params, or ask "
            "a question and let Text2Cypher write a read-only query. Templates:\n"
            + describe_templates()
            + "\nvendor_ids may be vendor names; seeds are document ids or numbers "
            "(INV-889, PO-1042)."
        ),
    )
    async def graph_query_tool(
        template: Annotated[str | None, Field(description="A template name (see above)")] = None,
        params: Annotated[
            dict[str, Any] | None, Field(description="The template's parameters")
        ] = None,
        question: Annotated[
            str | None, Field(description="A question for Text2Cypher, instead of a template")
        ] = None,
    ) -> GraphQueryResult:
        return await in_thread(
            graph_query, services, template=template, params=params, question=question
        )
