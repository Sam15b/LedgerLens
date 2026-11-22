"""``search_documents``: semantic + keyword search over the ledger's documents.

It runs the default retriever (``retrieval.default``: R4, hybrid search reranked and
expanded through the knowledge graph) with filters from the arguments. A vendor can be
given by master id (``V001``) or by name ("Acme", "Acme Supplies Ltd"); a name that
matches no vendor is an error rather than a silent search over every vendor.

Each hit carries its citation tag (``doc:INV-889``) and the document id, so an answer can
cite what it used; graph facts (R4) come back as text with their own citations.
"""

from __future__ import annotations

from datetime import date
from typing import Annotated, Any, Literal

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from pydantic import BaseModel, Field, ValidationError

from ledgerlens.core.schemas import DocType
from ledgerlens.mcp_server.services import Services
from ledgerlens.mcp_server.tools import READ_ONLY, in_thread, jsonable
from ledgerlens.observability.tracing import retrieval_attributes, span
from ledgerlens.retrieval import QueryFilters

SearchDocType = Literal["invoice", "purchase_order", "contract", "payment", "receipt"]
MAX_K = 20


class SearchHit(BaseModel):
    """One retrieved chunk."""

    chunk_id: str = ""
    """The chunk's id, ``{document_id}#c{n}``."""
    citation: str
    """Cite it as this tag: ``doc:<document number>`` (or ``doc:<id prefix>``)."""
    document_id: str
    doc_type: str | None = None
    doc_number: str | None = None
    vendor_id: str | None = None
    page: int | None = None
    score: float
    text: str


class SearchResult(BaseModel):
    query: str
    retriever: str = ""
    """The retriever that answered (``r4_graphrag``, ...)."""
    filters: dict[str, Any] = Field(default_factory=dict)
    """The filters applied (vendor names resolved to ids)."""
    hits: list[SearchHit] = Field(default_factory=list)
    facts: list[str] = Field(default_factory=list)
    """Facts from the knowledge graph about the hits' records."""
    citations: dict[str, str] = Field(default_factory=dict)
    """Citation tag -> document id, for every hit and fact."""
    document_ids: list[str] = Field(default_factory=list)


def resolve_vendor(services: Services, vendor: str) -> tuple[str, ...]:
    """Vendor ids for a master id or a name (canonical name, alias, distinctive first
    word, or a near spelling)."""
    finder = services.vendors
    if vendor in finder.names:
        return (vendor,)
    found = finder.find(vendor)
    if not found:
        raise ToolError(f"no vendor in the vendor master matches {vendor!r}")
    return tuple(found)


def search_documents(
    services: Services,
    query: str,
    *,
    doc_type: SearchDocType | None = None,
    vendor: str | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
    k: int = 5,
) -> SearchResult:
    if not query.strip():
        raise ToolError("query is empty")
    if not 1 <= k <= MAX_K:
        raise ToolError(f"k must be between 1 and {MAX_K}")
    try:
        filters = QueryFilters(
            doc_types=(DocType(doc_type),) if doc_type else (),
            vendor_ids=resolve_vendor(services, vendor) if vendor else (),
            date_from=date_from,
            date_to=date_to,
        )
    except ValidationError as exc:
        raise ToolError(f"bad filters: {exc.errors()[0]['msg']}") from exc
    with span("retrieve", kind="retriever", inputs={"query": query, "k": k}) as traced:
        result = services.ready_retriever().retrieve(query, filters=filters, k=k)
        traced.set_attributes(
            retrieval_attributes(result.retriever, [c.id for c in result.chunks], k=k)
        )
    return SearchResult(
        query=query,
        retriever=result.retriever,
        filters=jsonable(filters.model_dump(exclude_defaults=True)),
        hits=[
            SearchHit(
                chunk_id=chunk.id,
                citation=chunk.citation,
                document_id=chunk.document_id,
                doc_type=chunk.doc_type,
                doc_number=chunk.doc_number,
                vendor_id=chunk.vendor_id,
                page=chunk.page,
                score=round(chunk.score, 4),
                text=chunk.text,
            )
            for chunk in result.chunks
        ],
        facts=[fact.text for fact in result.graph_facts],
        citations=result.citations,
        document_ids=result.document_ids,
    )


def register(server: FastMCP, services: Services) -> None:
    @server.tool(name="search_documents", annotations=READ_ONLY)
    async def search_documents_tool(
        query: Annotated[str, Field(description="What to look for, in plain words")],
        doc_type: Annotated[
            SearchDocType | None, Field(description="Only documents of this type")
        ] = None,
        vendor: Annotated[
            str | None, Field(description="Only this vendor: a master id (V001) or a name")
        ] = None,
        date_from: Annotated[
            date | None, Field(description="Documents dated on or after (YYYY-MM-DD)")
        ] = None,
        date_to: Annotated[
            date | None, Field(description="Documents dated on or before (YYYY-MM-DD)")
        ] = None,
        k: Annotated[int, Field(ge=1, le=MAX_K, description="How many chunks")] = 5,
    ) -> SearchResult:
        """Search invoices, purchase orders, contracts and payments by meaning and by
        keyword. Returns the best matching chunks with citation tags (cite them as
        doc:<number>), plus knowledge-graph facts about the records they belong to."""
        return await in_thread(
            search_documents,
            services,
            query,
            doc_type=doc_type,
            vendor=vendor,
            date_from=date_from,
            date_to=date_to,
            k=k,
        )
