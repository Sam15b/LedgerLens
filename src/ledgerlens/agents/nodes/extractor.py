"""The extractor node: read an uploaded (or named) document into a typed record and save it.

For each document the request is about (``document_ids`` from an upload, else document
numbers the router found), up to :data:`MAX_DOCUMENTS`:

1. ``get_document_text`` (MCP) -- the text as read at ingestion, and its type;
2. extract the record: the configured extractor (``extraction.default``: the ``llm``
   extractor through the ``extractor_api`` alias, or the fine-tuned one through
   ``extractor``); if no model answers, the rules extractor reads it instead;
3. ``save_extraction`` (MCP) -- validated and written like ingestion writes it (Postgres,
   the graph, an anomaly score for invoices).

The first saved record goes to ``extracted`` (with its document, record id and any
anomaly verdict); the graph then reconciles it if it is an invoice.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, BaseMessage

from ledgerlens.agents.state import LedgerState
from ledgerlens.agents.tools import ToolBox, ToolOutcome
from ledgerlens.core.schemas import DocType
from ledgerlens.extraction.base import ExtractionResult, Extractor

if TYPE_CHECKING:
    from ledgerlens.llm.router import LLMClient

logger = logging.getLogger(__name__)

EXTRACTOR_TOOLS = ("get_document_text", "save_extraction")
MAX_DOCUMENTS = 3
NUMBER_FIELDS = ("invoice_number", "po_number", "contract_number", "payment_reference")


def default_extractor(llm: LLMClient | None = None) -> Extractor:
    """``extraction.default`` from ``configs/ingestion.yaml`` (rules without a model)."""
    from ledgerlens.core.config import get_config
    from ledgerlens.ingestion.pipeline import make_extractor

    name = get_config().ingestion.extraction.default
    return make_extractor(name if llm is not None else "rules", llm=llm)


def _summary(record: dict[str, Any]) -> str:
    """``invoice INV-889 from Acme Supplies Ltd, 4,200.00 GBP, dated 2026-03-14, due ...``"""
    kind = str(record.get("doc_type", "record")).replace("_", " ")
    number = next((record[f] for f in NUMBER_FIELDS if record.get(f)), None)
    parts = [f"{kind} {number}" if number else kind]
    if record.get("vendor_name"):
        parts[0] += f" from {record['vendor_name']}"
    amount = record.get("total") or record.get("amount")
    if amount is not None:
        parts.append(f"{float(amount):,.2f} {record.get('currency', '')}".strip())
    for field, label in (
        ("invoice_date", "dated"),
        ("order_date", "ordered"),
        ("paid_at", "paid"),
        ("due_date", "due"),
        ("po_number", "PO"),
    ):
        if record.get(field) and not (field == "po_number" and record.get("doc_type") != "invoice"):
            parts.append(f"{label} {record[field]}")
    return ", ".join(parts)


class ExtractorNode:
    """Reads documents into records through ``get_document_text`` / ``save_extraction``."""

    def __init__(
        self,
        tools: ToolBox,
        extractor: Extractor | None = None,
        *,
        fallback: Extractor | None = None,
    ) -> None:
        from ledgerlens.extraction.rules import RulesExtractor

        self.tools = tools.only(*EXTRACTOR_TOOLS)
        self.extractor = extractor or RulesExtractor()
        self.fallback = fallback or RulesExtractor()

    async def extract(self, text: str, doc_type: DocType) -> ExtractionResult:
        try:
            result = await asyncio.to_thread(self.extractor.extract, text, doc_type)
        except Exception as exc:  # no model answered, or it could not be parsed
            if self.extractor is self.fallback:
                raise
            logger.warning("%s extractor failed (%s); using rules", self.extractor.name, exc)
            return await asyncio.to_thread(self.fallback.extract, text, doc_type)
        if result.record is None and self.extractor is not self.fallback:
            logger.warning("%s extractor found no record; trying rules", self.extractor.name)
            return await asyncio.to_thread(self.fallback.extract, text, doc_type)
        return result

    async def read(
        self, document: str
    ) -> tuple[list[ToolOutcome], list[BaseMessage], dict[str, Any] | str]:
        """One document: the tool outcomes and messages, and the saved extraction (or why
        there is none)."""
        outcomes: list[ToolOutcome] = []
        messages: list[BaseMessage] = []
        text_out, message = await self.tools.call("get_document_text", {"document": document})
        outcomes.append(text_out)
        messages.append(message)
        if not text_out.ok or text_out.data is None:
            return outcomes, messages, f"could not read {document}: {text_out.text}"
        data = text_out.data
        doc_type = DocType(data["doc_type"])
        if doc_type is DocType.UNKNOWN:
            from ledgerlens.ingestion.classify import classify

            doc_type = classify(data["text"]).doc_type
        try:
            result = await self.extract(data["text"], doc_type)
        except Exception as exc:
            return outcomes, messages, f"could not extract {document}: {type(exc).__name__}: {exc}"
        if result.record is None:
            missing = ", ".join(result.missing or result.errors) or "no fields found"
            return (
                outcomes,
                messages,
                f"could not read a {doc_type.value} from {document} ({missing})",
            )
        record = result.record.model_dump(mode="json")
        saved_out, message = await self.tools.call(
            "save_extraction", {"document_id": data["document_id"], "record": record}
        )
        outcomes.append(saved_out)
        messages.append(message)
        if not saved_out.ok or saved_out.data is None:
            return outcomes, messages, f"read {document} but saving failed: {saved_out.text}"
        saved = saved_out.data
        return (
            outcomes,
            messages,
            {
                **saved,
                "filename": data.get("filename"),
                "record": record,
                "extractor": result.extractor,
                "missing": result.missing,
            },
        )

    async def __call__(self, state: LedgerState) -> dict[str, Any]:
        entities = state.get("entities") or {}
        documents = list(state.get("document_ids") or [])
        if not documents:
            documents = [
                n for key, values in entities.items() if key.endswith("_numbers") for n in values
            ]
        if not documents:
            text = "Which document should I read? Upload one, or give its id or number."
            return {"messages": [AIMessage(text, name="extractor")], "step_count": 1}
        outcomes: list[ToolOutcome] = []
        trajectory: list[BaseMessage] = []
        saved: list[dict[str, Any]] = []
        problems: list[str] = []
        for document in documents[:MAX_DOCUMENTS]:
            done, messages, result = await self.read(document)
            outcomes += done
            trajectory += messages
            if isinstance(result, str):
                problems.append(result)
            else:
                saved.append(result)
        lines = []
        citations: dict[str, str] = {}
        for item in saved:
            tag = item.get("citation")
            if tag:
                citations[tag] = item["document_id"]
            line = f"Read {item.get('filename') or item['document_id']}: {_summary(item['record'])}"
            how = f", {item['extractor']}" if item.get("extractor") else ""
            line += f" (saved as {item['record_id']}{how})"
            if tag:
                line += f" [{tag}]"
            anomaly = item.get("anomaly")
            if anomaly and anomaly.get("flagged"):
                line += f". Flagged as unusual (score {anomaly['score']:.2f}): " + "; ".join(
                    anomaly.get("reasons", [])[:3]
                )
            lines.append(line + ".")
        lines += [f"Problem: {problem}." for problem in problems]
        planned = AIMessage(
            "",
            name="extractor",
            tool_calls=[
                {"name": o.tool, "args": o.args, "id": o.call_id, "type": "tool_call"}
                for o in outcomes
            ],
        )
        return {
            "messages": [planned, *trajectory, AIMessage("\n".join(lines), name="extractor")],
            "extracted": saved[0] if saved else None,
            "document_ids": documents,
            "citations": citations,
            "errors": problems,
            "step_count": 1 + len(outcomes),
        }
