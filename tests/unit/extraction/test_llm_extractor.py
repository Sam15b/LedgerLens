import json
from collections.abc import Generator
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from ledgerlens.core.config import ExtractionConfig, ModelsConfig
from ledgerlens.core.schemas import Invoice, Receipt
from ledgerlens.extraction.base import ExtractionError, Extractor
from ledgerlens.extraction.llm_extractor import MAX_DOCUMENT_CHARS, LLMExtractor
from ledgerlens.llm.router import LLMClient
from ledgerlens.llm.types import LLMProviderError, LLMRequest, LLMResponse, Usage

FIXTURES = Path(__file__).parents[2] / "fixtures" / "documents"
INVOICE_JSON = (FIXTURES / "invoice_digital.json").read_text(encoding="utf-8")
RECEIPT_TEXT = "KEDAI KOPI SENTOSA\nReceipt: R-000732  Date: 14/03/2026\nTOTAL MYR  9.50"


class ScriptedProvider:
    """Replies with canned text (or raises) and records every request."""

    name = "openai"

    def __init__(self, *replies: str | Exception) -> None:
        self.replies = list(replies)
        self.requests: list[LLMRequest] = []

    def complete(self, req: LLMRequest) -> LLMResponse:
        self.requests.append(req)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return LLMResponse(
            text=reply, provider=self.name, model=req.model, usage=Usage(), latency_ms=1
        )

    def stream(self, req: LLMRequest) -> Generator[str, None, Usage]:
        raise NotImplementedError
        yield ""


def extractor(provider: ScriptedProvider, **kwargs: object) -> LLMExtractor:
    models = ModelsConfig.model_validate(
        {"aliases": {"extractor": ["openai/gpt-test"]}, "retry": {"max_attempts": 1}}
    )
    client = LLMClient(models, providers={"openai": provider}, sleep=lambda s: None)
    return LLMExtractor(client, today=date(2026, 11, 12), **kwargs)  # type: ignore[arg-type]


def test_is_an_extractor() -> None:
    assert isinstance(extractor(ScriptedProvider()), Extractor)


def test_valid_reply_becomes_the_record() -> None:
    provider = ScriptedProvider(INVOICE_JSON)

    result = extractor(provider).extract("INVOICE ...", "invoice")

    assert result.ok and isinstance(result.record, Invoice)
    assert result.record == Invoice.model_validate_json(INVOICE_JSON)
    assert result.fields["total"] == "498.00" and "vendor_tax_id" in result.fields
    assert result.extractor == "llm"


def test_request_carries_prompt_schema_and_document() -> None:
    provider = ScriptedProvider(INVOICE_JSON)

    extractor(provider).extract("INVOICE INV-2026-0417", "invoice")

    (request,) = provider.requests
    system, user = request.messages
    assert request.model == "gpt-test"  # the extractor alias
    assert request.json_schema == Invoice.model_json_schema()
    assert request.temperature == 0.0
    assert "The document is a **invoice**" in system.content
    assert "Today is 2026-11-12" in system.content
    assert "$" not in system.content.replace('"$"', "")  # every placeholder filled
    assert user.content.endswith("INVOICE INV-2026-0417")


def test_whitespace_is_tidied() -> None:
    reply = json.dumps(
        {
            "doc_type": "receipt",
            "vendor_name": "KEDAI  KOPI\nSENTOSA ",
            "receipt_date": "2026-03-14",
            "currency": "MYR",
            "total": "9.50",
        }
    )

    result = extractor(ScriptedProvider(reply)).extract(RECEIPT_TEXT, "receipt")

    assert isinstance(result.record, Receipt)
    assert result.record.vendor_name == "KEDAI KOPI SENTOSA"
    assert result.record.total == Decimal("9.50")


def test_invalid_reply_is_repaired_once() -> None:
    bad = json.dumps({"doc_type": "receipt", "vendor_name": "KEDAI", "total": "9.50"})
    good = json.dumps(
        {
            "doc_type": "receipt",
            "vendor_name": "KEDAI",
            "receipt_date": "2026-03-14",
            "total": "9.50",
        }
    )
    provider = ScriptedProvider(bad, good)

    result = extractor(provider).extract(RECEIPT_TEXT, "receipt")

    assert result.ok and len(provider.requests) == 2
    assert "receipt_date" in provider.requests[1].messages[-1].content  # told what was wrong


def test_still_invalid_after_repair_is_an_error_result() -> None:
    bad = json.dumps({"doc_type": "receipt", "vendor_name": "KEDAI", "total": "9.50"})

    result = extractor(ScriptedProvider(bad, bad)).extract(RECEIPT_TEXT, "receipt")

    assert not result.ok and result.errors and "receipt_date" in result.errors[0]


def test_unreachable_models_raise() -> None:
    down = ScriptedProvider(LLMProviderError("401 unauthorized", provider="openai"))

    with pytest.raises(ExtractionError, match="LLM extraction failed"):
        extractor(down).extract(RECEIPT_TEXT, "receipt")


def test_alias_comes_from_extraction_config() -> None:
    models = ModelsConfig.model_validate({"aliases": {"cheap": ["openai/mini"]}})
    provider = ScriptedProvider(INVOICE_JSON)
    client = LLMClient(models, providers={"openai": provider})

    LLMExtractor(client, config=ExtractionConfig(model_alias="cheap")).extract("x", "invoice")

    assert provider.requests[0].model == "mini"


def test_long_documents_are_truncated() -> None:
    provider = ScriptedProvider(INVOICE_JSON)

    extractor(provider).extract("z" * (MAX_DOCUMENT_CHARS + 500), "invoice")

    assert provider.requests[0].messages[-1].content.count("z") == MAX_DOCUMENT_CHARS
