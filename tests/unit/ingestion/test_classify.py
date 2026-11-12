from collections.abc import Generator
from pathlib import Path

import pytest

from ledgerlens.core.config import ModelsConfig
from ledgerlens.core.schemas import DocType
from ledgerlens.ingestion.classify import classify, classify_by_rules, score_text
from ledgerlens.ingestion.layout import PageContent, layout_pages
from ledgerlens.llm.router import LLMClient
from ledgerlens.llm.types import LLMProviderError, LLMRequest, LLMResponse, Usage

FIXTURES = Path(__file__).parents[2] / "fixtures" / "documents"
# What OCR reads off receipt_scanned.png (Tesseract 5.3, see test_tesseract.py).
RECEIPT_OCR = """\
KEDAI  SENTOSA
12 Jalan Bunga, 50450 Kuala Lumpur
Receipt: R-000732  Date: 14/03/2026
2x Kom O  440
1 x Roti bakar  5.10
TOTAL MYR  9.50
incl. SST 6%  0.54
Terima kasih!
"""
AMBIGUOUS = "Statement of account\nAcme Supplies Ltd\nBalance brought forward 1,200.00"


class ScriptedProvider:
    name = "ollama"

    def __init__(self, *replies: str | Exception) -> None:
        self.replies = list(replies)
        self.requests: list[LLMRequest] = []

    def complete(self, req: LLMRequest) -> LLMResponse:
        self.requests.append(req)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return LLMResponse(
            text=reply, provider=self.name, model=req.model, usage=Usage(), latency_ms=1.0
        )

    def stream(self, req: LLMRequest) -> Generator[str, None, Usage]:
        raise NotImplementedError
        yield ""


def client(provider: ScriptedProvider) -> LLMClient:
    config = ModelsConfig.model_validate(
        {"aliases": {"fast": ["ollama/tiny"]}, "retry": {"max_attempts": 1}}
    )
    return LLMClient(config, providers={"ollama": provider}, sleep=lambda s: None)


def fixture_text(name: str) -> str:
    pages = layout_pages(FIXTURES / name, "application/pdf")
    return "\n\n".join(p.text for p in pages if isinstance(p, PageContent))


@pytest.mark.parametrize(
    "name,expected",
    [
        ("invoice_digital.pdf", DocType.INVOICE),
        ("purchase_order.pdf", DocType.PURCHASE_ORDER),
        ("contract.pdf", DocType.CONTRACT),
        ("payment_advice.pdf", DocType.PAYMENT),
    ],
)
def test_rules_classify_every_fixture(name: str, expected: DocType) -> None:
    result = classify(fixture_text(name))

    assert (result.doc_type, result.method) == (expected, "rules")


def test_rules_classify_ocr_text_of_the_scanned_receipt() -> None:
    result = classify(RECEIPT_OCR)

    assert (result.doc_type, result.method) == (DocType.RECEIPT, "rules")


def test_title_lines_outweigh_mentions_in_the_body() -> None:
    # A PO says "invoice" in its terms; its title still wins.
    text = "PURCHASE ORDER\nNorthwind Traders\nPO number: 1\nSupplier: Acme\nQuote it on invoices."
    scores = score_text(text)

    assert scores[DocType.PURCHASE_ORDER] > 2 * scores[DocType.INVOICE]


def test_unclear_text_goes_to_the_fast_llm() -> None:
    provider = ScriptedProvider('{"doc_type": "payment"}')

    result = classify(AMBIGUOUS, llm=client(provider))

    assert classify_by_rules(AMBIGUOUS) is None
    assert (result.doc_type, result.method) == (DocType.PAYMENT, "llm")
    (request,) = provider.requests
    assert request.model == "tiny" and request.json_schema is not None
    assert "Statement of account" in request.messages[-1].content


def test_without_llm_or_when_it_fails_the_rules_guess_is_used() -> None:
    weak_invoice = "Invoice\nAcme"  # the title alone scores 6: rules decide, no LLM needed
    failing = client(ScriptedProvider(LLMProviderError("down", provider="ollama")))

    assert classify(AMBIGUOUS).doc_type is DocType.UNKNOWN
    assert classify(AMBIGUOUS, llm=failing).method == "fallback"
    assert classify(weak_invoice).doc_type is DocType.INVOICE
