"""Fine-tuned extractor: the request it sends, and the record it builds from the reply.

The contract test plays a recorded Ollama ``/api/chat`` reply
(``fixtures/llm/ollama/chat_finetuned_receipt.json``) through the real Ollama adapter and
router, and checks both directions: the request carries the exact training prompt and
decoding settings, and the reply becomes the receipt fixture's record.
"""

import json
from collections.abc import Generator
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest

from ledgerlens.core.config import ExtractionConfig, ModelsConfig, load_config
from ledgerlens.core.schemas import DocType, Receipt
from ledgerlens.extraction.base import ExtractionError, ExtractionResult, Extractor
from ledgerlens.extraction.finetuned import (
    KEYS_BY_DOC_TYPE,
    MAX_NEW_TOKENS,
    SYSTEM_PROMPT,
    FineTunedExtractor,
    parse_json_object,
    record_fields,
)
from ledgerlens.ingestion.pipeline import make_extractor
from ledgerlens.llm.providers.ollama_provider import OllamaProvider
from ledgerlens.llm.router import LLMClient
from ledgerlens.llm.types import LLMRequest, LLMResponse, Usage
from ml.finetune.infer import prompt_example
from ml.finetune.merge_and_export import MODEL_NAME
from ml.finetune.prepare_dataset import chat_messages

FIXTURES = Path(__file__).parents[2] / "fixtures"
REPLY = FIXTURES / "llm" / "ollama" / "chat_finetuned_receipt.json"
RECEIPT = Receipt.model_validate_json(
    (FIXTURES / "documents" / "receipt_scanned.json").read_text(encoding="utf-8")
)
RECEIPT_TEXT = (
    "KEDAI KOPI SENTOSA\n12 Jalan Bunga, 50450 Kuala Lumpur\nReceipt: R-000732\n"
    "Date: 14/03/2026\nKopi O 2 x 2.20 4.40\nRoti bakar 1 x 5.10 5.10\n"
    "SST 6% incl. 0.54\nTOTAL MYR 9.50"
)
REF = f"ollama/{MODEL_NAME}"


class Ollama:
    """An Ollama server behind ``httpx.MockTransport``: records requests, replies in turn."""

    def __init__(self, *replies: httpx.Response | Exception) -> None:
        self.replies = list(replies)
        self.requests: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def provider(self) -> OllamaProvider:
        return OllamaProvider("http://ollama.test", client=httpx.Client(transport=self.transport))

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)


class Scripted:
    """An API provider that answers with fixed text and records requests."""

    name = "openai"

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.requests: list[LLMRequest] = []

    def complete(self, req: LLMRequest) -> LLMResponse:
        self.requests.append(req)
        return LLMResponse(
            text=self.replies.pop(0),
            provider=self.name,
            model=req.model,
            usage=Usage(),
            latency_ms=1,
        )

    def stream(self, req: LLMRequest) -> Generator[str, None, Usage]:
        raise NotImplementedError
        yield ""


def client(ollama: Ollama, api: Scripted | None = None) -> LLMClient:
    models = ModelsConfig.model_validate(
        {
            "aliases": {"extractor": [REF, "openai/gpt-4o-mini"]},
            "retry": {"max_attempts": 2, "backoff_s": [0]},
        }
    )
    providers: dict[str, Any] = {"ollama": ollama.provider(), "openai": api or Scripted()}
    return LLMClient(models, providers=providers, sleep=lambda s: None)


def recorded() -> httpx.Response:
    return httpx.Response(200, text=REPLY.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- contract


def test_contract_recorded_ollama_reply_becomes_the_receipt() -> None:
    ollama = Ollama(recorded())

    result = FineTunedExtractor(client(ollama)).extract(RECEIPT_TEXT, "receipt")

    # request: the training prompt, JSON mode, greedy, the trained reply length
    (request,) = ollama.requests
    trained = chat_messages(prompt_example(RECEIPT_TEXT, "receipt"), with_answer=False)
    assert request["model"] == MODEL_NAME
    assert request["messages"] == trained
    assert request["messages"][0]["content"] == SYSTEM_PROMPT
    assert request["format"] == "json"
    assert request["options"] == {"temperature": 0.0, "num_predict": MAX_NEW_TOKENS}
    assert request["stream"] is False
    # reply: the fixture's record, except what receipts are not trained to give
    assert result.ok and result.extractor == "finetuned" and not result.errors
    assert result.record == RECEIPT.model_copy(update={"receipt_number": None, "currency": "USD"})
    assert "subtotal" not in result.fields  # a CORD key the Receipt record does not have


def test_falls_back_to_the_api_model_when_the_model_is_not_created() -> None:
    missing = httpx.Response(404, json={"error": f"model '{MODEL_NAME}' not found"})
    ollama = Ollama(missing)
    api = Scripted(json.loads(REPLY.read_text(encoding="utf-8"))["message"]["content"])

    result = FineTunedExtractor(client(ollama, api)).extract(RECEIPT_TEXT, "receipt")

    assert len(ollama.requests) == 1  # a 404 is not retried
    (request,) = api.requests
    assert request.model == "gpt-4o-mini" and request.json_mode
    assert [m.content for m in request.messages] == [
        m["content"] for m in ollama.requests[0]["messages"]
    ]
    assert isinstance(result.record, Receipt) and result.record.total == Decimal("9.50")


def test_retries_a_down_ollama_then_falls_back() -> None:
    down = httpx.ConnectError("connection refused")
    ollama = Ollama(down, down)
    api = Scripted('{"vendor_name": "Kedai", "receipt_date": "2026-03-14", "total": "9.50"}')

    result = FineTunedExtractor(client(ollama, api)).extract(RECEIPT_TEXT, "receipt")

    assert len(ollama.requests) == 2 and len(api.requests) == 1
    assert result.ok


def test_no_model_answers() -> None:
    ollama = Ollama(httpx.Response(404, json={"error": "not found"}))
    models = ModelsConfig.model_validate({"aliases": {"extractor": [REF]}})
    llm = LLMClient(models, providers={"ollama": ollama.provider()}, sleep=lambda s: None)

    with pytest.raises(ExtractionError, match="fine-tuned extraction failed"):
        FineTunedExtractor(llm).extract(RECEIPT_TEXT, "receipt")


def test_reply_without_json_is_an_error_result() -> None:
    body = json.loads(REPLY.read_text(encoding="utf-8"))
    body["message"]["content"] = "I cannot read this receipt."
    ollama = Ollama(httpx.Response(200, json=body))

    result = FineTunedExtractor(client(ollama)).extract(RECEIPT_TEXT, "receipt")

    assert not result.ok
    assert result.errors == [f"{REF}: reply is not a JSON object"]


# --------------------------------------------------------------------------- prompt


@pytest.mark.parametrize("doc_type", ["receipt", "invoice"])
def test_prompt_is_the_training_prompt(doc_type: str) -> None:
    sent = FineTunedExtractor(client(Ollama())).messages("TEXT", DocType(doc_type))

    trained = chat_messages(prompt_example("TEXT", doc_type), with_answer=False)
    assert [{"role": m.role, "content": m.content} for m in sent] == trained
    assert ", ".join(KEYS_BY_DOC_TYPE[doc_type]) in sent[1].content


def test_long_documents_are_cut() -> None:
    sent = FineTunedExtractor(client(Ollama())).messages("x" * 30_000, DocType.RECEIPT)
    assert sent[1].content.endswith("x" * 100) and len(sent[1].content) < 25_000


class Recorder:
    name = "llm"

    def __init__(self) -> None:
        self.calls: list[tuple[str, DocType | str]] = []

    def extract(self, text: str, doc_type: DocType | str) -> ExtractionResult:
        self.calls.append((text, doc_type))
        return ExtractionResult(extractor=self.name, doc_type=DocType(doc_type))


@pytest.mark.parametrize("doc_type", ["purchase_order", "payment", "contract"])
def test_untrained_document_types_go_to_the_fallback(doc_type: str) -> None:
    ollama, fallback = Ollama(), Recorder()

    result = FineTunedExtractor(client(ollama), fallback=fallback).extract("text", doc_type)

    assert fallback.calls == [("text", DocType(doc_type))] and result.extractor == "llm"
    assert ollama.requests == []


def test_default_fallback_is_the_llm_extractor_on_its_own_alias() -> None:
    extractor = FineTunedExtractor(client(Ollama()), config=ExtractionConfig())
    fallback: Any = extractor.fallback
    assert isinstance(extractor, Extractor) and fallback.model == "extractor_api"
    assert extractor.model == "extractor"


# --------------------------------------------------------------------------- reply -> record


def test_record_fields_normalises_and_drops_what_the_record_lacks() -> None:
    reply = {
        "vendor_name": "Acme",
        "invoice_number": "INV-1",
        "invoice_date": "14/03/2026",
        "due_date": "2026-04-13",
        "total": "1,250.00",
        "tax": 0,
        "subtotal": None,
        "line_items": [
            {"description": "Bolts", "quantity": "2", "unit_price": "5.00", "amount": "10.00"},
            {"description": "no amount", "quantity": "1", "amount": None},
            {"description": "", "amount": "3.00"},
            "not an item",
            {"description": "Nuts", "amount": "2", "colour": "red"},
        ],
        "confidence": 0.9,
    }

    fields = record_fields(DocType.INVOICE, reply, day_first=True)

    assert fields["invoice_date"].isoformat() == "2026-03-14"
    assert fields["due_date"].isoformat() == "2026-04-13"
    assert fields["total"] == Decimal("1250.00") and fields["tax"] == 0
    assert fields["line_items"] == [
        {
            "description": "Bolts",
            "quantity": Decimal("2"),
            "unit_price": Decimal("5.00"),
            "amount": Decimal("10.00"),
        },
        {"description": "Nuts", "amount": Decimal("2")},
    ]
    assert "confidence" not in fields
    # unreadable values are left for validation to report
    bad = record_fields(DocType.RECEIPT, {"total": "n/a", "receipt_date": "soon"}, day_first=True)
    assert bad == {"total": "n/a", "receipt_date": "soon"}


def test_invalid_values_come_back_as_errors() -> None:
    body = json.loads(REPLY.read_text(encoding="utf-8"))
    body["message"]["content"] = json.dumps(
        {"vendor_name": "Kedai", "receipt_date": "2026-03-14", "total": "-9.50"}
    )

    result = FineTunedExtractor(client(Ollama(httpx.Response(200, json=body)))).extract(
        RECEIPT_TEXT, "receipt"
    )

    assert not result.ok and result.errors and result.errors[0].startswith("receipt.total")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"total": "1.00"}', {"total": "1.00"}),
        ('```json\n{"total": "1.00"}\n```', {"total": "1.00"}),
        ('Sure! {"a": {"b": 1}} trailing {"c": 2}', {"a": {"b": 1}}),
        ("[1, 2]", None),
        ("no json", None),
    ],
)
def test_parse_json_object(text: str, expected: dict[str, Any] | None) -> None:
    assert parse_json_object(text) == expected


# --------------------------------------------------------------------------- wiring


def test_pipeline_builds_it_and_the_alias_serves_the_exported_model() -> None:
    assert isinstance(make_extractor("finetuned", llm=client(Ollama())), FineTunedExtractor)
    with pytest.raises(ValueError, match="unknown extractor"):
        make_extractor("gpt")
    repo = load_config(Path(__file__).parents[3] / "configs")
    assert repo.models.resolve(repo.ingestion.extraction.model_alias)[:2] == [
        REF,
        "openai/gpt-4o-mini",
    ]
    assert REF not in repo.models.resolve(repo.ingestion.extraction.llm_alias)
