"""LLM calls made while ingesting a document are recorded against that document."""

from pathlib import Path

from ledgerlens.core.logging import get_request_id, request_context
from ledgerlens.core.schemas import DocType
from ledgerlens.extraction.base import ExtractionResult
from ledgerlens.extraction.rules import RulesExtractor
from ledgerlens.ingestion.pipeline import IngestionPipeline
from ledgerlens.observability.cost import document_request_id

INVOICE = Path(__file__).parents[2] / "fixtures" / "documents" / "invoice_digital.pdf"


class SpyExtractor:
    """The rules extractor, noting the request id each extraction runs under."""

    name = "spy"

    def __init__(self) -> None:
        self.request_ids: list[str | None] = []
        self.rules = RulesExtractor()

    def extract(self, text: str, doc_type: DocType | str) -> ExtractionResult:
        self.request_ids.append(get_request_id())
        return self.rules.extract(text, doc_type)


def test_extraction_runs_under_the_documents_request_id() -> None:
    spy = SpyExtractor()
    pipeline = IngestionPipeline(extractor=spy)

    with request_context("api-request-1"):
        result = pipeline.process(INVOICE)
        assert get_request_id() == "api-request-1"  # restored afterwards

    assert spy.request_ids == [document_request_id(result.document.id)]
    assert result.document.doc_type is DocType.INVOICE
