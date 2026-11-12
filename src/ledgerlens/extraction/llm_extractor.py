"""Structured extraction with an LLM through the model router.

The document text and the versioned prompt (``agents/prompts/extract_invoice.md``) go to
the ``extraction.model_alias`` alias (``extractor`` in ``models.yaml``) with the record's
JSON schema as native structured output; the router validates the reply against the
Pydantic model and gives the model one chance to repair it. The record is then tidied by
:func:`tidy_record` (whitespace in text fields), the same pass every extractor's output
goes through, so evaluation compares like with like.
"""

from __future__ import annotations

import logging
import re
from datetime import date
from typing import Any

from ledgerlens.core.config import ExtractionConfig, get_config
from ledgerlens.core.prompts import Prompt, load_prompt
from ledgerlens.core.schemas import (
    DocType,
    ExtractedModel,
    ExtractedRecord,
    extraction_schema,
    parse_extracted,
)
from ledgerlens.extraction.base import ExtractionError, ExtractionResult
from ledgerlens.llm.router import LLMClient
from ledgerlens.llm.types import LLMError, LLMValidationError, Message

logger = logging.getLogger(__name__)

PROMPT_NAME = "extract_invoice"
MAX_DOCUMENT_CHARS = 24_000  # ~6k tokens: every fixture and SROIE receipt fits many times
_SPACES = re.compile(r"\s+")


class LLMExtractor:
    """:class:`~ledgerlens.extraction.base.Extractor` backed by ``LLMClient.structured``."""

    name = "llm"

    def __init__(
        self,
        client: LLMClient | None = None,
        *,
        config: ExtractionConfig | None = None,
        prompt: Prompt | None = None,
        today: date | None = None,
    ) -> None:
        self.config = config or get_config().ingestion.extraction
        self._client = client
        self.prompt = prompt or load_prompt(PROMPT_NAME)
        self.today = today

    @property
    def client(self) -> LLMClient:
        if self._client is None:
            self._client = LLMClient()
        return self._client

    @property
    def model(self) -> str:
        return self.config.model_alias

    def messages(self, text: str, doc_type: DocType) -> list[Message]:
        today = (self.today or date.today()).isoformat()
        system = self.prompt.render(doc_type=doc_type.value, today=today)
        if len(text) > MAX_DOCUMENT_CHARS:
            logger.warning("document text truncated from %d characters for extraction", len(text))
            text = text[:MAX_DOCUMENT_CHARS]
        return [Message.system(system), Message.user(f"Document text:\n\n{text}")]

    def extract(self, text: str, doc_type: DocType | str) -> ExtractionResult:
        """Extract ``doc_type`` from ``text``.

        A reply that still fails validation after the repair attempt comes back as a result
        with ``errors``; failing to reach any model raises :class:`ExtractionError`.
        """
        kind = DocType(doc_type)
        schema = extraction_schema(kind)
        try:
            record = self.client.structured(self.messages(text, kind), schema, model=self.model)
        except LLMValidationError as exc:
            logger.warning("%s output for %s did not validate: %s", self.prompt.id, kind, exc)
            return ExtractionResult(extractor=self.name, doc_type=kind, errors=[str(exc)])
        except LLMError as exc:
            raise ExtractionError(f"LLM extraction failed: {exc}") from exc
        tidy = tidy_record(record)
        return ExtractionResult(
            extractor=self.name,
            doc_type=kind,
            record=tidy,
            fields=tidy.model_dump(mode="json", exclude_none=True),
        )


def _tidy(value: Any) -> Any:
    if isinstance(value, str):
        return _SPACES.sub(" ", value).strip()
    if isinstance(value, list):
        return [_tidy(v) for v in value]
    if isinstance(value, dict):
        return {k: _tidy(v) for k, v in value.items()}
    return value


def tidy_record(record: ExtractedModel) -> ExtractedRecord:
    """Collapse runs of whitespace (OCR line breaks inside names and addresses)."""
    return parse_extracted(_tidy(record.model_dump(mode="json")))
