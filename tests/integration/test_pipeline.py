"""Ingest the five fixture documents end to end and compare with their expected JSON.

Runs the real pipeline -- loader, layout, Tesseract OCR for the scanned receipt,
classification, extraction, chunking, BM25 + dense embedding, JSONL sink. Only the models
are stand-ins: the dense encoder is a deterministic fake (no 440 MB download), and the
LLM extractor talks to a scripted provider that answers like a perfect model would, so the
test checks the plumbing around the model rather than the model.

Needs the Tesseract binary for the receipt (installed in CI; skipped locally without it).
"""

import hashlib
import json
import shutil
from collections.abc import Generator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ledgerlens.core.config import ModelsConfig
from ledgerlens.core.schemas import DocumentStatus
from ledgerlens.extraction.llm_extractor import LLMExtractor
from ledgerlens.extraction.rules import RulesExtractor
from ledgerlens.ingestion.embedding import BM25Encoder, ChunkEmbedder, DenseEmbedder
from ledgerlens.ingestion.pipeline import IngestionPipeline, JsonlSink, find_inputs, main
from ledgerlens.llm.router import LLMClient
from ledgerlens.llm.types import LLMRequest, LLMResponse, Usage

pytestmark = pytest.mark.integration

FIXTURES = Path(__file__).parents[1] / "fixtures" / "documents"
HAS_TESSERACT = shutil.which("tesseract") is not None
PDFS = ["invoice_digital", "purchase_order", "contract", "payment_advice"]
ALL = [*PDFS, "receipt_scanned"] if HAS_TESSERACT else PDFS


def expected(name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
    return data


class FakeEncoder:
    def encode(self, sentences: list[str], **kwargs: Any) -> Any:
        rows = [
            np.frombuffer(hashlib.sha256(s.encode()).digest()[:16], dtype=np.uint8) + 1.0
            for s in sentences
        ]
        return np.stack([row / np.linalg.norm(row) for row in rows])

    def get_embedding_dimension(self) -> int:
        return 16


class OracleProvider:
    """Answers each extraction request with the fixture JSON for the document it was sent."""

    name = "openai"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, req: LLMRequest) -> LLMResponse:
        self.calls += 1
        document = req.messages[-1].content
        markers = {
            "INV-2026-0417  498.00": "payment_advice",  # the remittance lists the invoice
            "INVOICE": "invoice_digital",
            "PURCHASE ORDER": "purchase_order",
            "AGREEMENT": "contract",
            "Receipt": "receipt_scanned",
        }
        name = next(n for marker, n in markers.items() if marker in document)
        return LLMResponse(
            text=json.dumps(expected(name)),
            provider=self.name,
            model=req.model,
            usage=Usage(tokens_in=500, tokens_out=200),
            latency_ms=1.0,
        )

    def stream(self, req: LLMRequest) -> Generator[str, None, Usage]:
        raise NotImplementedError
        yield ""


def pipeline(extractor: Any) -> IngestionPipeline:
    embedder = ChunkEmbedder(DenseEmbedder(model=FakeEncoder()), BM25Encoder())
    return IngestionPipeline(extractor=extractor, embedder=embedder)


def llm_extractor() -> tuple[LLMExtractor, OracleProvider]:
    oracle = OracleProvider()
    models = ModelsConfig.model_validate({"aliases": {"extractor": ["openai/oracle"]}})
    return LLMExtractor(LLMClient(models, providers={"openai": oracle})), oracle


def fixture_paths(tmp_path: Path) -> list[Path]:
    """The fixtures copied to a temp folder (the run must not write next to them)."""
    out = tmp_path / "raw"
    out.mkdir()
    for name in ALL:
        source = next(FIXTURES.glob(f"{name}.*[!n]"))  # .pdf / .png, not .json
        shutil.copy(source, out / source.name)
    return find_inputs(out)


def test_fixtures_ingest_to_their_expected_json(tmp_path: Path) -> None:
    extractor, oracle = llm_extractor()
    results = list(
        pipeline(extractor).run(fixture_paths(tmp_path), bm25_path=tmp_path / "bm25.json")
    )
    sink = JsonlSink(tmp_path / "processed")
    for result in results:
        sink.write(result)
    sink.close()

    rows = [json.loads(line) for line in (tmp_path / "processed" / "documents.jsonl").open()]
    by_name = {Path(r["document"]["filename"]).stem: r for r in rows}
    assert sorted(by_name) == sorted(ALL)
    for name in ALL:
        row = by_name[name]
        assert row["document"]["status"] == "ready", row["document"]["error"]
        assert row["document"]["doc_type"] == expected(name)["doc_type"]
        assert row["extraction"]["record"] == expected(name), name
        assert {"load", "layout", "classify", "extract", "chunk", "embed"} <= set(row["timings_ms"])
    assert oracle.calls == len(ALL)
    if HAS_TESSERACT:
        receipt = by_name["receipt_scanned"]["document"]
        assert (
            receipt["ocr_engine"] == "tesseract"
            and "ocr" in by_name["receipt_scanned"]["timings_ms"]
        )


def test_chunks_are_embedded_with_context_headers(tmp_path: Path) -> None:
    extractor, _ = llm_extractor()
    results = {
        r.document.filename: r
        for r in pipeline(extractor).run(fixture_paths(tmp_path), bm25_path=tmp_path / "bm25.json")
    }

    invoice = results["invoice_digital.pdf"]
    assert [c.kind for c in invoice.chunks] == ["header", "line_items"]
    header_text = str(invoice.embedded[0].metadata["text"])
    assert header_text.startswith(
        "[INVOICE INV-2026-0417 | Vendor: Acme Supplies Ltd | Date: 2026-10-02]"
    )
    assert all(len(e.values) == 16 and e.sparse.indices for e in invoice.embedded)
    contract = results["contract.pdf"]
    assert len(contract.chunks) == 7 and contract.embedded[-1].metadata["pages"] == ["2"]
    assert BM25Encoder.load(tmp_path / "bm25.json").n_docs == sum(
        len(r.chunks) for r in results.values()
    )


def test_rules_baseline_reads_every_scalar_field_of_the_digital_fixtures(tmp_path: Path) -> None:
    results = list(pipeline(RulesExtractor()).run(fixture_paths(tmp_path), embed=False))

    for result in results:
        name = Path(result.document.filename).stem
        assert result.document.status is DocumentStatus.READY
        assert result.extraction is not None and result.extraction.record is not None
        got = json.loads(result.extraction.record.model_dump_json())
        want = expected(name)
        if name == "receipt_scanned":  # OCR drops "KOPI" from the shop name
            assert {k: got[k] for k in ("receipt_number", "receipt_date", "total", "tax")} == {
                k: want[k] for k in ("receipt_number", "receipt_date", "total", "tax")
            }
            continue
        assert {k: v for k, v in got.items() if not isinstance(v, list)} == {
            k: v for k, v in want.items() if not isinstance(v, list)
        }, name


def test_reingesting_the_same_file_is_a_duplicate(tmp_path: Path) -> None:
    runner = pipeline(RulesExtractor())
    path = fixture_paths(tmp_path)[0]

    first = runner.process(path, embed=False)
    second = runner.process(path, embed=False)

    assert second.duplicate and second.document.id == first.document.id
    assert set(second.timings_ms) == {"load"}


def test_cli_ingests_a_folder(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fixture_paths(tmp_path)

    code = main(
        [
            str(tmp_path / "raw"),
            "--extractor",
            "rules",
            "--no-embed",
            "--out",
            str(tmp_path / "out"),
        ]
    )

    assert code == 0
    output = capsys.readouterr().out
    assert f"{len(ALL)} files: {len(ALL)} ready" in output and "median ingest time" in output
    lines = (tmp_path / "out" / "documents.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(ALL)
