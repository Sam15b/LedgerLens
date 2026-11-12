"""Ingest a document end to end: load -> layout/OCR -> classify -> extract -> chunk -> embed.

:class:`IngestionPipeline` runs the steps for one file and returns an :class:`IngestResult`
with the :class:`~ledgerlens.core.schemas.Document` (status ``ready`` or ``failed``), the
extraction, the chunks, their vectors, and how long each step took -- the per-step timings
are what the latency dashboards and the "median ingest time" exit criterion are built on.

Extraction runs *before* chunking (the plan lists it after): the chunks' context header
(``[INVOICE INV-889 | Vendor: ... | Date: ...]``) needs the extracted number, vendor and
date.

Embedding needs a fitted BM25 encoder. :meth:`IngestionPipeline.run` handles a batch the
way a first-time index build has to: every document is processed up to its chunks, BM25 is
fit on all of them, then each document's chunks are embedded.

Command line (also ``make ingest DATASET=sroie``)::

    python -m ledgerlens.ingestion.pipeline data/raw/sroie --limit 100 --extractor rules

writes ``data/processed/documents.jsonl`` and ``chunks.jsonl`` and prints the median time
per step.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from ledgerlens.core.config import IngestionConfig, get_config, repo_path
from ledgerlens.core.logging import configure_logging
from ledgerlens.core.schemas import DocType, Document, DocumentStatus
from ledgerlens.extraction.base import ExtractionError, ExtractionResult, Extractor
from ledgerlens.ingestion.chunking import Chunk, chunk_document
from ledgerlens.ingestion.classify import classify
from ledgerlens.ingestion.embedding import BM25Encoder, ChunkEmbedder, EmbeddedChunk
from ledgerlens.ingestion.layout import PageContent, ScannedPage, layout_pages
from ledgerlens.ingestion.loaders import FileLoader
from ledgerlens.ingestion.ocr import OCREngine, OCRError, get_engine
from ledgerlens.llm.router import LLMClient

logger = logging.getLogger(__name__)

STEPS = ("load", "layout", "ocr", "classify", "extract", "chunk", "embed")


class IngestResult(BaseModel):
    """Everything one pipeline run produced for one file."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    document: Document
    duplicate: bool = False  # same bytes ingested before: nothing else was run
    classification: str | None = None  # "rules" | "llm" | "fallback"
    extraction: ExtractionResult | None = None
    pages: list[PageContent] = Field(default_factory=list, exclude=True)
    chunks: list[Chunk] = Field(default_factory=list)
    embedded: list[EmbeddedChunk] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)

    @property
    def total_ms(self) -> float:
        return sum(self.timings_ms.values())

    @property
    def text(self) -> str:
        return "\n\n".join(page.text for page in self.pages if page.text)


def make_extractor(name: str, *, llm: LLMClient | None = None) -> Extractor:
    """The extractor named by ``extraction.default`` (``rules`` | ``llm`` | ``finetuned``)."""
    if name == "rules":
        from ledgerlens.extraction.rules import RulesExtractor

        return RulesExtractor()
    if name == "llm":
        from ledgerlens.extraction.llm_extractor import LLMExtractor

        return LLMExtractor(llm)
    raise ValueError(
        f"extractor {name!r} is not available yet (rules, llm; finetuned arrives in Phase 6a)"
    )


class _StepError(Exception):
    def __init__(self, step: str, cause: Exception) -> None:
        super().__init__(f"{step}: {cause}")
        self.step = step


class IngestionPipeline:
    """Runs the ingestion steps for one file at a time.

    Every collaborator can be injected (tests, workers that share one model); defaults
    come from ``configs/ingestion.yaml``. ``embedder=None`` skips embedding.
    """

    def __init__(
        self,
        config: IngestionConfig | None = None,
        *,
        loader: FileLoader | None = None,
        ocr: OCREngine | None = None,
        extractor: Extractor | None = None,
        embedder: ChunkEmbedder | None = None,
        llm: LLMClient | None = None,
    ) -> None:
        self.config = config or get_config().ingestion
        self.loader = loader or FileLoader(self.config.loader)
        self._ocr = ocr
        self.llm = llm
        self.extractor = extractor or make_extractor(self.config.extraction.default, llm=llm)
        self.embedder = embedder

    @property
    def ocr(self) -> OCREngine:
        """The OCR engine, created only when a scanned page needs it."""
        if self._ocr is None:
            self._ocr = get_engine(self.config.ocr)
        return self._ocr

    # ------------------------------------------------------------------ one document

    def process(self, path: Path | str, *, embed: bool = True) -> IngestResult:
        """Ingest one file. Loader errors (unsupported, too large, unreadable) are raised;
        a failure in a later step returns a ``failed`` document with the error."""
        timings: dict[str, float] = {}
        with _timed(timings, "load"):
            loaded = self.loader.load(path)
        document = loaded.to_document()
        if loaded.duplicate:
            logger.info("%s already ingested as %s", loaded.file.filename, loaded.document_id)
            return IngestResult(document=document, duplicate=True, timings_ms=timings)

        result = IngestResult(document=document.with_status(DocumentStatus.PROCESSING))
        result.timings_ms = timings
        try:
            self._read(result, Path(path), loaded.file.mime_type)
            self._understand(result)
            if embed and self.embedder is not None and self.embedder.sparse.is_fitted:
                self.embed(result)
        except _StepError as exc:
            logger.warning("ingesting %s failed at %s", loaded.file.filename, exc)
            result.document = result.document.with_status(DocumentStatus.FAILED, error=str(exc))
            return result
        result.document = result.document.with_status(DocumentStatus.READY)
        return result

    def _read(self, result: IngestResult, path: Path, mime_type: str) -> None:
        with _step(result, "layout"):
            pages = layout_pages(
                path,
                mime_type,
                min_chars=self.config.layout.text_layer_min_chars,
                dpi=self.config.ocr.dpi,
            )
        scanned = [p for p in pages if isinstance(p, ScannedPage)]
        if scanned:
            with _step(result, "ocr"):
                engine = self.ocr
                read = {p.page_number: p.to_page_content(engine.run(p.image)) for p in scanned}
            result.document = _with(result.document, ocr_engine=engine.name)
        result.pages = [
            page if isinstance(page, PageContent) else read[page.page_number] for page in pages
        ]

    def _understand(self, result: IngestResult) -> None:
        text = result.text
        with _step(result, "classify"):
            decided = classify(text, llm=self.llm)
        result.classification = decided.method
        doc_type = decided.doc_type
        result.document = _with(result.document, doc_type=doc_type)
        record = None
        if doc_type is not DocType.UNKNOWN:
            with _step(result, "extract"):
                result.extraction = self.extractor.extract(text, doc_type)
            record = result.extraction.record
        with _step(result, "chunk"):
            result.chunks = chunk_document(
                result.pages, doc_type, record=record, config=self.config.chunking
            )

    def embed(self, result: IngestResult) -> IngestResult:
        """Add dense + sparse vectors to ``result``'s chunks (needs a fitted BM25)."""
        if self.embedder is None:
            raise RuntimeError("no embedder configured")
        with _step(result, "embed"):
            result.embedded = self.embedder.embed(
                result.document.id,
                result.chunks,
                metadata={"filename": result.document.filename},
            )
        return result

    # ------------------------------------------------------------------ batches

    def run(
        self,
        paths: Sequence[Path],
        *,
        embed: bool = True,
        bm25_path: Path | None = None,
    ) -> Iterator[IngestResult]:
        """Ingest ``paths``; when embedding, fit BM25 on all their chunks first.

        Files the loader rejects are logged and skipped. BM25 params are saved to
        ``bm25_path`` (default ``embedding.sparse_params_path``).
        """
        results: list[IngestResult] = []
        for path in paths:
            try:
                results.append(self.process(path, embed=False))
            except ValueError as exc:  # LoaderError and its subclasses
                logger.warning("skipping %s: %s", path, exc)
        if embed and self.embedder is not None:
            corpus = [c.text for r in results for c in r.chunks]
            if corpus:
                self.embedder.sparse.fit(corpus)
                self.embedder.sparse.save(bm25_path)
            for result in results:
                if result.document.status is DocumentStatus.READY and result.chunks:
                    try:
                        self.embed(result)
                    except _StepError as exc:
                        result.document = result.document.with_status(
                            DocumentStatus.FAILED, error=str(exc)
                        )
        yield from results


def default_embedder(config: IngestionConfig | None = None) -> ChunkEmbedder:
    """Dense model from ``embedding.dense_model`` plus a BM25 encoder (loaded if saved)."""
    from ledgerlens.ingestion.embedding import DenseEmbedder

    config = config or get_config().ingestion
    try:
        sparse = BM25Encoder.load()
    except FileNotFoundError:
        sparse = BM25Encoder()
    return ChunkEmbedder(DenseEmbedder(config.embedding), sparse)


def _with(document: Document, **changes: object) -> Document:
    return Document.model_validate(document.model_dump() | changes)


@contextmanager
def _timed(timings: dict[str, float], step: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        timings[step] = timings.get(step, 0.0) + (time.perf_counter() - start) * 1000


@contextmanager
def _step(result: IngestResult, step: str) -> Iterator[None]:
    """Time ``step`` and turn its failure into :class:`_StepError`."""
    with _timed(result.timings_ms, step):
        try:
            yield
        except (OCRError, ExtractionError, OSError, ValueError, RuntimeError) as exc:
            raise _StepError(step, exc) from exc


# --------------------------------------------------------------------------- sinks

INPUT_SUFFIXES = (".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff")


class IngestSink(Protocol):
    """Where ingested documents go. JSONL for now; Postgres/Pinecone/Neo4j on Day 30."""

    def write(self, result: IngestResult) -> None: ...

    def close(self) -> None: ...


class JsonlSink:
    """Appends to ``documents.jsonl`` (document, extraction, timings) and ``chunks.jsonl``
    (one line per chunk, with its vectors when embedded) under ``out_dir``."""

    def __init__(self, out_dir: Path | str) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._documents = (self.out_dir / "documents.jsonl").open("a", encoding="utf-8")
        self._chunks = (self.out_dir / "chunks.jsonl").open("a", encoding="utf-8")

    def write(self, result: IngestResult) -> None:
        extraction = result.extraction
        row = {
            "document": result.document.model_dump(mode="json"),
            "duplicate": result.duplicate,
            "classification": result.classification,
            "extraction": None
            if extraction is None
            else {
                "extractor": extraction.extractor,
                "ok": extraction.ok,
                "record": extraction.record.model_dump(mode="json") if extraction.record else None,
                "missing": extraction.missing,
                "errors": extraction.errors,
            },
            "chunks": len(result.chunks),
            "timings_ms": {k: round(v, 2) for k, v in result.timings_ms.items()},
        }
        self._documents.write(json.dumps(row, ensure_ascii=False) + "\n")
        if result.embedded:
            for embedded in result.embedded:
                self._chunks.write(json.dumps(embedded.to_pinecone(), ensure_ascii=False) + "\n")
        else:
            for chunk in result.chunks:
                line = {
                    "id": chunk.vector_id(result.document.id),
                    **chunk.model_dump(mode="json"),
                }
                self._chunks.write(json.dumps(line, ensure_ascii=False) + "\n")

    def close(self) -> None:
        self._documents.close()
        self._chunks.close()


# --------------------------------------------------------------------------- CLI


def find_inputs(root: Path, limit: int | None = None) -> list[Path]:
    """Files under ``root`` (or ``root`` itself) that ingestion can read, sorted."""
    if root.is_file():
        return [root]
    if not root.is_dir():
        raise FileNotFoundError(f"{root} does not exist")
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in INPUT_SUFFIXES)
    return files[:limit] if limit is not None else files


def summarize(results: Sequence[IngestResult]) -> str:
    """Counts by status and median timings, one line per step."""
    statuses: dict[str, int] = {}
    for r in results:
        key = "duplicate" if r.duplicate else r.document.status.value
        statuses[key] = statuses.get(key, 0) + 1
    processed = [r for r in results if not r.duplicate]
    lines = [f"{len(results)} files: " + ", ".join(f"{n} {s}" for s, n in sorted(statuses.items()))]
    if processed:
        lines.append(f"median ingest time: {_median([r.total_ms for r in processed]):.0f} ms")
        for step in STEPS:
            times = [r.timings_ms[step] for r in processed if step in r.timings_ms]
            if times:
                lines.append(f"  {step:<9} median {_median(times):8.1f} ms  ({len(times)} docs)")
    return "\n".join(lines)


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ledgerlens.ingestion.pipeline",
        description="Ingest every PDF/image under a folder into JSONL (data/processed).",
    )
    parser.add_argument("source", type=Path, help="file or folder, e.g. data/raw/sroie")
    parser.add_argument("--limit", type=int, help="ingest at most N files (sorted by path)")
    parser.add_argument("--out", type=Path, default=repo_path("data/processed"))
    parser.add_argument(
        "--extractor",
        choices=["rules", "llm"],
        help="override extraction.default from configs/ingestion.yaml",
    )
    parser.add_argument("--no-embed", action="store_true", help="skip dense + sparse vectors")
    args = parser.parse_args(argv)

    configure_logging()
    config = get_config().ingestion
    pipeline = IngestionPipeline(
        config,
        extractor=make_extractor(args.extractor) if args.extractor else None,
        embedder=None if args.no_embed else default_embedder(config),
    )
    paths = find_inputs(args.source, args.limit)
    sink = JsonlSink(args.out)
    try:
        results = list(
            pipeline.run(paths, embed=not args.no_embed, bm25_path=args.out / "bm25.json")
        )
        for result in results:
            sink.write(result)
    finally:
        sink.close()
    print(summarize(results))
    print(f"wrote {args.out / 'documents.jsonl'} and {args.out / 'chunks.jsonl'}")
    return 0 if all(r.document.status is not DocumentStatus.FAILED for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
