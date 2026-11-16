"""Bulk-load a ledger -- the synthetic ``records.jsonl`` -- into all three stores.

Two passes, both idempotent (re-running a load changes nothing):

1. **Structured** (:class:`LedgerLoader`) -- every record goes straight into Postgres and
   Neo4j under its own id (``SINV-000001``, ``PO-2024-00001``, ...), no OCR and no
   extraction: the records *are* the ground truth. Vendors come from the vendor master
   lines; a record without a known vendor (the ``unknown_vendor`` anomaly) is resolved
   the way ingestion resolves one -- tax id, then name, then fuzzy match -- and becomes a
   new vendor if nothing matches. Every printed spelling that differs from the vendor's
   name is kept as an alias. Records are written vendors -> contracts -> POs -> invoices
   -> payments, so a payment's allocations find their invoices.
2. **Rendered documents** (:func:`ingest_rendered`) -- the PDFs rendered from those
   records (``pdf/manifest.jsonl``) go through the real pipeline (load, layout, classify,
   chunk, embed) into Postgres ``documents`` / ``chunks`` and Pinecone. Extraction is
   replaced by the known record, and :class:`LedgerSink` stores it under the record's
   own id and vendor, so each PDF attaches to the row the structured pass wrote instead of
   adding a second copy.

Each line of ``records.jsonl`` (written by :mod:`ml.synthetic.build`)::

    {"record_type": "vendor" | "contract" | "purchase_order" | "invoice" | "payment",
     "id": "...", "vendor_id": "..." | null, "data": {...}, ...labels...}

Labels (anomalies, payment status, duplicate_of) are not stored: evaluation reads them
from the file.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from neo4j import Driver
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from ledgerlens.core.config import get_config
from ledgerlens.core.schemas import (
    DocType,
    DocumentStatus,
    ExtractedRecord,
    Invoice,
    PurchaseOrder,
    parse_extracted,
)
from ledgerlens.extraction.base import ExtractionResult
from ledgerlens.ingestion.embedding import BM25Encoder, ChunkEmbedder, DenseEmbedder
from ledgerlens.ingestion.loaders import FileLoader, PostgresRegistry
from ledgerlens.ingestion.pipeline import IngestionPipeline, IngestResult
from ledgerlens.ingestion.sinks import StoresSink
from ledgerlens.storage.neo4j.driver import run_read
from ledgerlens.storage.neo4j.linker import (
    DEFAULT_MATCH_THRESHOLD,
    GraphLinker,
    VendorCandidate,
    VendorMatch,
    new_vendor_id,
    resolve_vendor,
)
from ledgerlens.storage.pinecone_store import PineconeStore
from ledgerlens.storage.postgres.db import session_scope
from ledgerlens.storage.postgres.models import (
    ChunkRow,
    ContractRow,
    DocumentRow,
    InvoiceRow,
    PaymentRow,
    PurchaseOrderRow,
    VendorRow,
)
from ledgerlens.storage.postgres.repositories import Repositories

logger = logging.getLogger(__name__)

RECORD_ORDER = ("vendor", "contract", "purchase_order", "invoice", "payment")
"""Write order: a payment's allocations are matched to invoices already stored."""

Progress = Callable[[str, int, int], None]
"""``progress(stage, done, total)``."""


# --------------------------------------------------------------------------- reading


@dataclass(frozen=True)
class MasterVendor:
    id: str
    name: str
    tax_id: str | None = None


@dataclass(frozen=True)
class LedgerLine:
    """One line of ``records.jsonl``: a vendor, or a record with its known vendor."""

    record_type: str
    id: str
    vendor_id: str | None = None
    record: ExtractedRecord | None = None
    vendor: MasterVendor | None = None


def parse_line(obj: Mapping[str, Any]) -> LedgerLine:
    kind = str(obj["record_type"])
    if kind not in RECORD_ORDER:
        raise ValueError(f"unknown record_type {kind!r}")
    data = obj["data"]
    if kind == "vendor":
        vendor = MasterVendor(id=str(obj["id"]), name=data["name"], tax_id=data.get("tax_id"))
        return LedgerLine(kind, vendor.id, vendor_id=vendor.id, vendor=vendor)
    return LedgerLine(kind, str(obj["id"]), obj.get("vendor_id"), record=parse_extracted(data))


def read_ledger(path: Path | str) -> list[LedgerLine]:
    with Path(path).open(encoding="utf-8") as lines:
        return [parse_line(json.loads(line)) for line in lines if line.strip()]


def _tax_id(record: ExtractedRecord) -> str | None:
    return record.vendor_tax_id if isinstance(record, Invoice | PurchaseOrder) else None


# --------------------------------------------------------------------------- vendors


@dataclass
class VendorPlan:
    name: str
    tax_id: str | None = None
    aliases: list[str] = field(default_factory=list)
    new: bool = False
    """Not in the vendor master: created for a record whose vendor nothing matched."""


@dataclass(frozen=True)
class Assignment:
    """A record line and the vendor it is stored under."""

    line: LedgerLine
    vendor_id: str

    @property
    def record(self) -> ExtractedRecord:
        assert self.line.record is not None  # vendor lines are never assigned
        return self.line.record


def plan_vendors(
    lines: Sequence[LedgerLine], *, threshold: float = DEFAULT_MATCH_THRESHOLD
) -> tuple[dict[str, VendorPlan], list[Assignment]]:
    """Decide every record's vendor and every vendor's aliases, without touching a store.

    A record's ``vendor_id`` wins when it names a master vendor; otherwise the printed
    vendor is resolved against the vendors so far (:func:`resolve_vendor`, as ingestion
    does), and a new vendor is created if nothing matches.
    """
    vendors: dict[str, VendorPlan] = {}
    for line in lines:
        if line.vendor is not None:
            vendors[line.id] = VendorPlan(line.vendor.name, line.vendor.tax_id)
    candidates = {
        vendor_id: VendorCandidate(vendor_id, plan.name, plan.tax_id)
        for vendor_id, plan in vendors.items()
    }
    assignments: list[Assignment] = []
    for line in lines:
        record = line.record
        if record is None:
            continue
        vendor_id = line.vendor_id
        if vendor_id is None or vendor_id not in vendors:
            match = resolve_vendor(
                record.vendor_name, _tax_id(record), candidates.values(), threshold=threshold
            )
            vendor_id = match.vendor_id if match else new_vendor_id(record.vendor_name)
            if vendor_id not in vendors:
                vendors[vendor_id] = VendorPlan(record.vendor_name, _tax_id(record), new=True)
                candidates[vendor_id] = VendorCandidate(vendor_id, record.vendor_name)
        plan = vendors[vendor_id]
        if record.vendor_name != plan.name and record.vendor_name not in plan.aliases:
            plan.aliases.append(record.vendor_name)
            candidates[vendor_id] = replace(candidates[vendor_id], aliases=tuple(plan.aliases))
        assignments.append(Assignment(line, vendor_id))
    return vendors, assignments


# --------------------------------------------------------------------------- structured load


@dataclass
class LoadReport:
    records: dict[str, int] = field(default_factory=dict)
    """Records written, by record type."""
    vendors: int = 0
    new_vendors: int = 0
    timings_s: dict[str, float] = field(default_factory=dict)
    """Wall time per stage: ``plan``, ``postgres``, ``neo4j``."""
    assignments: dict[str, Assignment] = field(default_factory=dict, repr=False)
    """Record id -> the line and its vendor (what :func:`ingest_rendered` attaches to)."""


def attached_documents(session: Session) -> dict[str, str]:
    """Record id -> document id, for records a document is already attached to (so a
    re-load keeps the link the rendered-documents pass made)."""
    out: dict[str, str] = {}
    for model in (ContractRow, PurchaseOrderRow, InvoiceRow, PaymentRow):
        rows = session.execute(
            select(model.id, model.document_id).where(model.document_id.is_not(None))
        )
        out.update({row.id: row.document_id for row in rows})
    return out


class LedgerLoader:
    """Structured pass: ledger lines -> Postgres (+ Neo4j when ``linker`` is given)."""

    def __init__(
        self,
        sessions: sessionmaker[Session],
        linker: GraphLinker | None = None,
        *,
        batch_size: int = 500,
        threshold: float = DEFAULT_MATCH_THRESHOLD,
    ) -> None:
        self.sessions = sessions
        self.linker = linker
        self.batch_size = batch_size
        self.threshold = threshold

    def load(self, lines: Iterable[LedgerLine], *, progress: Progress | None = None) -> LoadReport:
        order = {kind: n for n, kind in enumerate(RECORD_ORDER)}
        ordered = sorted(lines, key=lambda line: order[line.record_type])  # stable
        report = LoadReport()
        started = time.perf_counter()
        vendors, assignments = plan_vendors(ordered, threshold=self.threshold)
        report.timings_s["plan"] = time.perf_counter() - started
        report.vendors = len(vendors)
        report.new_vendors = sum(plan.new for plan in vendors.values())
        report.assignments = {item.line.id: item for item in assignments}
        for item in assignments:
            kind = item.line.record_type
            report.records[kind] = report.records.get(kind, 0) + 1

        started = time.perf_counter()
        documents = self._postgres(vendors, assignments, progress)
        report.timings_s["postgres"] = time.perf_counter() - started
        if self.linker is not None:
            started = time.perf_counter()
            self._graph(self.linker, vendors, assignments, documents, progress)
            report.timings_s["neo4j"] = time.perf_counter() - started
        return report

    def _postgres(
        self,
        vendors: Mapping[str, VendorPlan],
        assignments: Sequence[Assignment],
        progress: Progress | None,
    ) -> dict[str, str]:
        with session_scope(self.sessions) as session:
            repos = Repositories(session)
            documents = attached_documents(session)
            taken: set[str] = set()
            for vendor_id, plan in vendors.items():
                tax_id = plan.tax_id if plan.tax_id not in taken else None  # unique column
                if tax_id:
                    taken.add(tax_id)
                repos.vendors.upsert(vendor_id, plan.name, aliases=plan.aliases, tax_id=tax_id)
        for start in range(0, len(assignments), self.batch_size):
            batch = assignments[start : start + self.batch_size]
            with session_scope(self.sessions) as session:
                repos = Repositories(session)
                for item in batch:
                    repos.upsert_record(
                        item.record,
                        id=item.line.id,
                        document_id=documents.get(item.line.id),
                        vendor_id=item.vendor_id,
                    )
            if progress is not None:
                progress("postgres", start + len(batch), len(assignments))
        return documents

    def _graph(
        self,
        linker: GraphLinker,
        vendors: Mapping[str, VendorPlan],
        assignments: Sequence[Assignment],
        documents: Mapping[str, str],
        progress: Progress | None,
    ) -> None:
        for vendor_id, plan in vendors.items():
            linker.resolver.upsert(vendor_id, plan.name, tax_id=plan.tax_id, aliases=plan.aliases)
        for done, item in enumerate(assignments, start=1):
            linker.link(
                item.record,
                id=item.line.id,
                document_id=documents.get(item.line.id),
                vendor_id=item.vendor_id,
            )
            if progress is not None and (done % self.batch_size == 0 or done == len(assignments)):
                progress("neo4j", done, len(assignments))


# --------------------------------------------------------------------------- rendered PDFs


@dataclass(frozen=True)
class Rendered:
    """One rendered file from ``pdf/manifest.jsonl``."""

    record_id: str
    doc_type: str
    path: Path


def read_manifest(path: Path | str) -> list[Rendered]:
    """Entries of a render manifest; file paths are relative to its parent's parent
    (``data/synthetic`` for ``data/synthetic/pdf/manifest.jsonl``)."""
    manifest = Path(path)
    root = manifest.parent.parent
    entries: list[Rendered] = []
    with manifest.open(encoding="utf-8") as lines:
        for line in lines:
            if line.strip():
                obj = json.loads(line)
                entries.append(Rendered(obj["record_id"], obj["doc_type"], root / obj["pdf"]))
    return entries


class KnownRecordExtractor:
    """Stands in for extraction when the record is already known: returns ``record``,
    which the caller sets before each document."""

    name = "ledger"

    def __init__(self) -> None:
        self.record: ExtractedRecord | None = None

    def extract(self, text: str, doc_type: DocType | str) -> ExtractionResult:
        if self.record is None:
            raise RuntimeError("no known record set for this document")
        return ExtractionResult(
            extractor=self.name, doc_type=DocType(self.record.doc_type), record=self.record
        )


class LedgerSink(StoresSink):
    """:class:`StoresSink` for documents rendered from ledger records: the record is
    stored under its ledger id and vendor (named by the file, ``<record id>.pdf``) rather
    than under the document id with a resolved vendor."""

    def __init__(self, assignments: Mapping[str, Assignment], **stores: Any) -> None:
        super().__init__(**stores)
        self.assignments = assignments

    def _assignment(self, result: IngestResult) -> Assignment | None:
        return self.assignments.get(Path(result.document.filename).stem)

    def record_id(self, result: IngestResult) -> str:
        known = self._assignment(result)
        return known.line.id if known else super().record_id(result)

    def resolve_vendor(
        self, result: IngestResult, record: ExtractedRecord | None
    ) -> VendorMatch | None:
        known = self._assignment(result)
        if known is None:
            return super().resolve_vendor(result, record)
        return VendorMatch(known.vendor_id, "given", 100.0)


@dataclass
class RenderedReport:
    documents: int = 0
    ready: int = 0
    duplicates: int = 0
    failed: dict[str, str] = field(default_factory=dict)
    """File name -> error."""
    chunks: int = 0
    step_ms: dict[str, list[float]] = field(default_factory=dict)
    """Per pipeline step, the time each document spent in it."""
    timings_s: dict[str, float] = field(default_factory=dict)


def ingest_rendered(
    rendered: Sequence[Rendered],
    assignments: Mapping[str, Assignment],
    pipeline: IngestionPipeline,
    extractor: KnownRecordExtractor,
    sink: StoresSink,
    *,
    bm25_path: Path | None = None,
    progress: Progress | None = None,
) -> RenderedReport:
    """Run the rendered files through ``pipeline`` (whose extractor is ``extractor``) and
    store them with ``sink``, as :meth:`IngestionPipeline.ingest` does -- process all,
    fit BM25 on every chunk, embed, write -- but with each file's known record and type
    (classification is skipped: the type is known, and it is not what is being tested)."""
    report = RenderedReport()
    results: list[IngestResult] = []
    started = time.perf_counter()
    for done, item in enumerate(rendered, start=1):
        known = assignments.get(item.record_id)
        if known is None:
            logger.warning("%s: record %s is not in the ledger", item.path, item.record_id)
            continue
        extractor.record = known.record
        try:
            result = pipeline.process(
                item.path, embed=False, doc_type=DocType(known.record.doc_type)
            )
        except ValueError as exc:  # rejected by the loader
            report.failed[item.path.name] = str(exc)
            continue
        results.append(result)
        if progress is not None:
            progress("process", done, len(rendered))
    report.timings_s["process"] = time.perf_counter() - started

    started = time.perf_counter()
    embedder = pipeline.embedder
    if embedder is not None:
        corpus = [chunk.text for result in results for chunk in result.chunks]
        if corpus:
            embedder.sparse.fit(corpus)
            embedder.sparse.save(bm25_path)
        for done, result in enumerate(results, start=1):
            if result.document.status is DocumentStatus.READY and result.chunks:
                try:
                    pipeline.embed(result)
                except Exception as exc:  # the pipeline wraps step errors; keep going
                    failed = result.document.with_status(DocumentStatus.FAILED, error=str(exc))
                    result.document = failed
            if progress is not None:
                progress("embed", done, len(results))
    report.timings_s["embed"] = time.perf_counter() - started

    started = time.perf_counter()
    for done, result in enumerate(results, start=1):
        result.document = sink.write(result)
        report.documents += 1
        if result.duplicate:
            report.duplicates += 1
        elif result.document.status is DocumentStatus.READY:
            report.ready += 1
            report.chunks += len(result.chunks)
        else:
            report.failed[result.document.filename] = result.document.error or "failed"
        for step, ms in result.timings_ms.items():
            report.step_ms.setdefault(step, []).append(ms)
        if progress is not None:
            progress("store", done, len(results))
    report.timings_s["store"] = time.perf_counter() - started
    return report


# --------------------------------------------------------------------------- parity

PARITY = (
    ("vendors", "Vendor"),
    ("contracts", "Contract"),
    ("purchase_orders", "PurchaseOrder"),
    ("invoices", "Invoice"),
    ("payments", "Payment"),
)
"""Postgres table -> the Neo4j label holding one node per row."""


@dataclass(frozen=True)
class StoreCounts:
    postgres: dict[str, int]
    """Rows per table; ``chunks`` counts only chunks of ``ready`` documents."""
    pinecone: int
    """Vectors in the namespace."""
    neo4j: dict[str, int]
    """Nodes per label."""

    def mismatches(self) -> list[str]:
        """Where the stores disagree (empty when they all match)."""
        out = [
            f"{table}: postgres {self.postgres.get(table, 0)} != neo4j {self.neo4j.get(label, 0)}"
            for table, label in PARITY
            if self.postgres.get(table, 0) != self.neo4j.get(label, 0)
        ]
        if self.postgres.get("chunks", 0) != self.pinecone:
            out.append(
                f"chunks: postgres {self.postgres.get('chunks', 0)} != pinecone {self.pinecone}"
            )
        return out


def count_stores(
    sessions: sessionmaker[Session], pinecone: PineconeStore, driver: Driver
) -> StoreCounts:
    """Row, vector and node counts across the three stores."""
    with sessions() as session:
        postgres = {
            table: int(session.scalar(select(func.count()).select_from(model)) or 0)
            for table, model in (
                ("vendors", VendorRow),
                ("contracts", ContractRow),
                ("purchase_orders", PurchaseOrderRow),
                ("invoices", InvoiceRow),
                ("payments", PaymentRow),
                ("documents", DocumentRow),
            )
        }
        postgres["chunks"] = int(
            session.scalar(
                select(func.count())
                .select_from(ChunkRow)
                .join(DocumentRow, DocumentRow.id == ChunkRow.document_id)
                .where(DocumentRow.status == DocumentStatus.READY.value)
            )
            or 0
        )
    nodes = run_read(driver, "MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS n")
    return StoreCounts(
        postgres=postgres,
        pinecone=pinecone.count(),
        neo4j={row["label"]: int(row["n"]) for row in nodes},
    )


def load_rendered(
    rendered: Sequence[Rendered],
    loaded: LoadReport,
    *,
    sessions: sessionmaker[Session],
    pinecone: PineconeStore | None,
    linker: GraphLinker | None,
    embedder: ChunkEmbedder | None = None,
    bm25_path: Path | None = None,
    progress: Progress | None = None,
) -> RenderedReport:
    """:func:`ingest_rendered` with the production pipeline: files deduplicated against
    Postgres, the configured dense model (unless ``embedder`` is given) and a BM25 fit on
    these documents, stored by a :class:`LedgerSink` on the records ``loaded`` wrote."""
    config = get_config().ingestion
    extractor = KnownRecordExtractor()
    sink = LedgerSink(loaded.assignments, sessions=sessions, pinecone=pinecone, linker=linker)
    pipeline = IngestionPipeline(
        config,
        loader=FileLoader(config.loader, registry=PostgresRegistry(sessions)),
        extractor=extractor,
        embedder=embedder or ChunkEmbedder(DenseEmbedder(config.embedding), BM25Encoder()),
        on_status=sink.mark,
    )
    return ingest_rendered(
        rendered,
        loaded.assignments,
        pipeline,
        extractor,
        sink,
        bm25_path=bm25_path,
        progress=progress,
    )
