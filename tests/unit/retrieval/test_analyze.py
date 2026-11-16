"""Query analyzer: 30 sample questions, then each part on its own, then the LLM fallback."""

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pytest

from ledgerlens.core.schemas import DocType
from ledgerlens.llm.types import Message
from ledgerlens.retrieval import QueryFilters
from ledgerlens.retrieval.analyze import (
    LLMQueryAnalysis,
    QueryAnalyzer,
    VendorFinder,
    find_doc_numbers,
    parse_date_range,
    quarter_range,
)
from ledgerlens.storage.neo4j.linker import VendorCandidate

TODAY = date(2026, 11, 16)  # a Monday in Q4
VENDORS = [
    VendorCandidate("V001", "Acme Supplies Ltd", "GB123456789", ("ACME Supplies",)),
    VendorCandidate("V022", "Kopi Sentosa Trading"),
    VendorCandidate("V030", "Sparkle Facility Management Pte Ltd"),
    VendorCandidate("V040", "Global Office Solutions"),
    VendorCandidate("V041", "Brewhouse Coffee Co"),
    VendorCandidate("V043", "Boulangerie Lefort SARL"),
]

INV, PO, PAY, CT = (
    DocType.INVOICE,
    DocType.PURCHASE_ORDER,
    DocType.PAYMENT,
    DocType.CONTRACT,
)
D = date


@dataclass
class Case:
    question: str
    numbers: tuple[str, ...] = ()
    types: tuple[DocType, ...] = ()
    vendors: tuple[str, ...] = ()
    dates: tuple[date | None, date | None] = (None, None)
    graph: bool = False
    agg: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


CASES = [
    # single-document lookups
    Case("What's the total on INV-889?", ("INV-889",), (INV,)),
    Case("When is inv-2026-0417 due?", ("INV-2026-0417",), (INV,)),
    Case("Show me purchase order PO-1042", ("PO-1042",), (PO,)),
    Case("What was ordered on PO-2024-00001?", ("PO-2024-00001",), (PO,)),
    Case("How much was paid in PAY-2025-000227?", ("PAY-2025-000227",), (PAY,)),
    Case("Payment terms in contract CT-V001", ("CT-V001",), (CT,)),
    Case("Remittance RA-5531 details", ("RA-5531",), (PAY,)),
    # clause and document lookups by vendor
    Case("Late-payment penalty in the Acme contract?", types=(CT,), vendors=("V001",)),
    Case("What are the payment terms in the Kopi Sentosa agreement?", types=(CT,),
         vendors=("V022",)),
    Case("Invoices from Sparkle Facility in March", types=(INV,), vendors=("V030",),
         dates=(D(2026, 3, 1), D(2026, 3, 31))),
    Case("Brewhouse Coffee receipts from last month", types=(DocType.RECEIPT,), vendors=("V041",),
         dates=(D(2026, 10, 1), D(2026, 10, 31))),
    Case("ACME Supplies invoices dated 14/03/2026", types=(INV,), vendors=("V001",),
         dates=(D(2026, 3, 14), D(2026, 3, 14))),
    Case("Acme Suplies POs in FY2025", types=(PO,), vendors=("V001",),
         dates=(D(2025, 1, 1), D(2025, 12, 31))),
    Case("Global Office Solutions invoices this quarter", types=(INV,), vendors=("V040",),
         dates=(D(2026, 10, 1), D(2026, 12, 31))),
    Case("Payments since 01/03/2026", types=(PAY,), dates=(D(2026, 3, 1), None)),
    Case("Invoices between 2026-01-01 and 2026-02-15", types=(INV,),
         dates=(D(2026, 1, 1), D(2026, 2, 15))),
    Case("Contracts signed before 2025-06-30", types=(CT,), dates=(None, D(2025, 6, 29))),
    Case("Payments in the last 30 days", types=(PAY,), dates=(D(2026, 10, 17), TODAY)),
    # relationships: answered by the graph
    Case("Which payment settled INV-2026-0417?", ("INV-2026-0417",), (PAY,), graph=True),
    Case("Acme invoices whose PO has no payment", types=(INV, PO, PAY), vendors=("V001",),
         graph=True),
    Case("Invoices billed against PO-1042", ("PO-1042",), (INV,), graph=True),
    Case("Does INV-889's unit price match the contract rate?", ("INV-889",), (CT,), graph=True),
    Case("Invoices without a purchase order", types=(INV, PO), graph=True),
    Case("Possible duplicate invoices from Boulangerie Lefort", types=(INV,), vendors=("V043",),
         graph=True),
    # aggregations: answered by SQL / Text2Cypher
    Case("Total unpaid by vendor for Q1", dates=(D(2026, 1, 1), D(2026, 3, 31)), agg=True),
    Case("How much did we spend with Kopi Sentosa last month?", vendors=("V022",),
         dates=(D(2026, 10, 1), D(2026, 10, 31)), agg=True),
    Case("How many invoices are overdue?", types=(INV,), agg=True),
    Case("Top 5 vendors by spend in 2025", dates=(D(2025, 1, 1), D(2025, 12, 31)), agg=True),
    Case("Average days to pay per vendor year to date", dates=(D(2026, 1, 1), TODAY), agg=True),
    Case("Q4 2025 spend breakdown by month", dates=(D(2025, 10, 1), D(2025, 12, 31)), agg=True),
]  # fmt: skip


@pytest.fixture(scope="module")
def analyzer() -> QueryAnalyzer:
    return QueryAnalyzer(VENDORS, today=TODAY)


def test_there_are_thirty_sample_questions() -> None:
    assert len(CASES) == 30
    assert len({case.question for case in CASES}) == 30


@pytest.mark.parametrize("case", CASES, ids=[case.question for case in CASES])
def test_sample_questions_parse_correctly(analyzer: QueryAnalyzer, case: Case) -> None:
    result = analyzer.analyze(case.question)

    assert result.method == "rules"
    assert result.doc_numbers == case.numbers
    assert result.doc_types == case.types
    assert result.vendor_ids == case.vendors
    assert (result.date_from, result.date_to) == case.dates
    assert (result.needs_graph, result.aggregation) == (case.graph, case.agg)


# --------------------------------------------------------------------------- parts


def test_doc_numbers_keep_their_form_and_say_their_type() -> None:
    text = "Compare inv-889, INV-889 and SINV-000001 with PO#1042, MSA-2026-07 and CT-V030"
    assert find_doc_numbers(text) == [
        ("INV-889", INV),
        ("SINV-000001", INV),
        ("PO#1042", PO),
        ("MSA-2026-07", CT),
        ("CT-V030", CT),
    ]
    assert find_doc_numbers("the INVOICE for the POs and the PAYMENT") == []


def test_vendors_by_name_alias_first_word_and_typo() -> None:
    finder = VendorFinder(VENDORS)
    assert finder.find("ACME Supplies, Ltd. invoices") == ["V001"]  # canonical name
    assert finder.find("anything from acme?") == ["V001"]  # distinctive first word
    assert finder.find("Sparkle Facility Managment") == ["V030"]  # typo, fuzzy
    assert finder.find("kopi sentosa and brewhouse") == ["V022", "V041"]  # order of mention
    assert finder.find("global invoices") == []  # "Global" alone is too generic
    assert finder.find("nothing here") == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("in Q4", (D(2026, 10, 1), D(2026, 12, 31))),
        ("2nd quarter of 2025", (D(2025, 4, 1), D(2025, 6, 30))),
        ("last quarter", (D(2026, 7, 1), D(2026, 9, 30))),
        ("this week", (D(2026, 11, 16), D(2026, 11, 22))),
        ("last year", (D(2025, 1, 1), D(2025, 12, 31))),
        ("in December", (D(2025, 12, 1), D(2025, 12, 31))),  # not yet this year
        ("in March 2025", (D(2025, 3, 1), D(2025, 3, 31))),
        ("yesterday", (D(2026, 11, 15), D(2026, 11, 15))),
        ("past 3 months", (D(2026, 8, 16), TODAY)),
        ("after 2026-05-01", (D(2026, 5, 2), None)),
        ("may I see the invoices", (None, None)),  # "may" is not a month here
        ("no date at all", (None, None)),
    ],
)
def test_date_ranges(text: str, expected: tuple[date | None, date | None]) -> None:
    assert parse_date_range(text, TODAY) == expected


def test_fiscal_years_starting_in_april() -> None:
    assert quarter_range(2026, 1, fiscal_start=4) == (D(2025, 4, 1), D(2025, 6, 30))
    assert quarter_range(2026, 4, fiscal_start=4) == (D(2026, 1, 1), D(2026, 3, 31))
    assert parse_date_range("FY2026", TODAY, fiscal_start=4) == (D(2025, 4, 1), D(2026, 3, 31))
    assert parse_date_range("this quarter", TODAY, fiscal_start=4) == (
        D(2026, 10, 1),
        D(2026, 12, 31),
    )


def test_filters_keep_numbers_only_for_lookups(analyzer: QueryAnalyzer) -> None:
    lookup = analyzer.analyze("What's the total on INV-889?").filters()
    linked = analyzer.analyze("Which payment settled INV-889 in March 2026?").filters()
    total = analyzer.analyze("Total of Acme invoices in Q1").filters()

    assert lookup == QueryFilters(doc_types=(INV,), doc_numbers=("INV-889",))
    assert linked == QueryFilters(date_from=D(2026, 3, 1), date_to=D(2026, 3, 31))
    assert total == QueryFilters(
        doc_types=(INV,), vendor_ids=("V001",), date_from=D(2026, 1, 1), date_to=D(2026, 3, 31)
    )


def test_from_postgres_reads_the_vendor_master(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [type("Row", (), {"id": "V9", "canonical_name": "Zeta Labs", "tax_id": None,
                            "aliases": ["Zeta"]})()]  # fmt: skip

    class Repo:
        def __init__(self, session: Any) -> None:
            pass

        def all(self) -> list[Any]:
            return rows

    class Session:
        def __enter__(self) -> "Session":
            return self

        def __exit__(self, *exc: object) -> None:
            pass

    monkeypatch.setattr("ledgerlens.retrieval.analyze.VendorRepository", Repo)
    analyzer = QueryAnalyzer.from_postgres(lambda: Session(), today=TODAY)  # type: ignore[arg-type]

    assert analyzer.analyze("zeta labs invoices").vendor_names == ("Zeta Labs",)


# --------------------------------------------------------------------------- LLM fallback


class FakeLLM:
    def __init__(self, answer: LLMQueryAnalysis | Exception) -> None:
        self.answer = answer
        self.calls: list[tuple[list[Message], str]] = []

    def structured(self, messages: list[Message], schema: type[Any], *, model: str) -> Any:
        self.calls.append((messages, model))
        assert schema is LLMQueryAnalysis
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def llm_analyzer(llm: FakeLLM) -> QueryAnalyzer:
    return QueryAnalyzer(VENDORS, llm=llm, today=TODAY)  # type: ignore[arg-type]


def test_the_llm_is_asked_only_when_the_rules_find_nothing() -> None:
    llm = FakeLLM(LLMQueryAnalysis())
    analyzer = llm_analyzer(llm)

    assert analyzer.analyze("Total unpaid for Acme").method == "rules"
    assert llm.calls == []


def test_the_llm_answer_is_checked_like_the_rules() -> None:
    answer = LLMQueryAnalysis(
        doc_numbers=["inv-889", "not a number"],
        doc_types=["invoice"],
        vendor_names=["the coffee people at Brewhouse", "Unknown Corp"],
        date_from="2026-09-30",
        date_to="2026-07-01",  # backwards: swapped
        needs_graph=True,
    )
    llm = FakeLLM(answer)

    result = llm_analyzer(llm).analyze("Anything odd with what the café people sent us?")

    assert result.method == "llm"
    assert result.doc_numbers == ("INV-889",)
    assert result.doc_types == (INV,)
    assert result.vendor_ids == ("V041",) and result.vendor_names == ("Brewhouse Coffee Co",)
    assert (result.date_from, result.date_to) == (D(2026, 7, 1), D(2026, 9, 30))
    assert result.needs_graph and not result.aggregation
    ((messages, model),) = llm.calls
    assert model == "fast"
    assert "2026-11-16" in messages[0].content and "Kopi Sentosa Trading" in messages[0].content


def test_an_llm_failure_falls_back_to_the_rules_result() -> None:
    result = llm_analyzer(FakeLLM(TimeoutError("provider down"))).analyze("Anything odd?")

    assert result.method == "rules" and not result.found_anything
    assert result.filters().is_empty()
