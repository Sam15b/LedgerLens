"""The MCP record tools without a database: tool listing and annotations, amount
formatting, ambiguous numbers, and the argument checks that run before any query.
(The queries themselves: tests/integration/test_mcp_records_db.py.)"""

import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import pytest
from mcp.server.fastmcp.exceptions import ToolError
from mcp.shared.memory import create_connected_server_and_client_session

from ledgerlens.mcp_server.server import create_server
from ledgerlens.mcp_server.services import Services
from ledgerlens.mcp_server.tools import records
from ledgerlens.retrieval.analyze import VendorFinder
from ledgerlens.storage.neo4j.linker import VendorCandidate

RECORD_TOOLS = {
    "get_invoice",
    "get_po",
    "get_contract",
    "get_payments",
    "list_overdue",
    "run_analytics",
    "get_vendor_contact",
    "get_document_text",
    "save_extraction",
}


@pytest.fixture
def services() -> Services:
    return Services(vendors=VendorFinder([VendorCandidate("V001", "Acme Supplies Ltd")]))


@pytest.mark.anyio
async def test_record_tools_are_listed_with_output_schemas(services: Services) -> None:
    async with create_connected_server_and_client_session(create_server(services)) as client:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}
    assert set(tools) >= RECORD_TOOLS
    for name in RECORD_TOOLS:
        tool = tools[name]
        assert tool.outputSchema is not None, name
        assert tool.annotations is not None
        assert tool.annotations.readOnlyHint is (name != "save_extraction"), name
    save = tools["save_extraction"].annotations
    assert save is not None and save.idempotentHint and not save.destructiveHint
    assert "spend_by_vendor(start, end, vendor)" in str(tools["run_analytics"].description)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.parametrize(
    ("value", "expected"),
    [("783.0600", "783.06"), ("0", "0.00"), ("1000", "1000.00"), ("0.12345000", "0.12345")],
)
def test_amounts_read_like_money(value: str, expected: str) -> None:
    assert str(records.tidy_amount(Decimal(value))) == expected


@pytest.mark.parametrize(
    ("value", "expected"), [("17.0000", "17"), ("100", "100"), ("2.5000", "2.5")]
)
def test_quantities_drop_trailing_zeros(value: str, expected: str) -> None:
    assert str(records.tidy_quantity(Decimal(value))) == expected


def test_citation() -> None:
    assert records.citation("INV-889", "5a0aef53-1111") == "doc:INV-889"
    assert records.citation(None, "5a0aef53-1111") == "doc:5a0aef53"
    assert records.citation("INV-889", None) is None


@dataclass
class FakeRow:
    id: str
    vendor_name: str


def test_ambiguous_number_lists_candidates() -> None:
    rows = [FakeRow("I-1", "Acme Supplies Ltd"), FakeRow("I-2", "Brightline Packaging Ltd")]
    with pytest.raises(ToolError) as caught:
        records._pick(rows, "invoice", "INV-1")  # type: ignore[type-var]
    message = str(caught.value)
    assert "2 invoices are numbered 'INV-1'" in message
    assert "I-1 (Acme Supplies Ltd), I-2 (Brightline Packaging Ltd)" in message
    assert records._pick(rows[:1], "invoice", "INV-1") is rows[0]  # type: ignore[type-var]


@pytest.mark.parametrize(
    ("report", "params", "message"),
    [
        ("forecast", None, "unknown report 'forecast'"),
        ("aging", {"vendor": "Acme"}, "aging takes no vendor (it takes: as_of)"),
        ("unpaid", {"as_of": "last tuesday"}, "as_of must be a date"),
        ("days_to_pay", {"vendor": "Nobody Ltd"}, "no vendor in the vendor master"),
    ],
)
def test_run_analytics_checks_arguments_first(
    services: Services, report: str, params: dict[str, Any] | None, message: str
) -> None:
    with pytest.raises(ToolError, match=re.escape(message)):
        records.run_analytics(services, report, params)


def test_list_overdue_needs_a_positive_minimum(services: Services) -> None:
    with pytest.raises(ToolError, match="at least 1"):
        records.list_overdue(services, min_days=0)


def test_save_extraction_validates_before_writing(services: Services) -> None:
    with pytest.raises(ToolError, match="vendor_name: Field required"):
        records.save_extraction(services, "x", {"doc_type": "invoice", "invoice_number": "1"})
    with pytest.raises(ToolError, match="unexpected value|does not validate"):
        records.save_extraction(services, "x", {"doc_type": "memo"})
    valid = {
        "doc_type": "invoice",
        "invoice_number": "INV-1",
        "vendor_name": "Acme Supplies Ltd",
        "invoice_date": "2026-01-05",
        "total": "10.00",
    }
    with pytest.raises(ToolError, match="must be a stored document's id"):
        records.save_extraction(services, "INV-1", valid)


def test_document_ids_pass_through_without_a_lookup() -> None:
    document_id = "5A0AEF53-1111-4222-8333-444455556666"
    assert records.find_document_id(None, document_id) == document_id.lower()  # type: ignore[arg-type]
