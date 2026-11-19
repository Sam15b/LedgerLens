"""The anomaly features' vendor history, read from Postgres (throwaway database)."""

import math
from datetime import date
from decimal import Decimal

import pytest
from sqlalchemy.orm import Session, sessionmaker

from ledgerlens.anomaly.features import load_vendor_history, signed_log
from ledgerlens.core.schemas import Invoice
from ledgerlens.storage.postgres.db import session_scope
from ledgerlens.storage.postgres.repositories import InvoiceRepository, VendorRepository

pytestmark = pytest.mark.integration

Sessions = sessionmaker[Session]


def invoice(number: str, day: date, total: str) -> Invoice:
    return Invoice(
        invoice_number=number,
        vendor_name="Acme Supplies Ltd",
        invoice_date=day,
        total=Decimal(total),
    )


def test_history_holds_the_vendors_earlier_invoices_only(sessions: Sessions) -> None:
    with session_scope(sessions) as session:
        VendorRepository(session).upsert("V001", "Acme Supplies Ltd")
        VendorRepository(session).upsert("V002", "Brightline Packaging Ltd")
        repo = InvoiceRepository(session)
        repo.upsert(invoice("A-1", date(2026, 1, 5), "100.00"), id="I1", vendor_id="V001")
        repo.upsert(invoice("A-2", date(2026, 2, 5), "300.00"), id="I2", vendor_id="V001")
        repo.upsert(invoice("A-3", date(2026, 3, 5), "900.00"), id="I3", vendor_id="V001")
        repo.upsert(invoice("B-1", date(2026, 1, 9), "5.00"), id="I4", vendor_id="V002")

    with sessions() as session:
        history = load_vendor_history(session, "V001", date(2026, 3, 5), exclude_id="I3")
        same_day = load_vendor_history(session, "V001", date(2026, 3, 5))
        none = load_vendor_history(session, "V009", date(2026, 3, 5))

    logs = [signed_log(Decimal(v)) for v in ("100", "300")]
    assert history.count == 2 and history.last_invoice_date == date(2026, 2, 5)
    assert history.mean_log_total == pytest.approx(sum(logs) / 2)
    assert history.std_log_total == pytest.approx(abs(logs[1] - logs[0]) / 2)
    assert same_day.count == 3 and same_day.last_invoice_date == date(2026, 3, 5)
    assert none.count == 0 and math.isclose(none.mean_log_total, 0)
