"""analytics.sql on a small hand-checked ledger (``pytest -m integration``, needs Postgres).

Two vendors, eight invoices and four payments, chosen so every aging bucket, a partial
payment, a payment after the report date, a second currency, an unresolved vendor, a
till receipt and a void invoice each show up once. The expected numbers below were worked
out by hand from this table, as of 2026-06-30:

    id   vendor  currency invoice     due          total  paid (date)               status
    A1   acme    USD      2026-01-10  2026-02-09   1000   1000 (02-20)              paid
    A5   acme    USD      2026-01-20  2026-04-15    120   -                         unpaid
    R1   (cafe)  USD      2026-02-02  2026-02-02     50   receipt, paid on the spot paid
    V1   acme    USD      2026-02-01  2026-03-03    999   -                         void
    A2   acme    USD      2026-03-01  2026-03-31    500   200 (04-15)               partially
    B2   bolt    EUR      2026-03-15  2026-04-14   1200   1200 (04-14)              paid
    B1   bolt    USD      2026-04-01  2026-05-01    800   800 (07-10, after as_of)  paid
    A3   acme    USD      2026-05-20  2026-06-19    250   -                         unpaid
    A4   acme    USD      2026-06-15  2026-07-15    400   -                         unpaid
"""

from datetime import date
from decimal import Decimal

import pytest
from sqlalchemy.orm import Session, sessionmaker

from ledgerlens.core.schemas import Invoice, Payment, PaymentAllocation, Receipt
from ledgerlens.storage.postgres.analytics import (
    AgingRow,
    UnpaidTotal,
    VendorMonthSpend,
    ap_aging,
    average_days_to_pay,
    days_to_pay,
    spend_by_vendor_month,
    unpaid_totals,
)
from ledgerlens.storage.postgres.repositories import Repositories

pytestmark = pytest.mark.integration

AS_OF = date(2026, 6, 30)
ACME, BOLT = "v_acme", "v_bolt"
D = Decimal


def invoice(number: str, vendor: str, day: date, due: date, total: int, **kw: str) -> Invoice:
    return Invoice(
        invoice_number=number,
        vendor_name=vendor,
        invoice_date=day,
        due_date=due,
        total=D(total),
        **kw,  # type: ignore[arg-type]
    )


def payment(ref: str, vendor: str, day: date, allocations: dict[str, int], **kw: str) -> Payment:
    return Payment(
        payment_reference=ref,
        vendor_name=vendor,
        paid_at=day,
        amount=D(sum(allocations.values())),
        allocations=[
            PaymentAllocation(invoice_number=n, amount=D(a)) for n, a in allocations.items()
        ],
        **kw,  # type: ignore[arg-type]
    )


@pytest.fixture
def ledger(sessions: sessionmaker[Session]) -> sessionmaker[Session]:
    with sessions.begin() as session:
        repos = Repositories(session)
        repos.vendors.upsert(ACME, "Acme Supplies Ltd")
        repos.vendors.upsert(BOLT, "Bolt Logistics")
        for record, vendor_id, status in [
            (invoice("A1", "ACME Supplies", date(2026, 1, 10), date(2026, 2, 9), 1000), ACME, None),
            (invoice("A5", "Acme Supplies", date(2026, 1, 20), date(2026, 4, 15), 120), ACME, None),
            (invoice("V1", "Acme Supplies", date(2026, 2, 1), date(2026, 3, 3), 999), ACME, "void"),
            (invoice("A2", "Acme Supplies", date(2026, 3, 1), date(2026, 3, 31), 500), ACME, None),
            (
                invoice(
                    "B2",
                    "Bolt Logistics",
                    date(2026, 3, 15),
                    date(2026, 4, 14),
                    1200,
                    currency="EUR",
                ),
                BOLT,
                None,
            ),
            (invoice("B1", "Bolt Logistics", date(2026, 4, 1), date(2026, 5, 1), 800), BOLT, None),
            (invoice("A3", "Acme Supplies", date(2026, 5, 20), date(2026, 6, 19), 250), ACME, None),
            (invoice("A4", "Acme Supplies", date(2026, 6, 15), date(2026, 7, 15), 400), ACME, None),
        ]:
            repos.invoices.upsert(
                record, id=record.invoice_number, vendor_id=vendor_id, status=status
            )
        cafe = Receipt(
            vendor_name="Corner Cafe",
            receipt_number="R1",
            receipt_date=date(2026, 2, 2),
            total=D(50),
        )
        repos.invoices.upsert_receipt(cafe, id="R1")
        repos.payments.upsert(
            payment("P1", "Acme", date(2026, 2, 20), {"A1": 1000}), id="P1", vendor_id=ACME
        )
        repos.payments.upsert(
            payment("P2", "Acme", date(2026, 4, 15), {"A2": 200}), id="P2", vendor_id=ACME
        )
        repos.payments.upsert(
            payment("P3", "Bolt", date(2026, 7, 10), {"B1": 800}), id="P3", vendor_id=BOLT
        )
        repos.payments.upsert(
            payment("P4", "Bolt", date(2026, 4, 14), {"B2": 1200}, currency="EUR"),
            id="P4",
            vendor_id=BOLT,
        )
    return sessions


# --------------------------------------------------------------------------- AP aging


def test_ap_aging_buckets_what_was_owed_on_the_report_date(ledger: sessionmaker[Session]) -> None:
    with ledger() as session:
        rows = ap_aging(session, AS_OF)

    assert rows == [
        AgingRow("current", "USD", 1, D("400")),  # A4, due in 15 days
        AgingRow("0-30", "USD", 1, D("250")),  # A3, 11 days late
        AgingRow("31-60", "USD", 1, D("800")),  # B1, 60 days; paid only after the report date
        AgingRow("61-90", "USD", 1, D("120")),  # A5, 76 days
        AgingRow("90+", "USD", 1, D("300")),  # A2, 91 days, 500 less the 200 paid
    ]  # no EUR row: B2 was paid; R1 (receipt) and V1 (void) are never owed


def test_ap_aging_for_a_past_date_ignores_later_invoices_and_payments(
    ledger: sessionmaker[Session],
) -> None:
    with ledger() as session:
        rows = ap_aging(session, date(2026, 2, 15))

    assert rows == [
        AgingRow("current", "USD", 1, D("120")),  # A5, not due until April
        AgingRow("0-30", "USD", 1, D("1000")),  # A1, 6 days late; P1 came on the 20th
    ]


# --------------------------------------------------------------------------- spend


def test_spend_by_vendor_by_month(ledger: sessionmaker[Session]) -> None:
    with ledger() as session:
        rows = spend_by_vendor_month(session)

    assert [(r.month.month, r.vendor_name, r.currency, r.invoice_count, r.total) for r in rows] == [
        (1, "Acme Supplies Ltd", "USD", 2, D("1120")),  # A1 + A5, under the canonical name
        (2, "Corner Cafe", "USD", 1, D("50")),  # the receipt; V1 (void) is not spend
        (3, "Acme Supplies Ltd", "USD", 1, D("500")),
        (3, "Bolt Logistics", "EUR", 1, D("1200")),
        (4, "Bolt Logistics", "USD", 1, D("800")),
        (5, "Acme Supplies Ltd", "USD", 1, D("250")),
        (6, "Acme Supplies Ltd", "USD", 1, D("400")),
    ]
    assert rows[1] == VendorMonthSpend(None, "Corner Cafe", date(2026, 2, 1), "USD", 1, D("50"))


def test_spend_range_is_inclusive_and_open_ended(ledger: sessionmaker[Session]) -> None:
    with ledger() as session:
        march_to_april = spend_by_vendor_month(session, date(2026, 3, 15), date(2026, 4, 1))
        from_may = spend_by_vendor_month(session, start=date(2026, 5, 1))

    assert [(r.vendor_id, r.currency, r.total) for r in march_to_april] == [
        (BOLT, "EUR", D("1200")),  # B2 on the 15th: start is inclusive
        (BOLT, "USD", D("800")),  # B1 on the 1st: end is inclusive
    ]
    assert [r.total for r in from_may] == [D("250"), D("400")]


# --------------------------------------------------------------------------- unpaid


def test_unpaid_totals_per_vendor_and_currency(ledger: sessionmaker[Session]) -> None:
    with ledger() as session:
        rows = unpaid_totals(session, AS_OF)

    assert rows == [
        # A2 300 + A3 250 + A4 400 + A5 120; all but A4 past due; A2 is the oldest
        UnpaidTotal(ACME, "Acme Supplies Ltd", "USD", 4, D("1070"), 3, D("670"), date(2026, 3, 31)),
        UnpaidTotal(BOLT, "Bolt Logistics", "USD", 1, D("800"), 1, D("800"), date(2026, 5, 1)),
    ]


def test_nothing_is_unpaid_once_every_payment_is_in(ledger: sessionmaker[Session]) -> None:
    with ledger() as session:
        before_any_invoice = unpaid_totals(session, date(2025, 12, 31))
        rows = unpaid_totals(session, date(2026, 7, 31))

    assert before_any_invoice == []
    assert [(r.vendor_id, r.outstanding) for r in rows] == [(ACME, D("1070"))]  # B1 paid 07-10


# --------------------------------------------------------------------------- days to pay


def test_average_days_to_pay_per_vendor(ledger: sessionmaker[Session]) -> None:
    with ledger() as session:
        rows = days_to_pay(session)

    assert [
        (r.vendor_id, r.invoices_paid, r.avg_days_to_pay, r.avg_days_late, r.max_days_to_pay)
        for r in rows
    ] == [
        (ACME, 1, D(41), D(11), 41),  # A1: 01-10 -> 02-20, due 02-09; A2 is only part paid
        (BOLT, 2, D(65), D(35), 100),  # B2 30 days (on time), B1 100 days (70 late)
    ]
    assert average_days_to_pay(rows) == D(57)  # (41 + 30 + 100) / 3


def test_days_to_pay_counts_invoices_by_settlement_date(ledger: sessionmaker[Session]) -> None:
    with ledger() as session:
        first_half = days_to_pay(session, end=AS_OF)
        nothing = days_to_pay(session, start=date(2026, 8, 1))

    assert [(r.vendor_id, r.invoices_paid, r.avg_days_to_pay) for r in first_half] == [
        (ACME, 1, D(41)),
        (BOLT, 1, D(30)),  # B1 settled in July
    ]
    assert nothing == [] and average_days_to_pay(nothing) is None
