"""Accounts-payable analytics: the queries in ``analytics.sql``, returning typed rows.

The SQL lives in one file so it can be read, reviewed and run (``psql``) as SQL; each
query is introduced by a ``-- name: <query>`` line and :func:`load_queries` splits the
file on those. Every function here takes a :class:`~sqlalchemy.orm.Session` and returns
frozen dataclasses with ``Decimal`` amounts::

    with session_scope() as session:
        for row in ap_aging(session, as_of=date.today()):
            print(row.bucket, row.currency, row.outstanding)

Amounts are per currency -- no report adds USD to EUR.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from functools import lru_cache
from importlib.resources import files
from typing import Any, Literal

from sqlalchemy import Row, text
from sqlalchemy.orm import Session

AgingBucket = Literal["current", "0-30", "31-60", "61-90", "90+"]
AGING_BUCKETS: tuple[AgingBucket, ...] = ("current", "0-30", "31-60", "61-90", "90+")

_NAME = re.compile(r"^-- name: *(?P<name>[a-z_][a-z0-9_]*) *$", re.MULTILINE)


def parse_queries(script: str) -> dict[str, str]:
    """``{name: sql}`` for every ``-- name:`` block (text before the first one is a
    header and is dropped)."""
    marks = list(_NAME.finditer(script))
    queries: dict[str, str] = {}
    for mark, following in zip(marks, [*marks[1:], None], strict=True):
        name = mark["name"]
        if name in queries:
            raise ValueError(f"query {name!r} is defined twice")
        body = script[mark.end() : following.start() if following else len(script)]
        queries[name] = body.strip().rstrip(";")
    return queries


@lru_cache
def load_queries() -> dict[str, str]:
    """The queries in the packaged ``analytics.sql``."""
    script = files("ledgerlens.storage.postgres").joinpath("analytics.sql").read_text("utf-8")
    return parse_queries(script)


def run_query(session: Session, name: str, **params: Any) -> list[Row[Any]]:
    """Run the named query with ``params`` bound."""
    try:
        sql = load_queries()[name]
    except KeyError:
        raise KeyError(f"no query named {name!r} in analytics.sql") from None
    return list(session.execute(text(sql), params))


# --------------------------------------------------------------------------- AP aging


@dataclass(frozen=True)
class AgingRow:
    bucket: AgingBucket
    currency: str
    invoice_count: int
    outstanding: Decimal


def ap_aging(session: Session, as_of: date) -> list[AgingRow]:
    """What was owed on ``as_of`` by days past due, per currency; buckets with nothing
    owed are left out. Rows come youngest bucket first within each currency."""
    return [
        AgingRow(row.bucket, row.currency, int(row.invoice_count), Decimal(row.outstanding))
        for row in run_query(session, "ap_aging", as_of=as_of)
    ]


# --------------------------------------------------------------------------- spend


@dataclass(frozen=True)
class VendorMonthSpend:
    vendor_id: str | None
    """``None`` for invoices whose vendor is not resolved yet (grouped by printed name)."""
    vendor_name: str
    month: date
    """The first day of the month."""
    currency: str
    invoice_count: int
    total: Decimal


def spend_by_vendor_month(
    session: Session, start: date | None = None, end: date | None = None
) -> list[VendorMonthSpend]:
    """Invoiced spend per vendor per month, invoice dates from ``start`` to ``end``
    inclusive (``None``: unbounded), in month then vendor order."""
    return [
        VendorMonthSpend(
            vendor_id=row.vendor_id,
            vendor_name=row.vendor_name,
            month=row.month,
            currency=row.currency,
            invoice_count=int(row.invoice_count),
            total=Decimal(row.total),
        )
        for row in run_query(session, "spend_by_vendor_month", start=start, end=end)
    ]


# --------------------------------------------------------------------------- unpaid


@dataclass(frozen=True)
class UnpaidTotal:
    vendor_id: str | None
    vendor_name: str
    currency: str
    open_invoices: int
    outstanding: Decimal
    overdue_invoices: int
    overdue: Decimal
    """The part of ``outstanding`` that was past due."""
    oldest_due_date: date


def unpaid_totals(session: Session, as_of: date) -> list[UnpaidTotal]:
    """What each vendor was still owed on ``as_of``, per currency, largest first."""
    return [
        UnpaidTotal(
            vendor_id=row.vendor_id,
            vendor_name=row.vendor_name,
            currency=row.currency,
            open_invoices=int(row.open_invoices),
            outstanding=Decimal(row.outstanding),
            overdue_invoices=int(row.overdue_invoices),
            overdue=Decimal(row.overdue),
            oldest_due_date=row.oldest_due_date,
        )
        for row in run_query(session, "unpaid_totals", as_of=as_of)
    ]


# --------------------------------------------------------------------------- days to pay


@dataclass(frozen=True)
class DaysToPay:
    vendor_id: str | None
    vendor_name: str
    invoices_paid: int
    avg_days_to_pay: Decimal
    """From invoice date to the payment that settled it."""
    avg_days_late: Decimal | None
    """From due date to settlement (negative: early); ``None`` if no invoice had a due date."""
    max_days_to_pay: int


def days_to_pay(
    session: Session, start: date | None = None, end: date | None = None
) -> list[DaysToPay]:
    """Average days to pay per vendor, over invoices settled from ``start`` to ``end``
    inclusive (``None``: unbounded)."""
    return [
        DaysToPay(
            vendor_id=row.vendor_id,
            vendor_name=row.vendor_name,
            invoices_paid=int(row.invoices_paid),
            avg_days_to_pay=Decimal(row.avg_days_to_pay),
            avg_days_late=None if row.avg_days_late is None else Decimal(row.avg_days_late),
            max_days_to_pay=int(row.max_days_to_pay),
        )
        for row in run_query(session, "days_to_pay", start=start, end=end)
    ]


def average_days_to_pay(rows: Sequence[DaysToPay]) -> Decimal | None:
    """Across vendors, weighted by invoices paid (``None`` when nothing was paid)."""
    paid = sum(row.invoices_paid for row in rows)
    if not paid:
        return None
    return sum((row.avg_days_to_pay * row.invoices_paid for row in rows), Decimal(0)) / paid
