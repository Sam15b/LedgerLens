-- LedgerLens analytics: accounts-payable reports over the Postgres system of record.
--
-- Each query starts with a "-- name: <query>" line; ledgerlens.storage.postgres.analytics
-- loads them by name and returns typed rows. Parameters are :named binds.
--
-- Conventions shared by every query:
--   * Amounts are never added across currencies: every report is grouped by currency.
--   * "Paid" means payment allocations matched to the invoice (payment_allocations.invoice_id)
--     from payments made on or before :as_of, so a report for a past date shows what was
--     owed then. An invoice marked paid with no matched allocation at all (a till receipt,
--     or settled outside the payments table) counts as settled; void invoices never count.
--   * Amounts within 0.01 of zero are rounding, not money owed (ROUNDING_TOLERANCE).

-- name: ap_aging
-- Money owed on :as_of, bucketed by days past due: current (not yet due), 0-30, 31-60,
-- 61-90 and 90+. An invoice without a due date is aged from its invoice date.
WITH paid AS (
    SELECT a.invoice_id, SUM(a.amount) AS paid
    FROM payment_allocations a
    JOIN payments p ON p.id = a.payment_id
    WHERE a.invoice_id IS NOT NULL AND p.paid_at <= CAST(:as_of AS date)
    GROUP BY a.invoice_id
),
open_invoices AS (
    SELECT i.currency,
           i.total - COALESCE(paid.paid, 0) AS outstanding,
           CAST(:as_of AS date) - COALESCE(i.due_date, i.invoice_date) AS days_past_due
    FROM invoices i
    LEFT JOIN paid ON paid.invoice_id = i.id
    WHERE i.status <> 'void'
      AND i.invoice_date <= CAST(:as_of AS date)
      AND NOT (
          i.status = 'paid'
          AND NOT EXISTS (SELECT 1 FROM payment_allocations a WHERE a.invoice_id = i.id)
      )
)
SELECT CASE
           WHEN days_past_due < 0 THEN 'current'
           WHEN days_past_due <= 30 THEN '0-30'
           WHEN days_past_due <= 60 THEN '31-60'
           WHEN days_past_due <= 90 THEN '61-90'
           ELSE '90+'
       END AS bucket,
       currency,
       COUNT(*) AS invoice_count,
       SUM(outstanding) AS outstanding
FROM open_invoices
WHERE outstanding > 0.01
GROUP BY 1, currency
ORDER BY currency, MIN(days_past_due);

-- name: spend_by_vendor_month
-- Invoiced spend per vendor per calendar month (by invoice date), between :start and
-- :end inclusive (either may be NULL for an open range). Vendors not yet resolved to a
-- vendors row are grouped by the name printed on the invoice.
SELECT i.vendor_id,
       COALESCE(v.canonical_name, i.vendor_name) AS vendor_name,
       CAST(date_trunc('month', i.invoice_date) AS date) AS month,
       i.currency,
       COUNT(*) AS invoice_count,
       SUM(i.total) AS total
FROM invoices i
LEFT JOIN vendors v ON v.id = i.vendor_id
WHERE i.status <> 'void'
  AND (CAST(:start AS date) IS NULL OR i.invoice_date >= CAST(:start AS date))
  AND (CAST(:end AS date) IS NULL OR i.invoice_date <= CAST(:end AS date))
GROUP BY i.vendor_id, COALESCE(v.canonical_name, i.vendor_name), 3, i.currency
ORDER BY month, vendor_name, i.currency;
