---
name: reminder_final_notice
version: 1
description: Final notice for an invoice 46 or more days overdue; formal tone. States the invoice number, amount, original due date and PO number.
variables: [recipient, invoice_number, currency, amount, total, due_date, po_number, days_overdue, sender]
---
Dear $recipient,

FINAL NOTICE: invoice $invoice_number (PO $po_number)

Despite earlier reminders, invoice $invoice_number for $currency $total, originally due on
$due_date, is now $days_overdue days overdue with $currency $amount outstanding.

Please settle the outstanding amount within 5 working days. If payment is not received or
a dispute raised in that time, the matter will be escalated in line with the agreed terms.

$sender
