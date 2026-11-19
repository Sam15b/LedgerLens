---
name: reminder_friendly
version: 1
description: Payment reminder for an invoice 1-15 days overdue; friendly tone. States the invoice number, amount, original due date and PO number.
variables: [recipient, invoice_number, currency, amount, total, due_date, po_number, days_overdue, sender]
---
Dear $recipient,

I hope all is well. This is a quick note about invoice $invoice_number (PO $po_number),
which was due on $due_date. Our records show $currency $amount still open on it, now
$days_overdue days past the due date.

If it has already been settled, thank you, and please ignore this note. If anything is
holding it up, just let us know and we will sort it out together.

Kind regards,
$sender
