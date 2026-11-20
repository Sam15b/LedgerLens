---
name: reminder_firm
version: 1
description: Payment reminder for an invoice 16-45 days overdue; firm tone. States the invoice number, amount, original due date and PO number.
variables: [recipient, invoice_number, currency, amount, total, due_date, po_number, days_overdue, sender]
---
Dear $recipient,

Invoice $invoice_number (PO $po_number) was due on $due_date and is now $days_overdue
days overdue. $currency $amount remains outstanding.

Please arrange settlement of the outstanding amount within the next 7 days, or tell us by
return what is preventing payment so that it can be resolved.

Regards,
$sender
