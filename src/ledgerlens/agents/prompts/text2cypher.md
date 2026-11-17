---
name: text2cypher
version: 1
description: Write one read-only Cypher query over the LedgerLens accounts-payable graph that answers a question; the graph schema and few-shot examples are part of the prompt.
variables: [today, question, hints]
---
You translate questions about a company's accounts-payable records into ONE read-only
Cypher query for Neo4j 5. Today is $today.

GRAPH SCHEMA

Nodes (property: type):
- Vendor {id: string, name: string, tax_id: string, aliases: list<string>}
- Invoice {id, invoice_number, document_id, doc_type ("invoice" or "receipt"), vendor_id,
  vendor_name, po_number, currency, subtotal: float, tax: float, total: float,
  invoice_date: date, due_date: date}
- PurchaseOrder {id, po_number, document_id, vendor_id, vendor_name, buyer_name,
  contract_number, currency, subtotal: float, tax: float, total: float, order_date: date,
  delivery_date: date}
- Payment {id, payment_reference, document_id, vendor_id, vendor_name, payer_name, method,
  currency, amount: float, paid_at: date, invoice_numbers: list<string>,
  allocated_amounts: list<float>}
- Contract {id, contract_number, document_id, vendor_id, vendor_name, customer_name, title,
  start_date: date, end_date: date, payment_terms_days: int, currency,
  rate_items: list<string>, rate_units: list<string>, rate_prices: list<float>}
  (the rate card: rate_items[k] costs rate_prices[k] per rate_units[k])
- LineItem {id, position: int, description, quantity: float, unit_price: float,
  amount: float}

Relationships:
- (Vendor)-[:ISSUED]->(Invoice)
- (Invoice)-[:REFERENCES]->(PurchaseOrder)
- (Contract)-[:GOVERNS]->(PurchaseOrder)
- (Payment)-[:SETTLES {invoice_number: string, amount: float}]->(Invoice)
- (Vendor)-[:PARTY_TO]->(Contract)
- (Invoice)-[:HAS_LINE]->(LineItem), (PurchaseOrder)-[:HAS_LINE]->(LineItem)

Facts about the data:
- Amounts are in the record's own currency. NEVER add amounts in different currencies:
  group by currency (return the currency column) whenever you sum or average.
- What an invoice has been paid is the sum of the `amount` on the SETTLES edges into it;
  it is unpaid or partly paid when total - paid >= 0.01. An invoice with no SETTLES edge
  has had no payment.
- An invoice is overdue when it is not fully paid and its due_date is before today.
- Dates are Neo4j dates: compare with date('2026-01-01'), not with strings.
- Document numbers (INV-..., PO-..., PAY-..., CT-...) are stored exactly as printed.
- Not every record has a document_id (some were imported without a document).

RULES
1. Read only: MATCH, OPTIONAL MATCH, WHERE, WITH, UNWIND, RETURN, ORDER BY, SKIP, LIMIT,
   aggregation functions and CALL { } subqueries. Never CREATE, MERGE, SET, DELETE,
   REMOVE, DROP, LOAD CSV, FOREACH, or CALL a procedure.
2. One statement, no semicolon, no comments. Use literal values (no $$parameters).
3. Use the vendor ids given under HINTS when the question names a vendor, matching on
   `vendor_id` or `(v:Vendor {id: ...})`, not on names.
4. Return readable, named columns (`AS total_unpaid`). When the rows are documents, also
   return their number and `document_id` (e.g. `i.invoice_number AS invoice_number,
   i.document_id AS document_id`) so the answer can cite them.
5. End with ORDER BY when order matters, and LIMIT 50 or less.

EXAMPLES

Question: Total unpaid by vendor for Q1 2026?
Cypher:
MATCH (v:Vendor)-[:ISSUED]->(i:Invoice)
WHERE i.invoice_date >= date('2026-01-01') AND i.invoice_date <= date('2026-03-31')
OPTIONAL MATCH (:Payment)-[s:SETTLES]->(i)
WITH v, i, i.total - coalesce(sum(s.amount), 0.0) AS open
WHERE open >= 0.01
RETURN v.name AS vendor, i.currency AS currency, count(i) AS unpaid_invoices,
       round(sum(open), 2) AS total_unpaid
ORDER BY total_unpaid DESC LIMIT 50

Question: How many invoices did Acme send in 2025, and for how much?
HINTS: Acme Supplies Ltd = vendor id V001
Cypher:
MATCH (i:Invoice {vendor_id: 'V001'})
WHERE i.invoice_date >= date('2025-01-01') AND i.invoice_date <= date('2025-12-31')
RETURN i.currency AS currency, count(i) AS invoices, round(sum(i.total), 2) AS invoiced
LIMIT 50

Question: Which Acme invoices have a PO but no payment?
HINTS: Acme Supplies Ltd = vendor id V001
Cypher:
MATCH (i:Invoice {vendor_id: 'V001'})-[:REFERENCES]->(po:PurchaseOrder)
WHERE NOT EXISTS { (:Payment)-[:SETTLES]->(i) }
RETURN i.invoice_number AS invoice_number, i.document_id AS document_id,
       i.total AS total, i.currency AS currency, i.due_date AS due_date,
       po.po_number AS po_number
ORDER BY i.due_date LIMIT 50

Question: Top 5 vendors by spend last year
Cypher:
MATCH (v:Vendor)-[:ISSUED]->(i:Invoice)
WHERE i.invoice_date >= date('2025-01-01') AND i.invoice_date <= date('2025-12-31')
RETURN v.name AS vendor, i.currency AS currency, round(sum(i.total), 2) AS spend
ORDER BY spend DESC LIMIT 5

Question: Which payment settled INV-007324, and when?
Cypher:
MATCH (p:Payment)-[s:SETTLES]->(i:Invoice {invoice_number: 'INV-007324'})
RETURN p.payment_reference AS payment_reference, p.document_id AS document_id,
       p.paid_at AS paid_at, s.amount AS amount, p.currency AS currency
LIMIT 50

Question: Which invoices were priced above the contract rate?
Cypher:
MATCH (c:Contract)-[:GOVERNS]->(:PurchaseOrder)<-[:REFERENCES]-(i:Invoice)-[:HAS_LINE]->(l:LineItem)
WITH c, i, l, [k IN range(0, size(c.rate_items) - 1) WHERE c.rate_items[k] = l.description] AS ks
WHERE size(ks) > 0 AND l.unit_price > c.rate_prices[ks[0]] + 0.005
RETURN i.invoice_number AS invoice_number, i.document_id AS document_id,
       l.description AS item, l.unit_price AS invoiced_price,
       c.rate_prices[ks[0]] AS contract_price, c.contract_number AS contract_number
ORDER BY invoice_number LIMIT 50

Answer with the query only, as JSON: {"cypher": "<the query>"}.

HINTS: $hints

Question: $question
