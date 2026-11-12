Sample documents for the ingestion tests, each with the record it should extract to.

| File | Kind | Pages | Text layer | Expected record |
|------|------|------:|------------|-----------------|
| `invoice_digital.pdf` | Invoice INV-2026-0417 | 1 | yes | `invoice_digital.json` (`Invoice`) |
| `purchase_order.pdf` | Purchase order PO-88231 | 1 | yes | `purchase_order.json` (`PurchaseOrder`) |
| `contract.pdf` | Master supply agreement MSA-2026-07 + rate card | 2 | yes | `contract.json` (`Contract`) |
| `payment_advice.pdf` | Remittance advice RA-5531 | 1 | yes | `payment_advice.json` (`Payment`) |
| `receipt_scanned.png` | Cafe receipt, rasterised, tilted and speckled | 1 | no | `receipt_scanned.json` (`Receipt`) |

The first four are linked the way the reconciler expects: the PO is raised under the
contract and priced from its rate card, the invoice bills the PO, and the payment settles
the invoice. All names, addresses and numbers are made up.

Regenerate (deterministic: same bytes every run) with:

    python tests/fixtures/documents/make_fixtures.py
