// LedgerLens knowledge-graph schema: uniqueness constraints and lookup indexes.
// Every statement is idempotent (IF NOT EXISTS), so this file can be applied repeatedly.
// Applied by scripts/init_neo4j_schema.py. See docs/layers/02-storage.md.

// --- Node identity ------------------------------------------------------------------
CREATE CONSTRAINT vendor_id IF NOT EXISTS FOR (v:Vendor) REQUIRE v.id IS UNIQUE;
CREATE CONSTRAINT invoice_id IF NOT EXISTS FOR (i:Invoice) REQUIRE i.id IS UNIQUE;
CREATE CONSTRAINT po_id IF NOT EXISTS FOR (p:PurchaseOrder) REQUIRE p.id IS UNIQUE;
CREATE CONSTRAINT payment_id IF NOT EXISTS FOR (p:Payment) REQUIRE p.id IS UNIQUE;
CREATE CONSTRAINT contract_id IF NOT EXISTS FOR (c:Contract) REQUIRE c.id IS UNIQUE;
CREATE CONSTRAINT line_item_id IF NOT EXISTS FOR (l:LineItem) REQUIRE l.id IS UNIQUE;

// --- Linking keys: document numbers referenced across documents --------------------
CREATE INDEX invoice_number IF NOT EXISTS FOR (i:Invoice) ON (i.invoice_number);
CREATE INDEX po_number IF NOT EXISTS FOR (p:PurchaseOrder) ON (p.po_number);
CREATE INDEX vendor_name IF NOT EXISTS FOR (v:Vendor) ON (v.name);

// --- document_id: maps graph nodes back to Pinecone chunks and Postgres rows --------
CREATE INDEX invoice_document_id IF NOT EXISTS FOR (i:Invoice) ON (i.document_id);
CREATE INDEX po_document_id IF NOT EXISTS FOR (p:PurchaseOrder) ON (p.document_id);
CREATE INDEX contract_document_id IF NOT EXISTS FOR (c:Contract) ON (c.document_id);
