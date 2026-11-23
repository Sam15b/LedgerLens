import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { DocumentsTable } from "@/components/DocumentsTable";
import { ExtractedFields } from "@/components/ExtractedFields";
import { inProgress, StatusBadge } from "@/components/StatusBadge";
import type { DocumentOut } from "@/lib/api";

function doc(fields: Partial<DocumentOut> = {}): DocumentOut {
  return {
    id: "6f1c2a8e-4b7d-4c1e-9a3f-0d2b5e7c9a10",
    filename: "acme-invoice.pdf",
    mime_type: "application/pdf",
    doc_type: "invoice",
    status: "ready",
    pages: 2,
    sha256: "a".repeat(64),
    created_at: "2026-11-21T14:03:00Z",
    ...fields,
  };
}

describe("StatusBadge", () => {
  it("names the status in words, with the failure reason as its title", () => {
    render(<StatusBadge status="failed" title="ocr: no text" />);
    const badge = screen.getByText("Failed");
    expect(badge.getAttribute("title")).toBe("ocr: no text");
  });

  it("treats queued and processing as still in progress", () => {
    expect(["queued", "processing", "ready", "failed"].map((s) => inProgress(s as never))).toEqual([
      true,
      true,
      false,
      false,
    ]);
  });
});

describe("DocumentsTable", () => {
  it("lists each document with a link to its viewer, type, status and pages", () => {
    render(
      <DocumentsTable
        documents={[
          doc(),
          doc({
            id: "7f1c2a8e-4b7d-4c1e-9a3f-0d2b5e7c9a10",
            filename: "scan.png",
            doc_type: "purchase_order",
            status: "processing",
            pages: null,
          }),
        ]}
      />,
    );
    const rows = screen.getAllByRole("row").slice(1);
    expect(rows).toHaveLength(2);

    const first = within(rows[0]!);
    const link = first.getByRole("link", { name: "acme-invoice.pdf" });
    expect(link.getAttribute("href")).toBe("/documents/6f1c2a8e-4b7d-4c1e-9a3f-0d2b5e7c9a10");
    expect(first.getByText("Invoice")).toBeTruthy();
    expect(first.getByText("Ready")).toBeTruthy();
    expect(first.getByText("Nov 21, 2026, 14:03 UTC")).toBeTruthy();

    const second = within(rows[1]!);
    expect(second.getByText("Purchase order")).toBeTruthy();
    expect(second.getByText("Processing")).toBeTruthy();
    expect(second.getByText("–")).toBeTruthy();
  });

  it("says what to do when there is nothing yet", () => {
    render(<DocumentsTable documents={[]} />);
    expect(screen.queryByRole("table")).toBeNull();
    expect(screen.getByText(/No documents here yet/)).toBeTruthy();
  });
});

describe("ExtractedFields", () => {
  it("shows the record's fields and its line items as a table", () => {
    render(
      <ExtractedFields
        recordType="invoice"
        recordId="INV-7"
        record={{
          invoice_number: "INV-7",
          vendor_name: "Acme Supplies",
          currency: "EUR",
          total: "240.00",
          line_items: [{ description: "Toner", quantity: "2.000", amount: "240.00" }],
        }}
      />,
    );
    expect(screen.getByText("Vendor name").nextElementSibling?.textContent).toBe("Acme Supplies");
    expect(screen.getByText("Total").nextElementSibling?.textContent).toBe("€240.00");
    const table = screen.getByRole("table", { name: "Line items" });
    const cells = within(table)
      .getAllByRole("cell")
      .map((cell) => cell.textContent);
    expect(cells).toEqual(["Toner", "2", "€240.00"]);
  });
});
