import { describe, expect, it } from "vitest";

import { parseDocumentsQuery, documentsHref } from "@/lib/documents-query";
import { describeRecord, EMPTY, formatValue } from "@/lib/fields";
import {
  formatBytes,
  formatDate,
  formatDateTime,
  formatMoney,
  formatUsd,
  humanize,
} from "@/lib/format";

describe("format", () => {
  it("formats money in its currency, and an unknown code after the amount", () => {
    expect(formatMoney("1234.5", "USD")).toBe("$1,234.50");
    expect(formatMoney(-12, "EUR")).toBe("-€12.00");
    expect(formatMoney("10", "XX1")).toBe("10.00 XX1");
    expect(formatMoney("n/a", "USD")).toBe("n/a USD");
  });

  it("keeps cents-level LLM costs readable", () => {
    expect(formatUsd(0.0042)).toBe("$0.0042");
    expect(formatUsd(0)).toBe("$0.00");
    expect(formatUsd(12.5)).toBe("$12.50");
  });

  it("reads a calendar day as that day whatever the machine's zone", () => {
    // TZ is America/Los_Angeles in vitest.config.mts: local parsing would give Nov 20.
    expect(formatDate("2026-11-21")).toBe("Nov 21, 2026");
    expect(formatDateTime("2026-11-21T02:05:00Z")).toBe("Nov 21, 2026, 02:05 UTC");
    expect(formatDate("not a date")).toBe("not a date");
  });

  it("humanizes field names", () => {
    expect(humanize("invoice_number")).toBe("Invoice number");
    expect(humanize("po_number")).toBe("PO number");
    expect(humanize("vendor_tax_id")).toBe("Vendor tax ID");
  });

  it("formats sizes", () => {
    expect(formatBytes(900)).toBe("900 B");
    expect(formatBytes(5_500)).toBe("5.4 KB");
    expect(formatBytes(25 * 1024 * 1024)).toBe("25 MB");
  });
});

describe("extracted fields", () => {
  it("formats values by what they are", () => {
    expect(formatValue("total", "134.56", "MYR").replace(/\s/g, " ")).toBe("MYR 134.56");
    expect(formatValue("quantity", "1.0000", "MYR")).toBe("1");
    expect(formatValue("invoice_date", "2026-10-05")).toBe("Oct 5, 2026");
    expect(formatValue("due_date", null)).toBe(EMPTY);
    expect(formatValue("paid", true)).toBe("Yes");
    expect(formatValue("tags", ["a", "b"])).toBe("a, b");
  });

  it("lists scalars and turns lists of objects into tables", () => {
    const view = describeRecord({
      invoice_number: "INV-1",
      currency: "USD",
      total: "120.00",
      due_date: null,
      line_items: [
        { description: "Paper", quantity: "2", amount: "100.00" },
        { description: "Ink", quantity: "1", amount: "20.00", unit_price: "20.00" },
      ],
    });
    expect(view.fields.map((f) => [f.label, f.value])).toEqual([
      ["Invoice number", "INV-1"],
      ["Currency", "USD"],
      ["Total", "$120.00"],
      ["Due date", EMPTY],
    ]);
    expect(view.fields[3]?.empty).toBe(true);
    const [table] = view.tables;
    expect(table?.label).toBe("Line items");
    expect(table?.columns.map((c) => [c.label, c.numeric])).toEqual([
      ["Description", false],
      ["Quantity", true],
      ["Amount", true],
      ["Unit price", true],
    ]);
    expect(table?.rows).toEqual([
      ["Paper", "2", "$100.00", EMPTY],
      ["Ink", "1", "$20.00", "$20.00"],
    ]);
  });
});

describe("documents query", () => {
  it("keeps known statuses and positive pages only", () => {
    expect(parseDocumentsQuery({ status: "ready", page: "3" })).toEqual({ status: "ready", page: 3 });
    expect(parseDocumentsQuery({ status: "done", page: "-2" })).toEqual({ status: null, page: 1 });
    expect(parseDocumentsQuery({ status: ["failed", "ready"] })).toEqual({ status: "failed", page: 1 });
  });

  it("builds the list's URL without defaults", () => {
    expect(documentsHref({ status: null, page: 1 })).toBe("/documents");
    expect(documentsHref({ status: "queued", page: 2 })).toBe("/documents?status=queued&page=2");
  });
});
