import { render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { OverviewCards } from "@/components/OverviewCards";
import type { Overview } from "@/lib/api";

vi.mock("next/navigation", () => ({ usePathname: () => "/" }));

function overview(fields: Partial<Overview> = {}): Overview {
  return {
    as_of: "2026-11-23",
    documents: { total: 12, queued: 1, processing: 1, ready: 9, failed: 1 },
    unpaid: [
      { currency: "USD", open_invoices: 4, outstanding: "1500.00", overdue_invoices: 2, overdue: "600.00" },
      { currency: "EUR", open_invoices: 1, outstanding: "80.00", overdue_invoices: 0, overdue: "0" },
    ],
    open_invoices: 5,
    overdue_invoices: 2,
    flagged_invoices: 3,
    pending_reminders: 0,
    llm_cost_by_day: [
      { day: "2026-11-22", calls: 0, cost_usd: 0 },
      { day: "2026-11-23", calls: 40, cost_usd: 0.0123 },
    ],
    llm_costs_recorded: true,
    ...fields,
  };
}

function card(label: string) {
  const heading = screen.getByText(label);
  return within(heading.parentElement!);
}

describe("OverviewCards", () => {
  it("shows the headline numbers", () => {
    render(<OverviewCards overview={overview()} />);
    expect(card("Documents ingested").getByText("9")).toBeTruthy();
    expect(card("Documents ingested").getByText(/of 12 uploads · 2 in progress/)).toBeTruthy();
    expect(card("Unpaid").getByText("$1,500.00")).toBeTruthy();
    expect(card("Unpaid").getByText("+ €80.00")).toBeTruthy();
    expect(card("Overdue").getByText("2")).toBeTruthy();
    expect(card("Overdue").getByText("$600.00 past due")).toBeTruthy();
    expect(card("Flagged invoices").getByText("3")).toBeTruthy();
    expect(card("Reminders to approve").getByText("0")).toBeTruthy();
  });

  it("links only to pages that exist", () => {
    render(<OverviewCards overview={overview()} />);
    const links = screen.getAllByRole("link").map((a) => a.getAttribute("href"));
    expect(links).toEqual(["/documents", "/anomalies", "/reminders"]);
  });

  it("charts LLM cost per day with a label per bar and a table view", () => {
    render(<OverviewCards overview={overview()} />);
    const header = screen.getByRole("heading", { name: /LLM cost per day/ }).parentElement!;
    expect(header.querySelector("p")?.textContent).toBe("$0.0123 · 40 calls");
    expect(screen.getByLabelText("Nov 23: $0.0123, 40 calls")).toBeTruthy();
    const table = screen.getByRole("table");
    expect(within(table).getAllByRole("row")).toHaveLength(3);
  });

  it("explains an empty cost chart instead of showing $0", () => {
    render(<OverviewCards overview={overview({ llm_costs_recorded: false })} />);
    expect(screen.getByText(/No LLM calls are recorded/)).toBeTruthy();
    expect(screen.queryByRole("table")).toBeNull();
  });

  it("shows a zero unpaid total when nothing is open", () => {
    render(<OverviewCards overview={overview({ unpaid: [], open_invoices: 0, overdue_invoices: 0 })} />);
    expect(card("Unpaid").getByText("$0.00")).toBeTruthy();
    expect(card("Overdue").getByText("Nothing past due")).toBeTruthy();
  });
});
