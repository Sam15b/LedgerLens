import { act, fireEvent, render, screen, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { type Anomaly, AnomalyCard, readReasons } from "@/components/AnomalyCard";
import { EvalDashboard } from "@/components/EvalDashboard";
import { type Reminder, ReminderCard } from "@/components/ReminderCard";
import { ReviewActions } from "@/components/ReviewActions";
import { bySuite, change, type EvalRun, formatMetric, metricLabel } from "@/lib/eval";

const refresh = vi.fn();
vi.mock("next/navigation", () => ({ useRouter: () => ({ refresh }) }));

const http = vi.hoisted(() => ({ POST: vi.fn(), PATCH: vi.fn() }));
vi.mock("@/lib/api", async (importOriginal) => {
  const real = await importOriginal<typeof import("@/lib/api")>();
  return { ...real, api: { ...http, GET: vi.fn() } };
});

function ok(data: unknown) {
  return { data, response: new Response(null, { status: 200 }) };
}
function refused(status: number, message: string) {
  return { error: { error: { code: "x", message, request_id: "r1" } }, response: new Response(null, { status }) };
}

beforeEach(() => {
  refresh.mockClear();
  http.POST.mockReset();
  http.PATCH.mockReset();
});

const ANOMALY: Anomaly = {
  id: 849,
  invoice_id: "SINV-000842",
  invoice_number: "INV-005905",
  document_id: "6f1c2a8e-4b7d-4c1e-9a3f-0d2b5e7c9a10",
  vendor_id: "V045",
  vendor_name: "Telekom Nusantara",
  invoice_date: "2025-05-01",
  currency: "IDR",
  total: "50000000.0000",
  model_version: "ae-20261119",
  score: 0.9379,
  flagged: true,
  reasons: [
    { code: "log_mean_unit_price", text: "unit prices unusually low", share: 0.24, source: "autoencoder", related_ids: [] },
    { code: "over_po", text: "total is over what is left on PO-2025-00742", share: null, source: "rule", related_ids: ["PO-2025-00742"] },
  ],
  review: null,
  scored_at: "2026-11-19T19:21:00Z",
};

describe("anomalies", () => {
  it("shows the invoice, score and reasons, rules first", () => {
    render(<AnomalyCard anomaly={ANOMALY} />);
    expect(screen.getByRole("link", { name: "INV-005905" }).getAttribute("href")).toBe(
      "/documents/6f1c2a8e-4b7d-4c1e-9a3f-0d2b5e7c9a10",
    );
    expect(screen.getByText("0.94")).toBeTruthy();
    const reasons = within(screen.getByRole("list", { name: "Reasons" })).getAllByRole("listitem");
    expect(reasons[0]?.textContent).toContain("Rule");
    expect(reasons[0]?.textContent).toContain("PO-2025-00742");
    expect(reasons[1]?.textContent).toContain("(24% of the error)");
  });

  it("orders reasons by source, then share", () => {
    const order = readReasons([
      { code: "a", text: "a", share: 0.1, source: "autoencoder" },
      { code: "b", text: "b", share: 0.5, source: "autoencoder" },
      { code: "r", text: "r", source: "rule" },
    ]).map((r) => r.code);
    expect(order).toEqual(["r", "b", "a"]);
  });

  it("records a verdict with its note and refreshes the list", async () => {
    http.POST.mockResolvedValue(ok({ ...ANOMALY, review: "false_positive" }));
    render(<ReviewActions scoreId={849} />);
    fireEvent.change(screen.getByLabelText("Review note"), { target: { value: " seasonal bulk order " } });
    await act(async () => fireEvent.click(screen.getByRole("button", { name: "False positive" })));
    expect(http.POST).toHaveBeenCalledWith("/anomalies/{score_id}/review", {
      params: { path: { score_id: 849 } },
      body: { verdict: "false_positive", note: "seasonal bulk order" },
    });
    expect(screen.getByText(/Marked a false positive/)).toBeTruthy();
    expect(refresh).toHaveBeenCalled();
  });

  it("shows a refused verdict and keeps the buttons", async () => {
    http.POST.mockResolvedValue(refused(404, "no anomaly score 849"));
    render(<ReviewActions scoreId={849} />);
    await act(async () => fireEvent.click(screen.getByRole("button", { name: "Confirm anomaly" })));
    expect(http.POST.mock.calls[0]?.[1].body).toEqual({ verdict: "confirmed", note: null });
    expect(screen.getByRole("alert").textContent).toBe("no anomaly score 849 (request r1)");
    expect(screen.getByRole("button", { name: "Confirm anomaly" })).toBeTruthy();
  });
});

const REMINDER: Reminder = {
  id: "3afec327-20be-42f6-bc06-38b0f4f7c4a2",
  invoice_id: "SINV-000006",
  invoice_number: "INV-005307",
  vendor_name: "Kopi Sentosa Trading",
  currency: "MYR",
  total: "555.88",
  due_date: "2025-01-13",
  tone: "final_notice",
  status: "pending_approval",
  draft: "Dear Kopi Sentosa,\nPlease pay.",
  thread_id: "chat-1",
  created_at: "2026-11-23T15:40:00Z",
};

describe("reminders", () => {
  it("shows what the draft is about", () => {
    render(<ReminderCard reminder={REMINDER} />);
    expect(screen.getByText("INV-005307")).toBeTruthy();
    expect(screen.getByText("Final notice")).toBeTruthy();
    expect(screen.getByText(/MYR\s555\.88 · due Jan 13, 2025/)).toBeTruthy();
    expect(screen.getByText("Waiting for approval")).toBeTruthy();
  });

  it("saves an edited draft before it can be approved", async () => {
    http.PATCH.mockResolvedValue(ok({ ...REMINDER, draft: "Dear Kopi,\nPay now." }));
    http.POST.mockResolvedValue(ok({ reminder: { ...REMINDER, status: "approved" }, resumed: false }));
    render(<ReminderCard reminder={REMINDER} />);
    const approve = screen.getByRole("button", { name: "Approve" }) as HTMLButtonElement;

    fireEvent.change(screen.getByLabelText("Draft for INV-005307"), { target: { value: "Dear Kopi,\nPay now." } });
    expect(approve.disabled).toBe(true);
    await act(async () => fireEvent.click(screen.getByRole("button", { name: "Save changes" })));
    expect(http.PATCH).toHaveBeenCalledWith("/reminders/{reminder_id}", {
      params: { path: { reminder_id: REMINDER.id } },
      body: { draft: "Dear Kopi,\nPay now." },
    });
    expect(screen.getByRole("button", { name: "Saved" })).toBeTruthy();
    expect(approve.disabled).toBe(false);

    await act(async () => fireEvent.click(approve));
    expect(http.POST).toHaveBeenCalledWith("/reminders/{reminder_id}/approve", {
      params: { path: { reminder_id: REMINDER.id } },
    });
    expect(screen.getByText("Approved · not sent")).toBeTruthy();
    expect(screen.queryByRole("textbox")).toBeNull();
    expect(screen.getByText(/Pay now/)).toBeTruthy();
    expect(refresh).toHaveBeenCalled();
  });

  it("undoes unsaved changes and shows a refused decision", async () => {
    http.POST.mockResolvedValue(refused(409, "reminder is already approved"));
    render(<ReminderCard reminder={REMINDER} />);
    const box = screen.getByLabelText("Draft for INV-005307") as HTMLTextAreaElement;
    fireEvent.change(box, { target: { value: "changed" } });
    fireEvent.click(screen.getByRole("button", { name: "Undo changes" }));
    expect(box.value).toBe(REMINDER.draft);
    await act(async () => fireEvent.click(screen.getByRole("button", { name: "Reject" })));
    expect(screen.getByRole("alert").textContent).toBe("reminder is already approved (request r1)");
  });

  it("shows a decided draft read-only", () => {
    render(<ReminderCard reminder={{ ...REMINDER, status: "rejected", decided_at: "2026-11-23T16:00:00Z" }} />);
    expect(screen.queryByRole("textbox")).toBeNull();
    expect(screen.queryByRole("button")).toBeNull();
    expect(screen.getByText("Rejected")).toBeTruthy();
  });
});

function run(id: string, suite: string, day: number, metrics: EvalRun["metrics"]): EvalRun {
  return { id, suite, git_sha: `${id}${"0".repeat(39)}`, config: {}, metrics, created_at: `2026-11-${day}T15:00:00Z` };
}

const RUNS = [
  run("a", "full", 21, { "recall@5": 0.882, mrr: 0.899, "latency.retrieval.p95_ms": 13355.2, questions: 200 }),
  run("b", "full", 22, { "recall@5": 0.954, mrr: 0.929, "latency.retrieval.p95_ms": 13525.3, questions: 200, facts_found: null }),
  run("c", "ci", 22, { "recall@5": 0.967, mrr: 0.936 }),
];

describe("eval", () => {
  it("formats and names metrics", () => {
    expect(formatMetric("recall@5", 0.95444)).toBe("0.954");
    expect(formatMetric("latency.retrieval.p95_ms", 13525.3)).toBe("13.5 s");
    expect(formatMetric("latency.query.p50_ms", 840)).toBe("840 ms");
    expect(formatMetric("cost.query.mean_usd", 0.0042)).toBe("$0.0042");
    expect(formatMetric("questions", 200)).toBe("200");
    expect(metricLabel("recall@5.multi_hop")).toBe("Recall@5, multi-hop");
    expect(metricLabel("latency.retrieval.p95_ms")).toBe("Retrieval latency p95");
  });

  it("knows which way is better", () => {
    expect(change("recall@5", 0.95, 0.88)).toMatchObject({ better: true });
    expect(change("latency.retrieval.p95_ms", 13525, 13355)).toMatchObject({ better: false });
    expect(change("mrr", 0.9, 0.9)).toEqual({ delta: 0, better: null });
    expect(change("mrr", 0.9, undefined)).toBeNull();
  });

  it("groups runs by suite, newest first", () => {
    const suites = bySuite(RUNS);
    expect([...suites.keys()]).toEqual(["full", "ci"]);
    expect(suites.get("full")?.map((r) => r.id)).toEqual(["b", "a"]);
  });

  it("shows the latest run against the one before, and the trend", () => {
    const full = bySuite(RUNS).get("full")!;
    render(<EvalDashboard suites={["full", "ci"]} suite="full" runs={full} metric="recall@5" />);

    const recall = screen.getAllByRole("link", { name: /Recall@5/ })[0]!;
    expect(recall.textContent).toContain("0.954");
    expect(recall.textContent).toContain("▲ 0.072 better");
    const latency = screen.getAllByRole("link", { name: /Retrieval latency p95/ })[0]!;
    expect(latency.textContent).toContain("13.5 s");
    expect(latency.textContent).toContain("▲ 170 ms worse"); // slower is worse

    const trend = screen.getByRole("list", { name: "Recall@5" });
    expect(within(trend).getAllByRole("listitem").map((li) => li.getAttribute("aria-label"))).toEqual([
      "Nov 21 · a000000: 0.882",
      "Nov 22 · b000000: 0.954",
    ]);
    expect(screen.getByText("not measured")).toBeTruthy();
    expect(screen.getByRole("link", { name: "ci" }).getAttribute("href")).toBe("/eval?suite=ci&metric=recall%405");
  });
});
