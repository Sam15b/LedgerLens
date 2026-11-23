import { act, fireEvent, render, screen, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { Chat } from "@/components/Chat";
import { CitedText } from "@/components/Citation";
import { summarizeArgs, ToolTimeline } from "@/components/ToolTimeline";
import type { ChatEvent, ChatRequest } from "@/lib/chat";

const runs = vi.hoisted(() => ({
  calls: [] as {
    request: ChatRequest;
    emit: (e: ChatEvent) => void;
    finish: () => void;
    fail: (error: unknown) => void;
  }[],
}));
vi.mock("@/lib/chat", async (importOriginal) => {
  const real = await importOriginal<typeof import("@/lib/chat")>();
  return {
    ...real,
    streamChat: (request: ChatRequest, { onEvent }: { onEvent: (e: ChatEvent) => void }) =>
      new Promise((resolve, reject) => {
        runs.calls.push({
          request,
          emit: onEvent,
          finish: () => resolve({ threadId: "chat-1" }),
          fail: reject,
        });
      }),
  };
});

/** The typed client's POST: the reminder decisions. */
const post = vi.hoisted(() => vi.fn());
vi.mock("@/lib/api", async (importOriginal) => {
  const real = await importOriginal<typeof import("@/lib/api")>();
  return { ...real, api: { POST: post, GET: vi.fn() } };
});

beforeEach(() => {
  runs.calls.length = 0;
  post.mockReset();
});

function ask(text: string) {
  fireEvent.change(screen.getByLabelText("Message"), { target: { value: text } });
  fireEvent.click(screen.getByRole("button", { name: "Send" }));
}

describe("Chat", () => {
  it("streams the answer with its tool calls and citations", async () => {
    render(<Chat />);
    ask("Unpaid from Acme?");
    expect(runs.calls[0]?.request).toEqual({ message: "Unpaid from Acme?", thread_id: null, document_ids: [] });
    expect(screen.getByText("Working…")).toBeTruthy();

    const run = runs.calls[0]!;
    act(() => {
      run.emit({ type: "tool_call", data: { id: "c1", name: "find_invoices", args: { vendor: "Acme" }, node: "qa" } });
      run.emit({ type: "token", data: { text: "INV-1 ", node: "qa" } });
    });
    expect(screen.getByText("1 tool call · 1 running")).toBeTruthy();
    act(() => {
      run.emit({ type: "token", data: { text: "[doc:INV-1]", node: "qa" } });
      run.emit({ type: "citation", data: { tag: "doc:INV-1", document_id: "d-1" } });
      run.emit({ type: "done", data: { thread_id: "chat-1", status: "completed", intent: "question", errors: [] } });
    });
    await act(async () => run.finish());

    expect(screen.getAllByRole("link", { name: "INV-1" })[0]?.getAttribute("href")).toBe("/documents/d-1");
    expect(screen.getByText("Thread", { exact: false }).textContent).toContain("chat-1");

    ask("And from Globex?");
    expect(runs.calls[1]?.request.thread_id).toBe("chat-1");
  });

  it("shows a refused request in the turn", async () => {
    const { ChatError } = await import("@/lib/chat");
    render(<Chat />);
    ask("hello");
    await act(async () => runs.calls[0]!.fail(new ChatError("agents unavailable", 503)));
    expect(screen.getByRole("alert").textContent).toBe("agents unavailable");
    expect((screen.getByRole("button", { name: "Send" }) as HTMLButtonElement).disabled).toBe(true); // empty input

    ask("again");
    await act(async () => runs.calls[1]!.fail(new TypeError("network")));
    expect(screen.getAllByRole("alert")[1]?.textContent).toBe("The answer stream broke off.");
  });

  it("asks for a decision on reminder drafts and shows what the run answered", async () => {
    post.mockResolvedValue({
      data: {
        reminder: {},
        resumed: true,
        thread_status: "interrupted",
        reply: "Rejected: INV-7. Still waiting for a decision: INV-8.",
      },
      response: new Response(null, { status: 200 }),
    });
    render(<Chat />);
    ask("Draft reminders");
    const run = runs.calls[0]!;
    await act(async () => {
      run.emit({
        type: "interrupt",
        data: {
          thread_id: "chat-1",
          type: "reminder_approval",
          reminders: [
            { reminder_id: "r7", invoice_number: "INV-7", tone: "final_notice", body: "Pay INV-7." },
            { reminder_id: "r8", invoice_number: "INV-8", tone: "polite", vendor_email: "ap@acme.test", body: "Pay INV-8." },
          ],
        },
      });
      run.emit({ type: "done", data: { thread_id: "chat-1", status: "interrupted", intent: "reminder", errors: [] } });
      run.finish();
    });

    const card = screen.getByRole("region", { name: "Reminder drafts waiting for approval" });
    expect(within(card).getByText("2 reminder drafts waiting for your decision")).toBeTruthy();
    expect(within(card).getByText("To ap@acme.test")).toBeTruthy();
    expect((screen.getByLabelText("Message") as HTMLTextAreaElement).disabled).toBe(true);

    const [rejectFirst] = within(card).getAllByRole("button", { name: "Reject" });
    await act(async () => fireEvent.click(rejectFirst!));
    expect(post).toHaveBeenCalledWith("/reminders/{reminder_id}/reject", {
      params: { path: { reminder_id: "r7" } },
    });
    expect(within(card).getByText("Rejected.")).toBeTruthy();
    expect(screen.getByText("Rejected: INV-7. Still waiting for a decision: INV-8.")).toBeTruthy();
    expect((screen.getByLabelText("Message") as HTMLTextAreaElement).disabled).toBe(true);

    post.mockResolvedValue({
      data: { reminder: {}, resumed: true, thread_status: "completed", reply: "Approved: INV-8." },
      response: new Response(null, { status: 200 }),
    });
    await act(async () => fireEvent.click(within(card).getByRole("button", { name: "Approve" })));
    expect(within(card).getByText("Approved: not sent, send it yourself.")).toBeTruthy();
    expect((screen.getByLabelText("Message") as HTMLTextAreaElement).disabled).toBe(false);
  });

  it("shows why a decision failed and keeps the buttons", async () => {
    post.mockResolvedValue({
      error: { error: { code: "conflict", message: "already decided", request_id: "r1" } },
      response: new Response(null, { status: 409 }),
    });
    render(<Chat />);
    ask("Draft reminders");
    await act(async () => {
      runs.calls[0]!.emit({
        type: "interrupt",
        data: {
          thread_id: "t",
          type: "reminder_approval",
          reminders: [{ reminder_id: "r1", invoice_number: "INV-1", tone: "polite", body: "x" }],
        },
      });
      runs.calls[0]!.emit({
        type: "done",
        data: { thread_id: "t", status: "interrupted", intent: "reminder", errors: [] },
      });
      runs.calls[0]!.finish();
    });
    await act(async () => fireEvent.click(screen.getByRole("button", { name: "Approve" })));
    expect(screen.getByRole("alert").textContent).toBe("already decided (request r1)");
    expect(screen.getByRole("button", { name: "Approve" })).toBeTruthy();
  });
});

describe("ToolTimeline", () => {
  it("lists each call with its status, agent and arguments", () => {
    render(
      <ToolTimeline
        steps={[
          { id: "1", name: "get_invoice", args: { invoice: "INV-1" }, node: "reconciler", status: "success", content: "{}" },
          { id: "2", name: "get_po", args: { po: "PO-9" }, node: "reconciler", status: "error", content: "not found" },
          { id: "3", name: "get_contract", args: {}, node: "reconciler", status: "running" },
        ]}
      />,
    );
    expect(screen.getByText("3 tool calls · 1 running")).toBeTruthy();
    expect(screen.getByText("· 1 failed")).toBeTruthy();
    const list = screen.getByRole("list", { name: "Tool calls" });
    expect(within(list).getAllByRole("listitem")).toHaveLength(3);
    expect(within(list).getByLabelText("failed")).toBeTruthy();
    expect(within(list).getByText("not found")).toBeTruthy();
  });

  it("summarizes long arguments on one line", () => {
    expect(summarizeArgs({ invoice: "INV-1", limit: 5 })).toBe("invoice=INV-1, limit=5");
    expect(summarizeArgs({ body: "a\nb ".repeat(50) }, 20)).toHaveLength(20);
  });
});

describe("CitedText", () => {
  it("links known tags and leaves unknown ones as text", () => {
    render(<CitedText text="See [doc:INV-1] and [doc:X]." citations={{ "doc:INV-1": "d-1" }} />);
    const link = screen.getByRole("link", { name: "INV-1" });
    expect([link.getAttribute("href"), link.getAttribute("target")]).toEqual(["/documents/d-1", "_blank"]);
    expect(screen.getByText("[X]")).toBeTruthy();
  });
});
