import { describe, expect, it, vi } from "vitest";

import { ChatError, type ChatEvent, streamChat, toChatEvent } from "@/lib/chat";
import {
  applyEvent,
  endTurn,
  newAssistantTurn,
  splitCitations,
  type Turn,
  waitingForDecision,
} from "@/lib/chat-state";
import { SseParser } from "@/lib/sse";
import { waitForDocument } from "@/lib/upload";

const STREAM = [
  'event: tool_call\ndata: {"id": "c1", "name": "find_invoices", "args": {"vendor": "Acme"}, "node": "qa"}\n\n',
  'event: tool_result\ndata: {"id": "c1", "name": "find_invoices", "status": "success", "content": "2 invoices", "truncated": false, "node": "qa"}\n\n',
  'event: token\ndata: {"text": "INV-1 is unpaid ", "node": "qa"}\n\n',
  'event: token\ndata: {"text": "[doc:INV-1].", "node": "qa"}\n\n',
  'event: citation\ndata: {"tag": "doc:INV-1", "document_id": "d-1"}\n\n',
  'event: done\ndata: {"thread_id": "chat-1", "status": "completed", "intent": "question", "errors": []}\n\n',
].join("");

describe("SseParser", () => {
  it("parses whole events", () => {
    const events = new SseParser().feed("event: token\ndata: {\"text\": \"hi\"}\n\nevent: done\ndata: {}\n\n");
    expect(events).toEqual([
      { event: "token", data: '{"text": "hi"}' },
      { event: "done", data: "{}" },
    ]);
  });

  it("gives the same events however the stream is split", () => {
    const whole = new SseParser().feed(STREAM);
    expect(whole).toHaveLength(6);
    for (const size of [1, 2, 3, 7, 64]) {
      const parser = new SseParser();
      const events = [];
      for (let i = 0; i < STREAM.length; i += size) events.push(...parser.feed(STREAM.slice(i, i + size)));
      expect(events).toEqual(whole);
    }
  });

  it("accepts CRLF and CR line ends, also split between chunks", () => {
    const parser = new SseParser();
    const events = [
      ...parser.feed("event: a\r"),
      ...parser.feed("\ndata: 1\r"),
      ...parser.feed("\n\r"),
      ...parser.feed("\nevent: b\rdata: 2\r\r"),
    ];
    expect(events).toEqual([
      { event: "a", data: "1" },
      { event: "b", data: "2" },
    ]);
  });

  it("joins data lines, skips comments, keeps ids, and defaults the type", () => {
    const events = new SseParser().feed(
      "﻿: keep-alive\nid: 7\ndata:first\ndata: second\nretry: 100\n\nevent: x\n\n",
    );
    expect(events).toEqual([{ event: "message", data: "first\nsecond", id: "7" }]);
  });

  it("drops an event the stream ends in the middle of", () => {
    const parser = new SseParser();
    expect(parser.feed("event: token\ndata: {}")).toEqual([]);
    expect(parser.end()).toEqual([]);
    expect(parser.feed("\n\n")).toEqual([]); // the half event is gone, not completed later
  });

  it("keeps only known chat events with JSON bodies", () => {
    expect(toChatEvent({ event: "token", data: '{"text": "a", "node": "qa"}' })).toEqual({
      type: "token",
      data: { text: "a", node: "qa" },
    });
    expect(toChatEvent({ event: "ping", data: "{}" })).toBeNull();
    expect(toChatEvent({ event: "token", data: "not json" })).toBeNull();
    expect(toChatEvent({ event: "token", data: "3" })).toBeNull();
  });
});

function streamOf(chunks: string[], init: ResponseInit = {}): Response {
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      const encoder = new TextEncoder();
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
      controller.close();
    },
  });
  return new Response(body, {
    status: 200,
    headers: { "Content-Type": "text/event-stream", "X-Thread-Id": "chat-1" },
    ...init,
  });
}

describe("streamChat", () => {
  it("posts the message and hands over each event as it arrives", async () => {
    const fetcher = vi.fn(async () => streamOf([STREAM.slice(0, 50), STREAM.slice(50)]));
    const events: ChatEvent[] = [];
    const out = await streamChat(
      { message: "unpaid?", thread_id: null, document_ids: [] },
      { onEvent: (e) => events.push(e), baseUrl: "http://api.test", fetcher },
    );
    expect(out).toEqual({ threadId: "chat-1" });
    expect(events.map((e) => e.type)).toEqual([
      "tool_call",
      "tool_result",
      "token",
      "token",
      "citation",
      "done",
    ]);
    const [url, init] = fetcher.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe("http://api.test/chat");
    expect(init.method).toBe("POST");
    expect(JSON.parse(String(init.body))).toEqual({ message: "unpaid?", thread_id: null, document_ids: [] });
  });

  it("turns a refusal into a ChatError with the API's message", async () => {
    const body = { error: { code: "conflict", message: "thread is waiting", request_id: "r9" } };
    const fetcher = async () => new Response(JSON.stringify(body), { status: 409 });
    const run = streamChat({ message: "x" }, { onEvent: () => {}, baseUrl: "", fetcher });
    await expect(run).rejects.toEqual(new ChatError("thread is waiting (request r9)", 409));
  });

  it("says when the API cannot be reached", async () => {
    const fetcher = async () => {
      throw new TypeError("Failed to fetch");
    };
    const run = streamChat({ message: "x" }, { onEvent: () => {}, baseUrl: "", fetcher });
    await expect(run).rejects.toMatchObject({ name: "ChatError", status: null });
  });
});

function play(events: string): ReturnType<typeof newAssistantTurn> {
  return new SseParser()
    .feed(events)
    .map(toChatEvent)
    .reduce((turn, event) => (event ? applyEvent(turn, event) : turn), newAssistantTurn(1));
}

describe("chat state", () => {
  it("builds the turn from the stream", () => {
    const turn = play(STREAM);
    expect(turn.text).toBe("INV-1 is unpaid [doc:INV-1].");
    expect(turn.tools).toEqual([
      {
        id: "c1",
        name: "find_invoices",
        args: { vendor: "Acme" },
        node: "qa",
        status: "success",
        content: "2 invoices",
        truncated: false,
      },
    ]);
    expect(turn.citations).toEqual({ "doc:INV-1": "d-1" });
    expect(turn.status).toBe("completed");
    expect(turn.intent).toBe("question");
  });

  it("starts each agent's text on a new paragraph", () => {
    const turn = play(
      'event: token\ndata: {"text": "Matched.", "node": "reconciler"}\n\n' +
        'event: token\ndata: {"text": "Drafted 1.", "node": "reminder_writer"}\n\n',
    );
    expect(turn.text).toBe("Matched.\n\nDrafted 1.");
  });

  it("marks calls that never got a result as failed when the run ends", () => {
    const turn = play(
      'event: tool_call\ndata: {"id": "c1", "name": "t", "args": {}, "node": "qa"}\n\n' +
        'event: done\ndata: {"thread_id": "x", "status": "completed", "intent": null, "errors": []}\n\n',
    );
    expect(turn.tools[0]?.status).toBe("error");
  });

  it("keeps a run error as the turn's status", () => {
    const turn = play(
      'event: error\ndata: {"message": "RuntimeError: boom"}\n\n' +
        'event: done\ndata: {"thread_id": "x", "status": "completed", "intent": null, "errors": []}\n\n',
    );
    expect([turn.status, turn.error]).toEqual(["failed", "RuntimeError: boom"]);
  });

  it("ends a turn the stream left unfinished, but not a finished one", () => {
    expect(endTurn(newAssistantTurn(1), "stopped").status).toBe("stopped");
    expect(endTurn(play(STREAM), "failed", "cut").status).toBe("completed");
  });

  it("waits for a decision while a paused run has undecided drafts", () => {
    const paused = play(
      'event: interrupt\ndata: {"thread_id": "t", "type": "reminder_approval", "reminders": [' +
        '{"reminder_id": "r1", "invoice_number": "INV-1", "tone": "polite", "body": "Hi"},' +
        '{"reminder_id": "r2", "invoice_number": "INV-2", "tone": "firm", "body": "Hi"}]}\n\n' +
        'event: done\ndata: {"thread_id": "t", "status": "interrupted", "intent": "reminder", "errors": []}\n\n',
    );
    const reply = newAssistantTurn(2, { text: "Rejected: INV-1.", status: "interrupted" });
    const turns: Turn[] = [paused, reply];
    expect(waitingForDecision(turns)).toBe(true);
    const oneLeft = { ...paused, decisions: { r1: "rejected" as const } };
    expect(waitingForDecision([oneLeft, reply])).toBe(true);
    const allDone = { ...paused, decisions: { r1: "rejected" as const, r2: "approved" as const } };
    expect(waitingForDecision([allDone, reply])).toBe(false);
    expect(waitingForDecision([])).toBe(false);
  });

  it("splits an answer around its citation tags", () => {
    expect(splitCitations("A [doc:INV-1] and [doc:PO-9].", { "doc:INV-1": "d-1" })).toEqual([
      { text: "A " },
      { tag: "doc:INV-1", label: "INV-1", documentId: "d-1" },
      { text: " and " },
      { tag: "doc:PO-9", label: "PO-9", documentId: null },
      { text: "." },
    ]);
    expect(splitCitations("no tags [graph]", {})).toEqual([{ text: "no tags [graph]" }]);
  });
});

describe("waitForDocument", () => {
  it("polls until the worker is done", async () => {
    const statuses = ["queued", "processing", "ready"];
    const read = vi.fn(async () => ({ status: statuses.shift()!, error: null }));
    await expect(waitForDocument("d", { everyMs: 1, read })).resolves.toEqual({ status: "ready", error: null });
    expect(read).toHaveBeenCalledTimes(3);
  });

  it("retries a failed read and stops when aborted", async () => {
    const controller = new AbortController();
    const read = vi.fn(async () => {
      controller.abort(new Error("left the page"));
      return null;
    });
    await expect(waitForDocument("d", { everyMs: 1, read, signal: controller.signal })).rejects.toThrow(
      "left the page",
    );
  });
});
