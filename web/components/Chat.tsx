"use client";

import { type FormEvent, type KeyboardEvent, useEffect, useRef, useState } from "react";

import { CitedText, Sources } from "@/components/Citation";
import { type DecisionOut, InterruptCard } from "@/components/InterruptCard";
import { ToolTimeline } from "@/components/ToolTimeline";
import { ChatError, streamChat } from "@/lib/chat";
import {
  type AssistantTurn,
  applyEvent,
  type Decision,
  endTurn,
  newAssistantTurn,
  type Turn,
  waitingForDecision,
} from "@/lib/chat-state";

/** The user stories the agents answer (docs/01-project-overview.md). */
export const EXAMPLES = [
  "Show unpaid invoices from Acme over $5k",
  "Does INV-889 match its PO and contract rate?",
  "Draft reminders for invoices 30+ days overdue",
];

/**
 * Ask the agents: the answer streams in as it is produced. The conversation is one thread
 * on the API (its checkpointer keeps it); "New conversation" starts another.
 */
export function Chat() {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [threadId, setThreadId] = useState<string | null>(null);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const abort = useRef<AbortController | null>(null);
  const nextKey = useRef(0);
  const bottom = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottom.current?.scrollIntoView?.({ block: "end" });
  }, [turns]);

  useEffect(() => () => abort.current?.abort(), []);

  function updateAssistant(key: number, change: (turn: AssistantTurn) => AssistantTurn) {
    setTurns((current) =>
      current.map((t) => (t.role === "assistant" && t.key === key ? change(t) : t)),
    );
  }

  async function send(message: string) {
    const text = message.trim();
    if (!text || streaming) return;
    const userKey = nextKey.current++;
    const key = nextKey.current++;
    setTurns((current) => [
      ...current,
      { role: "user", key: userKey, text, attachments: [] },
      newAssistantTurn(key),
    ]);
    setInput("");
    setStreaming(true);
    const controller = new AbortController();
    abort.current = controller;
    try {
      const { threadId: thread } = await streamChat(
        { message: text, thread_id: threadId },
        {
          signal: controller.signal,
          onEvent: (event) => {
            if (event.type === "done") setThreadId(event.data.thread_id);
            updateAssistant(key, (turn) => applyEvent(turn, event));
          },
        },
      );
      if (thread) setThreadId(thread);
      updateAssistant(key, (turn) => endTurn(turn, "failed", "The answer stopped before it was finished."));
    } catch (error) {
      if (controller.signal.aborted) {
        updateAssistant(key, (turn) => endTurn(turn, "stopped"));
      } else {
        const message = error instanceof ChatError ? error.message : "The answer stream broke off.";
        updateAssistant(key, (turn) => endTurn(turn, "failed", message));
      }
    } finally {
      abort.current = null;
      setStreaming(false);
    }
  }

  /** A draft was decided: mark it, and show what the resumed run answered. */
  function onDecided(key: number, reminderId: string, decision: Decision, out: DecisionOut) {
    updateAssistant(key, (turn) => ({
      ...turn,
      decisions: { ...turn.decisions, [reminderId]: decision },
    }));
    const reply = out.reply ?? out.note;
    if (reply) {
      setTurns((current) => [
        ...current,
        newAssistantTurn(nextKey.current++, {
          text: reply,
          status: out.thread_status === "interrupted" ? "interrupted" : "completed",
        }),
      ]);
    }
  }

  function onSubmit(event: FormEvent) {
    event.preventDefault();
    void send(input);
  }

  function onKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      void send(input);
    }
  }

  function reset() {
    abort.current?.abort();
    setTurns([]);
    setThreadId(null);
  }

  const waiting = waitingForDecision(turns);
  const blocked = streaming || waiting;

  return (
    <div className="flex min-h-[70vh] flex-col gap-4">
      <div className="flex items-center justify-between gap-3 text-xs text-muted">
        <span>{threadId ? <>Thread <span className="font-mono">{threadId}</span></> : "New conversation"}</span>
        {turns.length > 0 ? (
          <button
            type="button"
            onClick={reset}
            className="rounded-md border border-border px-2.5 py-1 hover:text-foreground"
          >
            New conversation
          </button>
        ) : null}
      </div>

      <ol aria-label="Conversation" className="flex flex-1 flex-col gap-4">
        {turns.length === 0 ? (
          <li className="rounded-lg border border-dashed border-border p-6 text-sm text-muted">
            <p>Ask about invoices, purchase orders, payments and contracts. For example:</p>
            <div className="mt-3 flex flex-wrap gap-2">
              {EXAMPLES.map((example) => (
                <button
                  key={example}
                  type="button"
                  onClick={() => void send(example)}
                  className="rounded-full border border-border bg-surface px-3 py-1 text-foreground hover:border-accent"
                >
                  {example}
                </button>
              ))}
            </div>
          </li>
        ) : null}
        {turns.map((turn) =>
          turn.role === "user" ? (
            <li key={turn.key} className="self-end max-w-[80%] rounded-lg bg-accent-soft px-4 py-2 text-sm">
              <p className="whitespace-pre-wrap">{turn.text}</p>
            </li>
          ) : (
            <li key={turn.key} className="max-w-full">
              <AssistantMessage
                turn={turn}
                onDecided={(reminderId, decision, out) => onDecided(turn.key, reminderId, decision, out)}
              />
            </li>
          ),
        )}
      </ol>
      <div ref={bottom} />

      <form onSubmit={onSubmit} className="sticky bottom-0 flex flex-col gap-2 bg-background pb-2">
        {waiting ? (
          <p className="text-xs text-muted">Approve or reject the reminder drafts above to continue.</p>
        ) : null}
        <div className="flex items-end gap-2 rounded-lg border border-border bg-surface p-2 focus-within:border-accent">
          <textarea
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={onKeyDown}
            rows={2}
            maxLength={4000}
            disabled={waiting}
            placeholder="Ask the agents…  (Enter to send, Shift+Enter for a new line)"
            aria-label="Message"
            className="min-h-10 flex-1 resize-none bg-transparent px-2 py-1 text-sm outline-none disabled:opacity-60"
          />
          {streaming ? (
            <button
              type="button"
              onClick={() => abort.current?.abort()}
              className="rounded-md border border-border px-3 py-1.5 text-sm hover:border-danger hover:text-danger"
            >
              Stop
            </button>
          ) : (
            <button
              type="submit"
              disabled={blocked || !input.trim()}
              className="rounded-md bg-accent px-3 py-1.5 text-sm font-medium text-surface disabled:opacity-50"
            >
              Send
            </button>
          )}
        </div>
      </form>
    </div>
  );
}

function AssistantMessage({
  turn,
  onDecided,
}: {
  turn: AssistantTurn;
  onDecided: (reminderId: string, decision: Decision, out: DecisionOut) => void;
}) {
  return (
    <article
      aria-busy={turn.status === "streaming"}
      className="rounded-lg border border-border bg-surface px-4 py-3 text-sm"
    >
      <ToolTimeline steps={turn.tools} />
      {turn.text ? (
        <CitedText text={turn.text} citations={turn.citations} />
      ) : turn.status === "streaming" ? (
        <p className="animate-pulse text-muted">Working…</p>
      ) : null}
      {turn.error ? (
        <p role="alert" className="mt-2 text-danger">
          {turn.error}
        </p>
      ) : null}
      {turn.status === "stopped" ? <p className="mt-2 text-xs text-muted">Stopped.</p> : null}
      <Sources citations={turn.citations} />
      {turn.interrupt ? (
        <InterruptCard interrupt={turn.interrupt} decisions={turn.decisions} onDecided={onDecided} />
      ) : null}
    </article>
  );
}
