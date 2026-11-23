/**
 * The conversation as the chat page shows it, built from the stream's events by a pure
 * reducer ({@link applyEvent}), so it can be tested without a network or a browser.
 */
import type { ChatEvent, Interrupt } from "@/lib/chat";

export type ToolStep = {
  id: string;
  name: string;
  args: Record<string, unknown>;
  node: string;
  status: "running" | "success" | "error";
  content?: string;
  truncated?: boolean;
};

export type Decision = "approved" | "rejected";

export type UserTurn = { role: "user"; key: number; text: string; attachments: string[] };

export type AssistantTurn = {
  role: "assistant";
  key: number;
  /** The answer so far; each agent's text starts a new paragraph. */
  text: string;
  textNode: string | null;
  tools: ToolStep[];
  /** `doc:INV-889` -> document id, for the tags the answer may cite. */
  citations: Record<string, string>;
  interrupt: Interrupt | null;
  decisions: Record<string, Decision>;
  error: string | null;
  status: "streaming" | "completed" | "interrupted" | "failed" | "stopped";
  intent: string | null;
};

export type Turn = UserTurn | AssistantTurn;

export function newAssistantTurn(key: number, fields: Partial<AssistantTurn> = {}): AssistantTurn {
  return {
    role: "assistant",
    key,
    text: "",
    textNode: null,
    tools: [],
    citations: {},
    interrupt: null,
    decisions: {},
    error: null,
    status: "streaming",
    intent: null,
    ...fields,
  };
}

/** ``turn`` after one stream event. */
export function applyEvent(turn: AssistantTurn, event: ChatEvent): AssistantTurn {
  switch (event.type) {
    case "tool_call":
      return {
        ...turn,
        tools: [...turn.tools, { ...event.data, status: "running" }],
      };
    case "tool_result": {
      const { id, status, content, truncated } = event.data;
      let matched = false;
      const tools = turn.tools.map((step) => {
        if (matched || step.id !== id) return step;
        matched = true;
        return { ...step, status, content, truncated };
      });
      if (!matched) {
        // a result whose call was not streamed: show it on its own
        tools.push({ ...event.data, args: {}, status });
      }
      return { ...turn, tools };
    }
    case "token": {
      const { text, node } = event.data;
      const joiner = turn.text && turn.textNode !== node ? "\n\n" : "";
      return { ...turn, text: turn.text + joiner + text, textNode: node };
    }
    case "citation":
      return { ...turn, citations: { ...turn.citations, [event.data.tag]: event.data.document_id } };
    case "interrupt":
      return { ...turn, interrupt: event.data };
    case "error":
      return { ...turn, error: event.data.message, status: "failed" };
    case "done":
      return {
        ...turn,
        intent: event.data.intent,
        status: turn.status === "failed" ? "failed" : event.data.status,
        tools: turn.tools.map((step) =>
          step.status === "running" ? { ...step, status: "error", content: "No result was returned." } : step,
        ),
      };
  }
}

/** The stream ended without `done` (stopped by the user, or the connection dropped). */
export function endTurn(turn: AssistantTurn, status: "stopped" | "failed", error?: string): AssistantTurn {
  if (turn.status !== "streaming") return turn;
  return { ...turn, status, error: error ?? turn.error };
}

/** Whether the thread waits for decisions on reminder drafts (new messages get 409). */
export function waitingForDecision(turns: Turn[]): boolean {
  // The latest pause: replies to decisions are separate turns, so look past them.
  const paused = turns.findLast(
    (t): t is AssistantTurn => t.role === "assistant" && t.interrupt !== null,
  );
  if (!paused?.interrupt) return false;
  return paused.interrupt.reminders.some((r) => !(r.reminder_id in paused.decisions));
}

export type Segment = { text: string } | { tag: string; label: string; documentId: string | null };

const TAG = /\[(doc:[^\]\s]+)\]/g;

/**
 * The answer split around its citation tags (`[doc:INV-889]`): each tag carries the
 * document it names when the stream said so (`citation` events), else `null`.
 */
export function splitCitations(text: string, citations: Record<string, string>): Segment[] {
  const out: Segment[] = [];
  let last = 0;
  for (const match of text.matchAll(TAG)) {
    const tag = match[1]!;
    if (match.index > last) out.push({ text: text.slice(last, match.index) });
    out.push({ tag, label: tag.slice("doc:".length), documentId: citations[tag] ?? null });
    last = match.index + match[0].length;
  }
  if (last < text.length) out.push({ text: text.slice(last) });
  return out;
}
