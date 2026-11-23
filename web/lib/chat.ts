/**
 * `POST /chat`: send a message and receive the agents' run as typed events.
 *
 * The API streams server-sent events (see src/ledgerlens/api/routers/chat.py): what the
 * agents call and get back, the answer text in pieces, the documents it may cite, a pause
 * for reminder approval, and a final `done`. They are not in the OpenAPI schema (a stream
 * has no response model), so their types are written out here.
 */
import { apiBaseUrl, apiErrorMessage, UNREACHABLE } from "@/lib/api";
import { SseParser, type SseEvent } from "@/lib/sse";

export type ToolCall = { id: string; name: string; args: Record<string, unknown>; node: string };
export type ToolResult = {
  id: string;
  name: string;
  status: "success" | "error";
  content: string;
  truncated: boolean;
  node: string;
};
export type Token = { text: string; node: string };
export type Citation = { tag: string; document_id: string };
export type ReminderDraft = {
  reminder_id: string;
  invoice_number: string;
  tone: string;
  vendor_email?: string | null;
  body: string;
};
export type Interrupt = { thread_id: string; type: "reminder_approval"; reminders: ReminderDraft[] };
export type Done = {
  thread_id: string;
  status: "completed" | "interrupted";
  intent: string | null;
  errors: unknown[];
};

export type ChatEvent =
  | { type: "tool_call"; data: ToolCall }
  | { type: "tool_result"; data: ToolResult }
  | { type: "token"; data: Token }
  | { type: "citation"; data: Citation }
  | { type: "interrupt"; data: Interrupt }
  | { type: "error"; data: { message: string } }
  | { type: "done"; data: Done };

const TYPES = new Set<string>([
  "tool_call",
  "tool_result",
  "token",
  "citation",
  "interrupt",
  "error",
  "done",
]);

/** One SSE event as a chat event, or `null` for an unknown type or a body that is not JSON. */
export function toChatEvent(event: SseEvent): ChatEvent | null {
  if (!TYPES.has(event.event)) return null;
  try {
    const data: unknown = JSON.parse(event.data);
    if (typeof data !== "object" || data === null) return null;
    return { type: event.event, data } as ChatEvent;
  } catch {
    return null;
  }
}

export type ChatRequest = { message: string; thread_id?: string | null; document_ids?: string[] };

/** The request was refused before the run started (422, 409 waiting thread, 503 agents). */
export class ChatError extends Error {
  constructor(
    message: string,
    readonly status: number | null,
  ) {
    super(message);
    this.name = "ChatError";
  }
}

/**
 * Runs one chat turn, calling ``onEvent`` for each event as it arrives. Resolves with the
 * thread id when the stream ends; throws {@link ChatError} when the API refuses the request
 * or cannot be reached. Aborting ``signal`` stops reading (an `AbortError` is thrown).
 */
export async function streamChat(
  request: ChatRequest,
  {
    onEvent,
    signal,
    baseUrl = apiBaseUrl(),
    fetcher = fetch,
  }: {
    onEvent: (event: ChatEvent) => void;
    signal?: AbortSignal;
    baseUrl?: string;
    fetcher?: typeof fetch;
  },
): Promise<{ threadId: string | null }> {
  let response: Response;
  try {
    response = await fetcher(`${baseUrl}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify(request),
      signal,
    });
  } catch (error) {
    if (signal?.aborted) throw error;
    throw new ChatError(UNREACHABLE, null);
  }
  if (!response.ok || response.body === null) {
    const body: unknown = await response.json().catch(() => null);
    throw new ChatError(
      apiErrorMessage(body) ?? `The API answered ${response.status}.`,
      response.status,
    );
  }

  const parser = new SseParser();
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      for (const raw of parser.feed(value)) {
        const event = toChatEvent(raw);
        if (event) onEvent(event);
      }
    }
  } finally {
    parser.end();
    reader.releaseLock();
  }
  return { threadId: response.headers.get("X-Thread-Id") };
}
