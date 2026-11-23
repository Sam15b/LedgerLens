"use client";

import { useState } from "react";

import { api, apiErrorMessage, type Schemas, UNREACHABLE } from "@/lib/api";
import type { Interrupt, ReminderDraft } from "@/lib/chat";
import type { Decision } from "@/lib/chat-state";
import { humanize } from "@/lib/format";

export type DecisionOut = Schemas["DecisionOut"];

/** `POST /reminders/{id}/approve|reject`: the API's answer, or why it failed. */
export async function decide(
  reminderId: string,
  decision: Decision,
): Promise<{ ok: true; out: DecisionOut } | { ok: false; message: string }> {
  const path = decision === "approved" ? "/reminders/{reminder_id}/approve" : "/reminders/{reminder_id}/reject";
  try {
    const { data, error, response } = await api.POST(path, {
      params: { path: { reminder_id: reminderId } },
    });
    if (data) return { ok: true, out: data };
    return { ok: false, message: apiErrorMessage(error) ?? `The API answered ${response.status}.` };
  } catch {
    return { ok: false, message: UNREACHABLE };
  }
}

/**
 * The run paused for a person: each reminder draft with Approve and Reject. A decision is
 * recorded on the API, which resumes the run; ``onDecided`` gets its answer. Approving
 * sends nothing: LedgerLens only drafts.
 */
export function InterruptCard({
  interrupt,
  decisions,
  onDecided,
}: {
  interrupt: Interrupt;
  decisions: Record<string, Decision>;
  onDecided: (reminderId: string, decision: Decision, out: DecisionOut) => void;
}) {
  const undecided = interrupt.reminders.filter((r) => !(r.reminder_id in decisions)).length;
  return (
    <section
      aria-label="Reminder drafts waiting for approval"
      className="mt-3 rounded-lg border border-accent/50 bg-accent-soft/40 p-3"
    >
      <h3 className="text-sm font-medium">
        {undecided > 0
          ? `${undecided} reminder draft${undecided === 1 ? "" : "s"} waiting for your decision`
          : "All drafts decided"}
      </h3>
      <p className="mt-0.5 text-xs text-muted">
        Approving records your decision; nothing is sent. Send the approved text yourself.
      </p>
      <ul className="mt-3 flex flex-col gap-3">
        {interrupt.reminders.map((draft) => (
          <DraftItem
            key={draft.reminder_id}
            draft={draft}
            decision={decisions[draft.reminder_id] ?? null}
            onDecided={onDecided}
          />
        ))}
      </ul>
    </section>
  );
}

function DraftItem({
  draft,
  decision,
  onDecided,
}: {
  draft: ReminderDraft;
  decision: Decision | null;
  onDecided: (reminderId: string, decision: Decision, out: DecisionOut) => void;
}) {
  const [pending, setPending] = useState<Decision | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function choose(choice: Decision) {
    setPending(choice);
    setError(null);
    const result = await decide(draft.reminder_id, choice);
    setPending(null);
    if (result.ok) onDecided(draft.reminder_id, choice, result.out);
    else setError(result.message);
  }

  return (
    <li className="rounded-md border border-border bg-surface p-3">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <p className="text-sm font-medium">
          {draft.invoice_number}
          <span className="ml-2 rounded bg-border/70 px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-muted">
            {humanize(draft.tone)}
          </span>
        </p>
        {draft.vendor_email ? <p className="text-xs text-muted">To {draft.vendor_email}</p> : null}
      </div>
      <pre className="mt-2 max-h-56 overflow-auto rounded bg-background p-2 font-sans text-xs whitespace-pre-wrap">
        {draft.body}
      </pre>
      {decision ? (
        <p className={`mt-2 text-xs font-medium ${decision === "approved" ? "text-success" : "text-muted"}`}>
          {decision === "approved" ? "Approved: not sent, send it yourself." : "Rejected."}
        </p>
      ) : (
        <div className="mt-2 flex items-center gap-2">
          <button
            type="button"
            disabled={pending !== null}
            onClick={() => void choose("approved")}
            className="rounded-md bg-accent px-3 py-1 text-xs font-medium text-surface disabled:opacity-50"
          >
            {pending === "approved" ? "Approving…" : "Approve"}
          </button>
          <button
            type="button"
            disabled={pending !== null}
            onClick={() => void choose("rejected")}
            className="rounded-md border border-border px-3 py-1 text-xs hover:border-danger hover:text-danger disabled:opacity-50"
          >
            {pending === "rejected" ? "Rejecting…" : "Reject"}
          </button>
          {error ? (
            <p role="alert" className="text-xs text-danger">
              {error}
            </p>
          ) : null}
        </div>
      )}
    </li>
  );
}
