"use client";

import { useRouter } from "next/navigation";
import { useState, useTransition } from "react";

import { api, apiErrorMessage, type Schemas, UNREACHABLE } from "@/lib/api";
import { formatDate, formatDateTime, formatMoney, humanize } from "@/lib/format";

export type Reminder = Schemas["ReminderOut"];
type Decision = "approved" | "rejected";

const STATUS: Record<string, { label: string; className: string }> = {
  pending_approval: { label: "Waiting for approval", className: "text-accent" },
  approved: { label: "Approved · not sent", className: "text-success" },
  rejected: { label: "Rejected", className: "text-muted" },
};

/**
 * One drafted reminder. While it waits for approval its text can be edited and saved
 * (`PATCH /reminders/{id}`), then approved or rejected; a decided draft is read-only.
 * Approving sends nothing: the person sends the approved text themselves.
 */
export function ReminderCard({ reminder }: { reminder: Reminder }) {
  const router = useRouter();
  const [saved, setSaved] = useState(reminder.draft);
  const [text, setText] = useState(reminder.draft);
  const [busy, setBusy] = useState<"save" | Decision | null>(null);
  const [decided, setDecided] = useState<Decision | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [, startTransition] = useTransition();

  const pending = reminder.status === "pending_approval" && decided === null;
  const dirty = text !== saved;

  async function run(action: "save" | Decision) {
    setBusy(action);
    setError(null);
    try {
      if (action === "save") {
        const { data, error: body, response } = await api.PATCH("/reminders/{reminder_id}", {
          params: { path: { reminder_id: reminder.id } },
          body: { draft: text },
        });
        if (data) setSaved(data.draft);
        else setError(apiErrorMessage(body) ?? `The API answered ${response.status}.`);
        return;
      }
      const path =
        action === "approved" ? "/reminders/{reminder_id}/approve" : "/reminders/{reminder_id}/reject";
      const { data, error: body, response } = await api.POST(path, {
        params: { path: { reminder_id: reminder.id } },
      });
      if (data) {
        setDecided(action);
        startTransition(() => router.refresh());
      } else {
        setError(apiErrorMessage(body) ?? `The API answered ${response.status}.`);
      }
    } catch {
      setError(UNREACHABLE);
    } finally {
      setBusy(null);
    }
  }

  const status = STATUS[decided ?? reminder.status] ?? { label: humanize(reminder.status), className: "text-muted" };
  return (
    <article className="rounded-lg border border-border bg-surface p-4">
      <header className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <h2 className="font-medium">
            {reminder.invoice_number ?? reminder.invoice_id}
            {reminder.vendor_name ? <span className="ml-2 font-normal text-muted">{reminder.vendor_name}</span> : null}
            <span className="ml-2 rounded bg-border/70 px-1.5 py-0.5 align-middle text-[10px] uppercase tracking-wide text-muted">
              {humanize(reminder.tone)}
            </span>
          </h2>
          <p className="mt-0.5 text-xs text-muted">
            {[
              reminder.total != null && reminder.currency ? formatMoney(reminder.total, reminder.currency) : null,
              reminder.due_date ? `due ${formatDate(reminder.due_date)}` : null,
              `drafted ${formatDateTime(reminder.created_at)}`,
              reminder.decided_at ? `decided ${formatDateTime(reminder.decided_at)}` : null,
            ]
              .filter(Boolean)
              .join(" · ")}
          </p>
        </div>
        <p className={`text-xs font-medium ${status.className}`}>{status.label}</p>
      </header>

      {pending ? (
        <textarea
          value={text}
          onChange={(event) => setText(event.target.value)}
          rows={Math.min(14, Math.max(5, text.split("\n").length + 1))}
          maxLength={10_000}
          aria-label={`Draft for ${reminder.invoice_number ?? reminder.invoice_id}`}
          className="mt-3 w-full resize-y rounded-md border border-border bg-background p-2 font-sans text-sm outline-none focus:border-accent"
        />
      ) : (
        <pre className="mt-3 max-h-72 overflow-auto rounded-md bg-background p-2 font-sans text-sm whitespace-pre-wrap">
          {saved}
        </pre>
      )}

      {pending ? (
        <div className="mt-3 flex flex-wrap items-center gap-2">
          <button
            type="button"
            disabled={busy !== null || !dirty || !text.trim()}
            onClick={() => void run("save")}
            className="rounded-md border border-border px-3 py-1 text-xs hover:border-accent disabled:opacity-50"
          >
            {busy === "save" ? "Saving…" : dirty ? "Save changes" : "Saved"}
          </button>
          <button
            type="button"
            disabled={busy !== null || dirty}
            title={dirty ? "Save or undo your changes first" : undefined}
            onClick={() => void run("approved")}
            className="rounded-md bg-accent px-3 py-1 text-xs font-medium text-surface disabled:opacity-50"
          >
            {busy === "approved" ? "Approving…" : "Approve"}
          </button>
          <button
            type="button"
            disabled={busy !== null}
            onClick={() => void run("rejected")}
            className="rounded-md border border-border px-3 py-1 text-xs hover:border-danger hover:text-danger disabled:opacity-50"
          >
            {busy === "rejected" ? "Rejecting…" : "Reject"}
          </button>
          {dirty ? (
            <button
              type="button"
              disabled={busy !== null}
              onClick={() => setText(saved)}
              className="text-xs text-muted hover:text-foreground"
            >
              Undo changes
            </button>
          ) : null}
        </div>
      ) : null}
      {error ? (
        <p role="alert" className="mt-2 text-xs text-danger">
          {error}
        </p>
      ) : null}
    </article>
  );
}
