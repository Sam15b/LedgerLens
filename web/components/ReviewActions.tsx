"use client";

import { useRouter } from "next/navigation";
import { useState, useTransition } from "react";

import { api, apiErrorMessage, UNREACHABLE } from "@/lib/api";

type Verdict = "confirmed" | "false_positive";

/**
 * A person's verdict on one flag (`POST /anomalies/{id}/review`), with an optional note.
 * The verdict is kept on the score as a label for retraining; a later one replaces it.
 */
export function ReviewActions({ scoreId, current }: { scoreId: number; current?: Verdict | null }) {
  const router = useRouter();
  const [note, setNote] = useState("");
  const [sending, setSending] = useState<Verdict | null>(null);
  const [done, setDone] = useState<Verdict | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, startTransition] = useTransition();

  async function review(verdict: Verdict) {
    setSending(verdict);
    setError(null);
    try {
      const { data, error: body, response } = await api.POST("/anomalies/{score_id}/review", {
        params: { path: { score_id: scoreId } },
        body: { verdict, note: note.trim() || null },
      });
      if (data) {
        setDone(verdict);
        startTransition(() => router.refresh()); // the flag leaves the "to review" list
      } else {
        setError(apiErrorMessage(body) ?? `The API answered ${response.status}.`);
      }
    } catch {
      setError(UNREACHABLE);
    } finally {
      setSending(null);
    }
  }

  if (done) {
    return (
      <p className="text-xs text-muted" aria-live="polite">
        Marked {done === "confirmed" ? "confirmed" : "a false positive"}
        {refreshing ? "…" : "."}
      </p>
    );
  }
  const busy = sending !== null;
  return (
    <div className="flex flex-wrap items-center gap-2">
      <input
        value={note}
        onChange={(event) => setNote(event.target.value)}
        maxLength={2000}
        placeholder="Note (optional)"
        aria-label="Review note"
        className="min-w-48 flex-1 rounded-md border border-border bg-background px-2 py-1 text-xs outline-none focus:border-accent"
      />
      <button
        type="button"
        disabled={busy}
        onClick={() => void review("confirmed")}
        className={`rounded-md px-3 py-1 text-xs font-medium disabled:opacity-50 ${
          current === "confirmed" ? "border border-danger text-danger" : "bg-danger text-surface"
        }`}
      >
        {sending === "confirmed" ? "Saving…" : "Confirm anomaly"}
      </button>
      <button
        type="button"
        disabled={busy}
        onClick={() => void review("false_positive")}
        className="rounded-md border border-border px-3 py-1 text-xs hover:border-accent disabled:opacity-50"
      >
        {sending === "false_positive" ? "Saving…" : "False positive"}
      </button>
      {error ? (
        <p role="alert" className="w-full text-xs text-danger">
          {error}
        </p>
      ) : null}
    </div>
  );
}
