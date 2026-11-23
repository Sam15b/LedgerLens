import Link from "next/link";
import type { ReactNode } from "react";

import type { Schemas } from "@/lib/api";
import { formatDate, formatDateTime, formatMoney, humanize } from "@/lib/format";

export type Anomaly = Schemas["AnomalyOut"];

export type Reason = {
  code: string;
  text: string;
  /** The autoencoder's share of the reconstruction error (`null` for a rule). */
  share: number | null;
  source: "rule" | "autoencoder" | string;
  related_ids: string[];
};

/** The scorer's reasons, rules first (they are why most flags exist), then by share. */
export function readReasons(raw: Anomaly["reasons"]): Reason[] {
  const reasons = raw.map((r) => ({
    code: String(r.code ?? ""),
    text: String(r.text ?? r.code ?? ""),
    share: typeof r.share === "number" ? r.share : null,
    source: String(r.source ?? "rule"),
    related_ids: Array.isArray(r.related_ids) ? r.related_ids.map(String) : [],
  }));
  return reasons.sort(
    (a, b) =>
      Number(b.source === "rule") - Number(a.source === "rule") || (b.share ?? 0) - (a.share ?? 0),
  );
}

/** A 0..1 score as a number and a bar (the bar is decoration: the number is the value). */
export function ScoreMeter({ score }: { score: number }) {
  const percent = Math.round(score * 100);
  return (
    <div className="flex items-center gap-2" title={`Anomaly score ${score.toFixed(3)}`}>
      <span className="w-10 text-right text-lg font-semibold tabular-nums">{score.toFixed(2)}</span>
      <div aria-hidden className="h-1.5 w-20 overflow-hidden rounded-full bg-border">
        <div className="h-full rounded-full bg-danger" style={{ width: `${percent}%` }} />
      </div>
    </div>
  );
}

const VERDICTS: Record<string, string> = {
  confirmed: "Confirmed",
  false_positive: "False positive",
};

/** One flagged invoice: what it is, how anomalous, and why. ``actions`` goes at the end. */
export function AnomalyCard({ anomaly, actions }: { anomaly: Anomaly; actions?: ReactNode }) {
  const reasons = readReasons(anomaly.reasons);
  return (
    <article className="rounded-lg border border-border bg-surface p-4">
      <header className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <h2 className="font-medium">
            {anomaly.document_id ? (
              <Link href={`/documents/${anomaly.document_id}`} className="text-accent hover:underline">
                {anomaly.invoice_number}
              </Link>
            ) : (
              anomaly.invoice_number
            )}
            <span className="ml-2 font-normal text-muted">{anomaly.vendor_name}</span>
          </h2>
          <p className="mt-0.5 text-xs text-muted">
            {formatMoney(anomaly.total, anomaly.currency)} · invoiced {formatDate(anomaly.invoice_date)} ·
            scored {formatDateTime(anomaly.scored_at)} by {anomaly.model_version}
          </p>
        </div>
        <ScoreMeter score={anomaly.score} />
      </header>
      <ul aria-label="Reasons" className="mt-3 flex flex-col gap-1.5 text-sm">
        {reasons.map((reason, i) => (
          <li key={`${reason.code}-${i}`} className="flex items-baseline gap-2">
            <span
              className={`shrink-0 rounded px-1.5 py-0.5 text-[10px] uppercase tracking-wide ${
                reason.source === "rule" ? "bg-danger/10 text-danger" : "bg-border/70 text-muted"
              }`}
            >
              {reason.source === "rule" ? "Rule" : humanize(reason.source)}
            </span>
            <span className="min-w-0">
              {reason.text}
              {reason.share !== null ? (
                <span className="ml-1 text-xs text-muted">({Math.round(reason.share * 100)}% of the error)</span>
              ) : null}
              {reason.related_ids.length > 0 ? (
                <span className="ml-1 font-mono text-xs text-muted">{reason.related_ids.join(", ")}</span>
              ) : null}
            </span>
          </li>
        ))}
      </ul>
      {anomaly.review ? (
        <p className="mt-3 text-xs">
          <span className={anomaly.review === "confirmed" ? "font-medium text-danger" : "font-medium text-muted"}>
            {VERDICTS[anomaly.review]}
          </span>
          {anomaly.reviewed_at ? <span className="text-muted"> · {formatDateTime(anomaly.reviewed_at)}</span> : null}
          {anomaly.review_note ? <span className="text-muted"> · {anomaly.review_note}</span> : null}
        </p>
      ) : null}
      {actions ? <div className="mt-3 border-t border-border pt-3">{actions}</div> : null}
    </article>
  );
}
