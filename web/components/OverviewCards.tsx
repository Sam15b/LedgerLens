import Link from "next/link";
import type { ReactNode } from "react";

import { CostChart } from "@/components/CostChart";
import { pageReady } from "@/components/nav";
import type { Overview } from "@/lib/api";
import { formatCount, formatMoney, formatUsd } from "@/lib/format";

type Tone = "default" | "danger";

function KpiCard({
  label,
  value,
  detail,
  href,
  tone = "default",
}: {
  label: string;
  value: ReactNode;
  detail?: ReactNode;
  href?: string;
  tone?: Tone;
}) {
  const body = (
    <>
      <p className="text-xs font-medium uppercase tracking-wide text-muted">{label}</p>
      <p
        className={`mt-2 text-2xl font-semibold tabular-nums tracking-tight ${
          tone === "danger" ? "text-danger" : ""
        }`}
      >
        {value}
      </p>
      {detail ? <div className="mt-1 text-xs text-muted">{detail}</div> : null}
    </>
  );
  const box = "block rounded-lg border border-border bg-surface p-4";
  if (href && pageReady(href)) {
    return (
      <Link href={href} className={`${box} hover:border-accent`}>
        {body}
      </Link>
    );
  }
  return <div className={box}>{body}</div>;
}

function plural(n: number, one: string, many = `${one}s`): string {
  return `${formatCount(n)} ${n === 1 ? one : many}`;
}

/** The overview's headline numbers (GET /stats/overview) and LLM spend per day. */
export function OverviewCards({ overview }: { overview: Overview }) {
  const { documents, unpaid } = overview;
  const inProgress = documents.queued + documents.processing;
  const [main, ...otherCurrencies] = unpaid;
  const overdue = unpaid.filter((u) => u.overdue_invoices > 0);
  const totalCost = overview.llm_cost_by_day.reduce((sum, d) => sum + d.cost_usd, 0);
  const totalCalls = overview.llm_cost_by_day.reduce((sum, d) => sum + d.calls, 0);

  return (
    <div className="mt-6 flex flex-col gap-4">
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
        <KpiCard
          label="Documents ingested"
          value={formatCount(documents.ready)}
          href="/documents"
          detail={
            <>
              of {plural(documents.total, "upload")}
              {inProgress > 0 ? ` · ${formatCount(inProgress)} in progress` : ""}
              {documents.failed > 0 ? (
                <span className="text-danger"> · {formatCount(documents.failed)} failed</span>
              ) : null}
            </>
          }
        />
        <KpiCard
          label="Unpaid"
          value={main ? formatMoney(main.outstanding, main.currency) : formatMoney(0, "USD")}
          detail={
            <>
              {plural(overview.open_invoices, "open invoice")}
              {otherCurrencies.map((u) => (
                <span key={u.currency} className="block">
                  + {formatMoney(u.outstanding, u.currency)}
                </span>
              ))}
            </>
          }
        />
        <KpiCard
          label="Overdue"
          value={formatCount(overview.overdue_invoices)}
          tone={overview.overdue_invoices > 0 ? "danger" : "default"}
          detail={
            overdue.length > 0
              ? overdue.map((u) => (
                  <span key={u.currency} className="block">
                    {formatMoney(u.overdue, u.currency)} past due
                  </span>
                ))
              : "Nothing past due"
          }
        />
        <KpiCard
          label="Flagged invoices"
          value={formatCount(overview.flagged_invoices)}
          href="/anomalies"
          tone={overview.flagged_invoices > 0 ? "danger" : "default"}
          detail="Anomaly flags nobody has reviewed"
        />
        <KpiCard
          label="Reminders to approve"
          value={formatCount(overview.pending_reminders)}
          href="/reminders"
          detail="Drafts waiting for a decision"
        />
      </div>

      <section className="rounded-lg border border-border bg-surface p-4">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <h2 className="text-sm font-medium">
            LLM cost per day
            <span className="ml-2 font-normal text-muted">
              last {overview.llm_cost_by_day.length} days, UTC
            </span>
          </h2>
          {overview.llm_costs_recorded ? (
            <p className="text-sm tabular-nums">
              {formatUsd(totalCost)} <span className="text-muted">· {plural(totalCalls, "call")}</span>
            </p>
          ) : null}
        </div>
        {overview.llm_costs_recorded ? (
          <div className="mt-4">
            <CostChart days={overview.llm_cost_by_day} />
          </div>
        ) : (
          <p className="mt-3 text-sm text-muted">
            No LLM calls are recorded. Start the API and worker with{" "}
            <code className="font-mono text-xs">LLM_COST_DB=true</code> to keep each call&apos;s
            tokens and cost.
          </p>
        )}
      </section>
      <p className="text-xs text-muted">
        As of {overview.as_of}. Amounts are summed per currency, never converted.
      </p>
    </div>
  );
}
