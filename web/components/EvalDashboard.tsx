import Link from "next/link";

import { TrendChart } from "@/components/TrendChart";
import {
  change,
  type EvalRun,
  formatMetric,
  HEADLINE,
  metricLabel,
  numericMetrics,
} from "@/lib/eval";
import { formatDateTime, formatShortDate } from "@/lib/format";

function runLabel(run: EvalRun): string {
  return `${formatShortDate(run.created_at.slice(0, 10))}${run.git_sha ? ` · ${run.git_sha.slice(0, 7)}` : ""}`;
}

function evalHref(suite: string, metric: string): string {
  return `/eval?${new URLSearchParams({ suite, metric })}`;
}

/**
 * One suite's latest run (headline metrics with their change since the run before), the
 * trend of one metric across its runs, and every metric of the latest run.
 */
export function EvalDashboard({
  suites,
  suite,
  runs,
  metric,
}: {
  suites: string[];
  suite: string;
  /** This suite's runs, newest first. */
  runs: EvalRun[];
  metric: string;
}) {
  const [latest, previous] = runs;
  if (!latest) return null;
  const now = numericMetrics(latest);
  const before = previous ? numericMetrics(previous) : {};
  const headline = HEADLINE.filter((key) => key in now);
  const plotted = runs
    .slice()
    .reverse()
    .flatMap((run) => {
      const value = numericMetrics(run)[metric];
      return value === undefined
        ? []
        : [{ key: run.id, label: runLabel(run), value, text: formatMetric(metric, value) }];
    });
  const choices = Object.keys(now).filter((key) => runs.filter((r) => key in numericMetrics(r)).length > 0);

  return (
    <div className="mt-6 flex flex-col gap-6">
      <nav aria-label="Suite" className="flex flex-wrap gap-1">
        {suites.map((name) => (
          <Link
            key={name}
            href={evalHref(name, metric)}
            aria-current={name === suite ? "page" : undefined}
            className={`rounded-md px-3 py-1 text-sm ${
              name === suite ? "bg-accent-soft font-medium text-accent" : "text-muted hover:text-foreground"
            }`}
          >
            {name}
          </Link>
        ))}
      </nav>

      <section aria-labelledby="latest">
        <h2 id="latest" className="text-sm font-medium">
          Latest run
          <span className="ml-2 font-normal text-muted">
            {formatDateTime(latest.created_at)}
            {latest.git_sha ? ` · ${latest.git_sha.slice(0, 7)}` : ""}
            {typeof latest.metrics.questions === "number" ? ` · ${latest.metrics.questions} questions` : ""}
            {previous ? ` · compared with ${runLabel(previous)}` : ""}
          </span>
        </h2>
        <div className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {headline.map((key) => {
            const moved = change(key, now[key]!, before[key]);
            return (
              <Link
                key={key}
                href={evalHref(suite, key)}
                className={`rounded-lg border bg-surface p-4 hover:border-accent ${
                  key === metric ? "border-accent" : "border-border"
                }`}
              >
                <p className="text-xs font-medium uppercase tracking-wide text-muted">{metricLabel(key)}</p>
                <p className="mt-2 text-2xl font-semibold tabular-nums">{formatMetric(key, now[key]!)}</p>
                {moved ? (
                  <p className="mt-1 text-xs text-muted">
                    {moved.better === null ? (
                      "unchanged"
                    ) : (
                      <>
                        <span className={moved.better ? "text-success" : "text-danger"}>
                          {moved.delta > 0 ? "▲" : "▼"} {formatMetric(key, Math.abs(moved.delta))}
                        </span>{" "}
                        {moved.better ? "better" : "worse"}
                      </>
                    )}
                  </p>
                ) : null}
              </Link>
            );
          })}
        </div>
      </section>

      <section aria-labelledby="trend" className="rounded-lg border border-border bg-surface p-4">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <h2 id="trend" className="text-sm font-medium">
            {metricLabel(metric)}
            <span className="ml-2 font-normal text-muted">
              {plotted.length} run{plotted.length === 1 ? "" : "s"} of {suite}
            </span>
          </h2>
          <form className="text-xs" action="/eval">
            <input type="hidden" name="suite" value={suite} />
            <label className="text-muted">
              Metric{" "}
              <select
                name="metric"
                defaultValue={metric}
                className="rounded border border-border bg-background px-1.5 py-0.5 text-foreground"
              >
                {choices.map((key) => (
                  <option key={key} value={key}>
                    {metricLabel(key)}
                  </option>
                ))}
              </select>
            </label>{" "}
            <button type="submit" className="rounded border border-border px-2 py-0.5 hover:border-accent">
              Show
            </button>
          </form>
        </div>
        <div className="mt-4">
          <TrendChart points={plotted} title={metricLabel(metric)} />
        </div>
      </section>

      <section aria-labelledby="all-metrics">
        <h2 id="all-metrics" className="text-sm font-medium">
          Every metric of the latest run
        </h2>
        <div className="mt-2 overflow-x-auto rounded-lg border border-border bg-surface">
          <table className="w-full text-left text-sm">
            <thead className="border-b border-border text-xs text-muted">
              <tr>
                <th className="px-4 py-2 font-medium">Metric</th>
                <th className="px-4 py-2 text-right font-medium">Latest</th>
                <th className="px-4 py-2 text-right font-medium">Run before</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(latest.metrics).map(([key, value]) => (
                <tr key={key} className="border-t border-border first:border-t-0">
                  <td className="px-4 py-1.5">
                    {key in now ? (
                      <Link href={evalHref(suite, key)} className="hover:text-accent hover:underline">
                        {metricLabel(key)}
                      </Link>
                    ) : (
                      metricLabel(key)
                    )}
                  </td>
                  <td className="px-4 py-1.5 text-right tabular-nums">
                    {typeof value === "number" ? formatMetric(key, value) : <span className="text-muted">not measured</span>}
                  </td>
                  <td className="px-4 py-1.5 text-right tabular-nums text-muted">
                    {before[key] !== undefined ? formatMetric(key, before[key]) : "–"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
