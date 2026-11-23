import type { Schemas } from "@/lib/api";
import { formatCount, formatShortDate, formatUsd } from "@/lib/format";

type DailyCost = Schemas["DailyCost"];

/**
 * LLM spend per day as thin bars (one series, so no legend: the heading names it). Each
 * bar shows its day, cost and call count on hover or keyboard focus, and the same numbers
 * are one click away as a table.
 */
export function CostChart({ days }: { days: DailyCost[] }) {
  const max = Math.max(0, ...days.map((d) => d.cost_usd));
  const first = days[0];
  const last = days[days.length - 1];
  return (
    <figure>
      <div className="relative h-36 border-b border-border">
        {max > 0 ? (
          <span className="absolute -top-1 left-0 text-[11px] text-muted">{formatUsd(max)}</span>
        ) : null}
        <div
          aria-hidden
          className="absolute inset-x-0 top-1.5 border-t border-dashed border-border"
        />
        <ol className="absolute inset-x-0 top-1.5 bottom-0 flex items-end gap-0.5 pl-0">
          {days.map((day) => {
            const height = max > 0 ? Math.max((day.cost_usd / max) * 100, day.calls ? 2 : 0) : 0;
            const label = `${formatShortDate(day.day)}: ${formatUsd(day.cost_usd)}, ${formatCount(day.calls)} calls`;
            return (
              <li
                key={day.day}
                tabIndex={0}
                aria-label={label}
                className="group relative flex h-full min-w-0 flex-1 items-end justify-center outline-none"
              >
                <span
                  className="w-full max-w-6 rounded-t bg-chart group-hover:opacity-80 group-focus-visible:ring-2 group-focus-visible:ring-accent"
                  style={{ height: `${height}%` }}
                />
                <span
                  role="tooltip"
                  className="pointer-events-none absolute bottom-full z-10 mb-1 hidden whitespace-nowrap rounded-md border border-border bg-surface px-2 py-1 text-xs shadow-sm group-hover:block group-focus-visible:block"
                >
                  <span className="block font-medium">{formatShortDate(day.day)}</span>
                  <span className="block text-muted">
                    {formatUsd(day.cost_usd)} · {formatCount(day.calls)} calls
                  </span>
                </span>
              </li>
            );
          })}
        </ol>
      </div>
      {first && last ? (
        <div className="mt-1 flex justify-between text-[11px] text-muted">
          <span>{formatShortDate(first.day)}</span>
          <span>{formatShortDate(last.day)}</span>
        </div>
      ) : null}
      <details className="mt-3 text-sm">
        <summary className="cursor-pointer text-xs text-muted hover:text-foreground">
          Show as a table
        </summary>
        <table className="mt-2 w-full text-left text-xs">
          <thead className="text-muted">
            <tr>
              <th className="py-1 font-medium">Day (UTC)</th>
              <th className="py-1 text-right font-medium">Calls</th>
              <th className="py-1 text-right font-medium">Cost</th>
            </tr>
          </thead>
          <tbody>
            {days.map((day) => (
              <tr key={day.day} className="border-t border-border">
                <td className="py-1">{formatShortDate(day.day)}</td>
                <td className="py-1 text-right tabular-nums">{formatCount(day.calls)}</td>
                <td className="py-1 text-right tabular-nums">{formatUsd(day.cost_usd)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </figure>
  );
}
