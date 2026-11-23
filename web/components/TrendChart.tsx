export type TrendPoint = { key: string; label: string; value: number; text: string };

/**
 * One metric across runs, oldest to newest: a 2px line with a marker per run. One series,
 * so no legend (the heading names it). Each marker shows its run and value on hover or
 * keyboard focus; the same numbers are one click away as a table.
 */
export function TrendChart({ points, title }: { points: TrendPoint[]; title: string }) {
  if (points.length === 0) return <p className="text-sm text-muted">No run measured this metric.</p>;
  const values = points.map((p) => p.value);
  let min = Math.min(...values);
  let max = Math.max(...values);
  if (max - min < 1e-9) {
    min -= Math.abs(min) * 0.05 || 1;
    max += Math.abs(max) * 0.05 || 1;
  }
  const pad = (max - min) * 0.1;
  const low = min - pad;
  const high = max + pad;
  const x = (i: number) => (points.length === 1 ? 50 : (i / (points.length - 1)) * 100);
  const y = (v: number) => 100 - ((v - low) / (high - low)) * 100;
  const line = points.map((p, i) => `${x(i)},${y(p.value)}`).join(" ");
  const top = points.reduce((a, b) => (b.value > a.value ? b : a));
  const bottom = points.reduce((a, b) => (b.value < a.value ? b : a));

  return (
    <figure>
      <div className="relative ml-14 h-48 border-b border-l border-border">
        <span className="absolute -left-14 w-12 text-right text-[11px] text-muted" style={{ top: `${y(top.value)}%`, transform: "translateY(-50%)" }}>
          {top.text}
        </span>
        {bottom.value !== top.value ? (
          <span className="absolute -left-14 w-12 text-right text-[11px] text-muted" style={{ top: `${y(bottom.value)}%`, transform: "translateY(-50%)" }}>
            {bottom.text}
          </span>
        ) : null}
        <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="absolute inset-0 h-full w-full overflow-visible" aria-hidden>
          <line x1="0" x2="100" y1={y(top.value)} y2={y(top.value)} className="stroke-border" strokeDasharray="2 2" vectorEffect="non-scaling-stroke" />
          <polyline points={line} fill="none" className="stroke-chart" strokeWidth={2} strokeLinejoin="round" vectorEffect="non-scaling-stroke" />
        </svg>
        <ol aria-label={title} className="absolute inset-0">
          {points.map((p, i) => (
            <li
              key={p.key}
              tabIndex={0}
              aria-label={`${p.label}: ${p.text}`}
              className="group absolute size-6 -translate-x-1/2 -translate-y-1/2 outline-none"
              style={{ left: `${x(i)}%`, top: `${y(p.value)}%` }}
            >
              <span className="absolute inset-1.5 rounded-full border-2 border-surface bg-chart group-hover:scale-125 group-focus-visible:ring-2 group-focus-visible:ring-accent" />
              <span
                role="tooltip"
                className={`pointer-events-none absolute bottom-full z-10 mb-1 hidden whitespace-nowrap rounded-md border border-border bg-surface px-2 py-1 text-xs shadow-sm group-hover:block group-focus-visible:block ${
                  i > points.length / 2 ? "right-0" : "left-0"
                }`}
              >
                <span className="block font-medium">{p.text}</span>
                <span className="block text-muted">{p.label}</span>
              </span>
            </li>
          ))}
        </ol>
      </div>
      <div className="ml-14 mt-1 flex justify-between text-[11px] text-muted">
        <span>{points[0]!.label}</span>
        {points.length > 1 ? <span>{points[points.length - 1]!.label}</span> : null}
      </div>
      <details className="mt-3 text-sm">
        <summary className="cursor-pointer text-xs text-muted hover:text-foreground">Show as a table</summary>
        <table className="mt-2 w-full text-left text-xs">
          <thead className="text-muted">
            <tr>
              <th className="py-1 font-medium">Run</th>
              <th className="py-1 text-right font-medium">{title}</th>
            </tr>
          </thead>
          <tbody>
            {points.map((p) => (
              <tr key={p.key} className="border-t border-border">
                <td className="py-1">{p.label}</td>
                <td className="py-1 text-right tabular-nums">{p.text}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </figure>
  );
}
