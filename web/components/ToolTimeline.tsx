import type { ToolStep } from "@/lib/chat-state";
import { humanize } from "@/lib/format";

const STATUS: Record<ToolStep["status"], { label: string; mark: string; className: string }> = {
  running: { label: "running", mark: "…", className: "text-accent animate-pulse" },
  success: { label: "done", mark: "✓", className: "text-success" },
  error: { label: "failed", mark: "✕", className: "text-danger" },
};

/** `{"invoice": "INV-1", "limit": 5}` -> `invoice=INV-1, limit=5` (long values cut). */
export function summarizeArgs(args: Record<string, unknown>, max = 80): string {
  const text = Object.entries(args)
    .map(([key, value]) => `${key}=${typeof value === "string" ? value : JSON.stringify(value)}`)
    .join(", ")
    .replace(/\s+/g, " ");
  return text.length > max ? `${text.slice(0, max - 1)}…` : text;
}

/**
 * The tools the agents called, in order, as they run: which agent called which tool with
 * what, and whether it worked. Each step opens to the full arguments and the result.
 */
export function ToolTimeline({ steps }: { steps: ToolStep[] }) {
  if (steps.length === 0) return null;
  const running = steps.filter((s) => s.status === "running").length;
  const failed = steps.filter((s) => s.status === "error").length;
  return (
    <details className="group/timeline mb-3 rounded-md border border-border bg-background" open={running > 0}>
      <summary className="cursor-pointer select-none px-3 py-2 text-xs text-muted hover:text-foreground">
        {steps.length} tool call{steps.length === 1 ? "" : "s"}
        {running > 0 ? ` · ${running} running` : ""}
        {failed > 0 ? <span className="text-danger"> · {failed} failed</span> : null}
      </summary>
      <ol aria-label="Tool calls" className="relative flex flex-col gap-1 border-t border-border px-3 py-2">
        {steps.map((step, index) => {
          const status = STATUS[step.status];
          return (
            <li key={`${step.id}-${index}`} className="text-xs">
              <details>
                <summary className="flex cursor-pointer items-baseline gap-2 py-0.5">
                  <span
                    aria-label={status.label}
                    title={status.label}
                    className={`w-3 shrink-0 text-center font-bold ${status.className}`}
                  >
                    {status.mark}
                  </span>
                  <span className="font-mono font-medium">{step.name}</span>
                  <span className="rounded bg-border/70 px-1.5 text-[10px] uppercase tracking-wide text-muted">
                    {humanize(step.node)}
                  </span>
                  <span className="min-w-0 truncate font-mono text-muted">{summarizeArgs(step.args)}</span>
                </summary>
                <div className="ml-5 mt-1 flex flex-col gap-2 pb-2">
                  <div>
                    <p className="text-[10px] uppercase tracking-wide text-muted">Arguments</p>
                    <pre className="mt-0.5 max-h-48 overflow-auto rounded bg-surface p-2 font-mono text-[11px] whitespace-pre-wrap">
                      {JSON.stringify(step.args, null, 2)}
                    </pre>
                  </div>
                  {step.content !== undefined ? (
                    <div>
                      <p className="text-[10px] uppercase tracking-wide text-muted">
                        Result{step.truncated ? " (first 2,000 characters)" : ""}
                      </p>
                      <pre
                        className={`mt-0.5 max-h-64 overflow-auto rounded bg-surface p-2 font-mono text-[11px] whitespace-pre-wrap ${
                          step.status === "error" ? "text-danger" : ""
                        }`}
                      >
                        {step.content}
                      </pre>
                    </div>
                  ) : null}
                </div>
              </details>
            </li>
          );
        })}
      </ol>
    </details>
  );
}
