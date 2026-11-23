/** Reading `GET /eval/runs`: which metrics to show, how to format them, which way is better. */
import type { Schemas } from "@/lib/api";
import { formatUsd } from "@/lib/format";

export type EvalRun = Schemas["EvalRunOut"];

/** The headline metrics, in display order (a run shows those it has). */
export const HEADLINE = [
  "recall@5",
  "mrr",
  "evidence_recall",
  "faithfulness",
  "citations_ok",
  "latency.retrieval.p95_ms",
  "latency.query.p95_ms",
  "cost.query.mean_usd",
];

const LABELS: Record<string, string> = {
  "recall@5": "Recall@5",
  mrr: "MRR",
  evidence_recall: "Evidence recall",
  faithfulness: "Faithfulness",
  citations_ok: "Citations correct",
  facts_found: "Facts found",
  questions: "Questions",
  answerable: "Answerable",
  answered: "Answered",
  errors: "Errors",
};

export function metricLabel(key: string): string {
  if (LABELS[key]) return LABELS[key];
  const shape = /^recall@5\.(.+)$/.exec(key);
  if (shape) return `Recall@5, ${shape[1]!.replace(/_/g, "-")}`;
  const latency = /^latency\.(\w+)\.(p\d+)_ms$/.exec(key);
  if (latency) return `${latency[1]} latency ${latency[2]}`.replace(/^./, (c) => c.toUpperCase());
  const cost = /^cost\.(.+)$/.exec(key);
  if (cost) return `Cost ${cost[1]!.replace(/[._]/g, " ").replace(/ usd$/, "")}`;
  return key;
}

/** Latency, cost and errors go down when things improve; scores go up. */
export function lowerIsBetter(key: string): boolean {
  return key.startsWith("latency.") || key.startsWith("cost.") || key === "errors";
}

export function isCount(key: string): boolean {
  return ["questions", "answerable", "answered", "errors"].includes(key);
}

export function formatMetric(key: string, value: number): string {
  if (key.endsWith("_ms")) return value >= 1000 ? `${(value / 1000).toFixed(1)} s` : `${Math.round(value)} ms`;
  if (key.startsWith("cost.")) return formatUsd(value);
  if (isCount(key)) return String(Math.round(value));
  return value.toFixed(3);
}

/** A run's numeric metrics only (`null` = not measured; strings are not plotted). */
export function numericMetrics(run: EvalRun): Record<string, number> {
  const out: Record<string, number> = {};
  for (const [key, value] of Object.entries(run.metrics)) {
    if (typeof value === "number" && Number.isFinite(value)) out[key] = value;
  }
  return out;
}

export type Change = { delta: number; better: boolean | null };

/** How ``key`` moved since ``previous`` (`better` is `null` when it did not move). */
export function change(key: string, latest: number, previous: number | undefined): Change | null {
  if (previous === undefined) return null;
  const delta = latest - previous;
  if (Math.abs(delta) < 1e-9) return { delta: 0, better: null };
  return { delta, better: lowerIsBetter(key) ? delta < 0 : delta > 0 };
}

/** Runs grouped by suite, each newest first; suites in order of their newest run. */
export function bySuite(runs: EvalRun[]): Map<string, EvalRun[]> {
  const sorted = [...runs].sort((a, b) => b.created_at.localeCompare(a.created_at));
  const out = new Map<string, EvalRun[]>();
  for (const run of sorted) out.set(run.suite, [...(out.get(run.suite) ?? []), run]);
  return out;
}
