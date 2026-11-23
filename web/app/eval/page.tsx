import type { Metadata } from "next";
import { connection } from "next/server";
import { Suspense } from "react";

import { ErrorPanel, Placeholder } from "@/components/ErrorPanel";
import { EvalDashboard } from "@/components/EvalDashboard";
import { api, load } from "@/lib/api";
import { bySuite, numericMetrics } from "@/lib/eval";

export const metadata: Metadata = { title: "Evaluation" };

const RUNS = 500;

function first(value: string | string[] | undefined): string | undefined {
  return Array.isArray(value) ? value[0] : value;
}

export default function EvalPage({ searchParams }: PageProps<"/eval">) {
  return (
    <section>
      <h1 className="text-2xl font-semibold tracking-tight">Evaluation</h1>
      <p className="mt-1 text-sm text-muted">
        Retrieval and answer quality, latency and cost from each recorded run of the eval
        harness (<code className="font-mono text-xs">python -m eval.runner</code>). The full
        suite is the 200-question golden set; ci is the 40-question subset the CI gate runs.
      </p>
      <Suspense fallback={<Placeholder className="mt-6 h-96" />}>
        <EvalData searchParams={searchParams} />
      </Suspense>
    </section>
  );
}

async function EvalData({ searchParams }: { searchParams: PageProps<"/eval">["searchParams"] }) {
  const params = await searchParams;
  await connection();
  const result = await load((init) => api.GET("/eval/runs", { ...init, params: { query: { limit: RUNS } } }));
  if (!result.ok) {
    return (
      <div className="mt-6">
        <ErrorPanel title="The evaluation runs could not be loaded" message={result.message} />
      </div>
    );
  }
  const suites = bySuite(result.data);
  if (suites.size === 0) {
    return (
      <p className="mt-6 rounded-lg border border-dashed border-border px-4 py-8 text-center text-sm text-muted">
        No evaluation runs are recorded yet. Run the eval harness with a database to store one.
      </p>
    );
  }
  const names = [...suites.keys()];
  const asked = first(params.suite);
  const suite = asked && suites.has(asked) ? asked : suites.has("full") ? "full" : names[0]!;
  const runs = suites.get(suite)!;
  const metrics = numericMetrics(runs[0]!);
  const askedMetric = first(params.metric);
  const metric = askedMetric && askedMetric in metrics ? askedMetric : "recall@5" in metrics ? "recall@5" : Object.keys(metrics)[0] ?? "recall@5";
  return <EvalDashboard suites={names} suite={suite} runs={runs} metric={metric} />;
}
