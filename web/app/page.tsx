import { connection } from "next/server";
import { Suspense } from "react";

import { ErrorPanel, Placeholder } from "@/components/ErrorPanel";
import { OverviewCards } from "@/components/OverviewCards";
import { api, load } from "@/lib/api";

const COST_DAYS = 14;

export default function OverviewPage() {
  return (
    <section>
      <h1 className="text-2xl font-semibold tracking-tight">Overview</h1>
      <p className="mt-1 text-sm text-muted">
        Documents ingested, what is unpaid or overdue, flagged invoices and LLM spend.
      </p>
      <Suspense fallback={<OverviewPending />}>
        <OverviewData />
      </Suspense>
    </section>
  );
}

/** The numbers change with every upload and payment: read them on each request. */
async function OverviewData() {
  await connection();
  const result = await load((init) =>
    api.GET("/stats/overview", { ...init, params: { query: { days: COST_DAYS } } }),
  );
  if (!result.ok) {
    return (
      <div className="mt-6">
        <ErrorPanel title="The overview could not be loaded" message={result.message} />
      </div>
    );
  }
  return <OverviewCards overview={result.data} />;
}

function OverviewPending() {
  return (
    <div className="mt-6 flex flex-col gap-4" aria-busy="true" aria-label="Loading the overview">
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
        {Array.from({ length: 5 }, (_, i) => (
          <Placeholder key={i} className="h-28" />
        ))}
      </div>
      <Placeholder className="h-56" />
    </div>
  );
}
