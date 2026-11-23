import type { Metadata } from "next";
import Link from "next/link";
import { connection } from "next/server";
import { Suspense } from "react";

import { AnomalyCard } from "@/components/AnomalyCard";
import { ErrorPanel, Placeholder } from "@/components/ErrorPanel";
import { ReviewActions } from "@/components/ReviewActions";
import { api, load } from "@/lib/api";

export const metadata: Metadata = { title: "Anomalies" };

const FILTERS = [
  { value: "unreviewed", label: "To review" },
  { value: "confirmed", label: "Confirmed" },
  { value: "false_positive", label: "False positives" },
  { value: "all", label: "All" },
] as const;

type Filter = (typeof FILTERS)[number]["value"];

function parseFilter(value: string | string[] | undefined): Filter {
  const text = Array.isArray(value) ? value[0] : value;
  return FILTERS.find((f) => f.value === text)?.value ?? "unreviewed";
}

export default function AnomaliesPage({ searchParams }: PageProps<"/anomalies">) {
  return (
    <section>
      <h1 className="text-2xl font-semibold tracking-tight">Anomalies</h1>
      <p className="mt-1 text-sm text-muted">
        Invoices the anomaly scorer flagged, highest score first, with the rules and features
        behind each score. Your verdicts are kept as labels for retraining.
      </p>
      <Suspense fallback={<Placeholder className="mt-6 h-96" />}>
        <AnomalyList searchParams={searchParams} />
      </Suspense>
    </section>
  );
}

async function AnomalyList({ searchParams }: { searchParams: PageProps<"/anomalies">["searchParams"] }) {
  const filter = parseFilter((await searchParams).review);
  await connection();
  const result = await load((init) =>
    api.GET("/anomalies", { ...init, params: { query: { review: filter, limit: 100 } } }),
  );
  return (
    <div className="mt-6 flex flex-col gap-3">
      <nav aria-label="Filter by review" className="flex flex-wrap gap-1">
        {FILTERS.map((f) => (
          <Link
            key={f.value}
            href={f.value === "unreviewed" ? "/anomalies" : `/anomalies?review=${f.value}`}
            aria-current={filter === f.value ? "page" : undefined}
            className={`rounded-md px-3 py-1 text-sm ${
              filter === f.value ? "bg-accent-soft font-medium text-accent" : "text-muted hover:text-foreground"
            }`}
          >
            {f.label}
          </Link>
        ))}
      </nav>
      {!result.ok ? (
        <ErrorPanel title="The anomalies could not be loaded" message={result.message} />
      ) : result.data.length === 0 ? (
        <p className="rounded-lg border border-dashed border-border px-4 py-8 text-center text-sm text-muted">
          {filter === "unreviewed" ? "No flags waiting for review." : "Nothing here."}
        </p>
      ) : (
        <>
          <p className="text-xs text-muted">
            {result.data.length === 100 ? "The 100 highest scores" : `${result.data.length} flagged invoices`}
          </p>
          <ul className="flex flex-col gap-3">
            {result.data.map((anomaly) => (
              <li key={anomaly.id}>
                <AnomalyCard
                  anomaly={anomaly}
                  actions={<ReviewActions scoreId={anomaly.id} current={anomaly.review} />}
                />
              </li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}
