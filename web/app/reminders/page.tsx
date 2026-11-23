import type { Metadata } from "next";
import Link from "next/link";
import { connection } from "next/server";
import { Suspense } from "react";

import { ErrorPanel, Placeholder } from "@/components/ErrorPanel";
import { ReminderCard } from "@/components/ReminderCard";
import { api, load } from "@/lib/api";

export const metadata: Metadata = { title: "Reminders" };

const FILTERS = [
  { value: "pending_approval", label: "Waiting for approval" },
  { value: "approved", label: "Approved" },
  { value: "rejected", label: "Rejected" },
  { value: "all", label: "All" },
] as const;

type Filter = (typeof FILTERS)[number]["value"];

function parseFilter(value: string | string[] | undefined): Filter {
  const text = Array.isArray(value) ? value[0] : value;
  return FILTERS.find((f) => f.value === text)?.value ?? "pending_approval";
}

export default function RemindersPage({ searchParams }: PageProps<"/reminders">) {
  return (
    <section>
      <h1 className="text-2xl font-semibold tracking-tight">Reminders</h1>
      <p className="mt-1 text-sm text-muted">
        Payment reminders the agents drafted. Edit a draft, then approve or reject it.
        LedgerLens never sends anything: send an approved reminder yourself.
      </p>
      <Suspense fallback={<Placeholder className="mt-6 h-96" />}>
        <ReminderQueue searchParams={searchParams} />
      </Suspense>
    </section>
  );
}

async function ReminderQueue({ searchParams }: { searchParams: PageProps<"/reminders">["searchParams"] }) {
  const filter = parseFilter((await searchParams).status);
  await connection();
  const result = await load((init) =>
    api.GET("/reminders", { ...init, params: { query: { status: filter, limit: 100 } } }),
  );
  return (
    <div className="mt-6 flex flex-col gap-3">
      <nav aria-label="Filter by status" className="flex flex-wrap gap-1">
        {FILTERS.map((f) => (
          <Link
            key={f.value}
            href={f.value === "pending_approval" ? "/reminders" : `/reminders?status=${f.value}`}
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
        <ErrorPanel title="The reminders could not be loaded" message={result.message} />
      ) : result.data.length === 0 ? (
        <p className="rounded-lg border border-dashed border-border px-4 py-8 text-center text-sm text-muted">
          {filter === "pending_approval"
            ? "No drafts waiting. Ask the chat to draft reminders for overdue invoices."
            : "Nothing here."}
        </p>
      ) : (
        <ul className="flex flex-col gap-3">
          {result.data.map((reminder) => (
            <li key={reminder.id}>
              <ReminderCard reminder={reminder} />
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
