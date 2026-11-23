import type { Metadata } from "next";
import Link from "next/link";
import { connection } from "next/server";
import { Suspense } from "react";

import { AutoRefresh } from "@/components/AutoRefresh";
import { DocumentsTable } from "@/components/DocumentsTable";
import { ErrorPanel, Placeholder } from "@/components/ErrorPanel";
import { DOCUMENT_STATUSES, inProgress } from "@/components/StatusBadge";
import { api, load } from "@/lib/api";
import {
  documentsHref,
  type DocumentsQuery,
  PAGE_SIZE,
  parseDocumentsQuery,
} from "@/lib/documents-query";
import { humanize } from "@/lib/format";

export const metadata: Metadata = { title: "Documents" };

export default function DocumentsPage({ searchParams }: PageProps<"/documents">) {
  return (
    <section>
      <h1 className="text-2xl font-semibold tracking-tight">Documents</h1>
      <p className="mt-1 text-sm text-muted">
        Uploaded invoices, receipts, purchase orders and contracts, newest first. Open one to
        see the fields extracted from it.
      </p>
      <Suspense fallback={<Placeholder className="mt-6 h-72" />}>
        <DocumentsList searchParams={searchParams} />
      </Suspense>
    </section>
  );
}

async function DocumentsList({
  searchParams,
}: {
  searchParams: PageProps<"/documents">["searchParams"];
}) {
  const query = parseDocumentsQuery(await searchParams);
  await connection();
  const result = await load((init) =>
    api.GET("/documents", {
      ...init,
      params: {
        query: {
          status: query.status ?? undefined,
          limit: PAGE_SIZE,
          offset: (query.page - 1) * PAGE_SIZE,
        },
      },
    }),
  );
  return (
    <div className="mt-6 flex flex-col gap-3">
      <StatusFilter query={query} />
      {result.ok ? (
        <>
          <DocumentsTable documents={result.data} />
          <div className="flex items-center justify-between gap-4">
            <AutoRefresh active={result.data.some((doc) => inProgress(doc.status))} />
            <Pager query={query} hasMore={result.data.length === PAGE_SIZE} />
          </div>
        </>
      ) : (
        <ErrorPanel title="The documents could not be loaded" message={result.message} />
      )}
    </div>
  );
}

function StatusFilter({ query }: { query: DocumentsQuery }) {
  const options = [null, ...DOCUMENT_STATUSES];
  return (
    <nav aria-label="Filter by status" className="flex flex-wrap gap-1">
      {options.map((status) => {
        const active = query.status === status;
        return (
          <Link
            key={status ?? "all"}
            href={documentsHref({ status, page: 1 })}
            aria-current={active ? "page" : undefined}
            className={`rounded-md px-3 py-1 text-sm ${
              active ? "bg-accent-soft font-medium text-accent" : "text-muted hover:text-foreground"
            }`}
          >
            {status ? humanize(status) : "All"}
          </Link>
        );
      })}
    </nav>
  );
}

function Pager({ query, hasMore }: { query: DocumentsQuery; hasMore: boolean }) {
  if (query.page === 1 && !hasMore) return null;
  const link = "rounded-md border border-border px-3 py-1 text-sm hover:border-accent";
  return (
    <div className="ml-auto flex items-center gap-2 text-sm">
      {query.page > 1 ? (
        <Link href={documentsHref({ ...query, page: query.page - 1 })} className={link}>
          Newer
        </Link>
      ) : null}
      <span className="text-muted">Page {query.page}</span>
      {hasMore ? (
        <Link href={documentsHref({ ...query, page: query.page + 1 })} className={link}>
          Older
        </Link>
      ) : null}
    </div>
  );
}
