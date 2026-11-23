import type { DocumentStatus } from "@/lib/api";

export const PAGE_SIZE = 50;

const STATUSES = new Set<string>(["queued", "processing", "ready", "failed"]);

export type DocumentsQuery = { status: DocumentStatus | null; page: number };

type SearchParams = Record<string, string | string[] | undefined>;

function first(value: string | string[] | undefined): string | undefined {
  return Array.isArray(value) ? value[0] : value;
}

/** `?status=ready&page=2` -> `{status: "ready", page: 2}`; anything unknown is dropped. */
export function parseDocumentsQuery(params: SearchParams): DocumentsQuery {
  const status = first(params.status);
  const page = Number.parseInt(first(params.page) ?? "", 10);
  return {
    status: status && STATUSES.has(status) ? (status as DocumentStatus) : null,
    page: Number.isFinite(page) && page > 1 ? page : 1,
  };
}

/** The list's URL for ``query`` (page 1 and "all statuses" are left out). */
export function documentsHref(query: DocumentsQuery): string {
  const search = new URLSearchParams();
  if (query.status) search.set("status", query.status);
  if (query.page > 1) search.set("page", String(query.page));
  const text = search.toString();
  return text ? `/documents?${text}` : "/documents";
}
