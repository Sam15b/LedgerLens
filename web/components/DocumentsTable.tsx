import Link from "next/link";

import { StatusBadge } from "@/components/StatusBadge";
import type { DocumentOut } from "@/lib/api";
import { formatDateTime, humanize } from "@/lib/format";

/** Uploaded documents, newest first; each row opens the document's viewer. */
export function DocumentsTable({ documents }: { documents: DocumentOut[] }) {
  if (documents.length === 0) {
    return (
      <p className="rounded-lg border border-dashed border-border px-4 py-8 text-center text-sm text-muted">
        No documents here yet. Upload an invoice, receipt, purchase order or contract above.
      </p>
    );
  }
  return (
    <div className="overflow-x-auto rounded-lg border border-border bg-surface">
      <table className="w-full text-left text-sm">
        <thead className="border-b border-border text-xs text-muted">
          <tr>
            <th scope="col" className="px-4 py-2 font-medium">
              Document
            </th>
            <th scope="col" className="px-4 py-2 font-medium">
              Type
            </th>
            <th scope="col" className="px-4 py-2 font-medium">
              Status
            </th>
            <th scope="col" className="px-4 py-2 text-right font-medium">
              Pages
            </th>
            <th scope="col" className="px-4 py-2 font-medium">
              Uploaded
            </th>
          </tr>
        </thead>
        <tbody>
          {documents.map((doc) => (
            <tr key={doc.id} className="border-t border-border first:border-t-0 hover:bg-background">
              <td className="max-w-80 truncate px-4 py-2">
                <Link
                  href={`/documents/${doc.id}`}
                  className="font-medium text-accent hover:underline"
                  title={doc.filename}
                >
                  {doc.filename}
                </Link>
              </td>
              <td className="px-4 py-2 text-muted">{humanize(doc.doc_type)}</td>
              <td className="px-4 py-2">
                <StatusBadge status={doc.status} title={doc.error ?? undefined} />
              </td>
              <td className="px-4 py-2 text-right tabular-nums text-muted">{doc.pages ?? "–"}</td>
              <td className="whitespace-nowrap px-4 py-2 text-muted">
                {formatDateTime(doc.created_at)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
