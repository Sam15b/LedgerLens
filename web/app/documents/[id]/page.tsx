import type { Metadata } from "next";
import Link from "next/link";
import { connection } from "next/server";
import { type ReactNode, Suspense } from "react";

import { AutoRefresh } from "@/components/AutoRefresh";
import { ErrorPanel, Placeholder } from "@/components/ErrorPanel";
import { ExtractedFields } from "@/components/ExtractedFields";
import { inProgress, StatusBadge } from "@/components/StatusBadge";
import { api, type DocumentDetail, load, publicApiUrl } from "@/lib/api";
import { formatCount, formatDateTime, humanize } from "@/lib/format";

export const metadata: Metadata = { title: "Document" };

export default function DocumentPage({ params }: PageProps<"/documents/[id]">) {
  return (
    <section>
      <Link href="/documents" className="text-sm text-accent hover:underline">
        ← All documents
      </Link>
      <Suspense fallback={<ViewerPending />}>
        <DocumentViewer params={params} />
      </Suspense>
    </section>
  );
}

async function DocumentViewer({ params }: { params: PageProps<"/documents/[id]">["params"] }) {
  const { id } = await params;
  await connection();
  const result = await load((init) =>
    api.GET("/documents/{document_id}", { ...init, params: { path: { document_id: id } } }),
  );
  if (!result.ok) {
    return (
      <div className="mt-4">
        <ErrorPanel
          title={result.status === 404 ? "No such document" : "The document could not be loaded"}
          message={result.message}
        />
      </div>
    );
  }
  const doc = result.data;
  return (
    <div className="mt-3 flex flex-col gap-6">
      <header className="flex flex-col gap-2">
        <div className="flex flex-wrap items-center gap-3">
          <h1 className="min-w-0 truncate text-2xl font-semibold tracking-tight" title={doc.filename}>
            {doc.filename}
          </h1>
          <StatusBadge status={doc.status} />
        </div>
        <p className="text-sm text-muted">
          {[
            humanize(doc.doc_type),
            doc.pages ? `${formatCount(doc.pages)} page${doc.pages === 1 ? "" : "s"}` : null,
            doc.ocr_engine ? `OCR: ${doc.ocr_engine}` : doc.pages ? "Text layer" : null,
            doc.chunks ? `${formatCount(doc.chunks)} chunks indexed` : null,
            `Uploaded ${formatDateTime(doc.created_at)}`,
          ]
            .filter(Boolean)
            .join(" · ")}
        </p>
        <AutoRefresh active={inProgress(doc.status)} />
      </header>
      {doc.status === "failed" ? (
        <ErrorPanel title="Ingestion failed" message={doc.error ?? "No reason was recorded."} />
      ) : null}
      <div className="grid gap-6 lg:grid-cols-2">
        <Panel title="Original">
          <Preview doc={doc} />
        </Panel>
        <Panel title="Extracted fields">
          {doc.record ? (
            <ExtractedFields record={doc.record} recordType={doc.record_type} recordId={doc.record_id} />
          ) : (
            <p className="text-sm text-muted">{noFieldsReason(doc)}</p>
          )}
        </Panel>
      </div>
    </div>
  );
}

function noFieldsReason(doc: DocumentDetail): string {
  if (inProgress(doc.status)) return "The fields appear here once the worker has stored the document.";
  if (doc.status === "failed") return "Nothing was extracted: ingestion failed.";
  if (doc.doc_type === "unknown") return "The document's type was not recognised, so no fields were extracted.";
  return "No record is stored for this document.";
}

function Panel({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="min-w-0 rounded-lg border border-border bg-surface p-4">
      <h2 className="mb-3 text-sm font-medium">{title}</h2>
      {children}
    </section>
  );
}

/** The uploaded file: the browser's PDF viewer, or the image itself. */
function Preview({ doc }: { doc: DocumentDetail }) {
  if (!doc.has_file) {
    return (
      <p className="text-sm text-muted">
        The original file is not kept for this document (only files uploaded here are).
      </p>
    );
  }
  const src = `${publicApiUrl()}/documents/${doc.id}/file`;
  if (doc.mime_type?.startsWith("image/") && doc.mime_type !== "image/tiff") {
    return (
      // An uploaded file from the API, not a static asset next/image could optimise.
      // eslint-disable-next-line @next/next/no-img-element
      <img src={src} alt={`Scan of ${doc.filename}`} className="w-full rounded-md border border-border" />
    );
  }
  if (doc.mime_type === "application/pdf") {
    return (
      <iframe
        src={src}
        title={`${doc.filename} (PDF)`}
        className="h-[70vh] w-full rounded-md border border-border bg-background"
      />
    );
  }
  return (
    <p className="text-sm">
      Browsers cannot show this file type.{" "}
      <a href={src} className="text-accent hover:underline">
        Download {doc.filename}
      </a>
    </p>
  );
}

function ViewerPending() {
  return (
    <div className="mt-3 flex flex-col gap-6" aria-busy="true" aria-label="Loading the document">
      <Placeholder className="h-14 w-2/3" />
      <div className="grid gap-6 lg:grid-cols-2">
        <Placeholder className="h-[60vh]" />
        <Placeholder className="h-80" />
      </div>
    </div>
  );
}
