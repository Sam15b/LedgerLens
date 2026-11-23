import type { DocumentStatus } from "@/lib/api";

export const DOCUMENT_STATUSES: DocumentStatus[] = ["queued", "processing", "ready", "failed"];

const STYLES: Record<DocumentStatus, { label: string; className: string; dot: string }> = {
  queued: { label: "Queued", className: "text-muted", dot: "bg-muted" },
  processing: { label: "Processing", className: "text-accent", dot: "bg-accent animate-pulse" },
  ready: { label: "Ready", className: "text-success", dot: "bg-success" },
  failed: { label: "Failed", className: "text-danger", dot: "bg-danger" },
};

/** Whether the worker may still change the document (the page keeps refreshing). */
export function inProgress(status: DocumentStatus): boolean {
  return status === "queued" || status === "processing";
}

/** A document's pipeline status: a word and a dot, never the colour alone. */
export function StatusBadge({ status, title }: { status: DocumentStatus; title?: string }) {
  const style = STYLES[status];
  return (
    <span
      title={title}
      className={`inline-flex items-center gap-1.5 rounded-full border border-border px-2 py-0.5 text-xs font-medium ${style.className}`}
    >
      <span aria-hidden className={`size-1.5 rounded-full ${style.dot}`} />
      {style.label}
    </span>
  );
}
