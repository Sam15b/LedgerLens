import Link from "next/link";

import { splitCitations } from "@/lib/chat-state";

/** One `[doc:INV-889]` tag: a link to the document it names, or plain text if unknown. */
export function Citation({ label, documentId }: { label: string; documentId: string | null }) {
  if (!documentId) {
    return (
      <span title="No document was given for this tag" className="font-mono text-xs text-muted">
        [{label}]
      </span>
    );
  }
  return (
    <Link
      href={`/documents/${documentId}`}
      target="_blank"
      title={`Open ${label} (new tab)`}
      className="rounded bg-accent-soft px-1 py-0.5 font-mono text-xs text-accent no-underline hover:underline"
    >
      {label}
    </Link>
  );
}

/** The answer text with its citation tags turned into links to their documents. */
export function CitedText({ text, citations }: { text: string; citations: Record<string, string> }) {
  return (
    <p className="whitespace-pre-wrap leading-relaxed">
      {splitCitations(text, citations).map((segment, i) =>
        "tag" in segment ? (
          <Citation key={i} label={segment.label} documentId={segment.documentId} />
        ) : (
          <span key={i}>{segment.text}</span>
        ),
      )}
    </p>
  );
}

/** Every document the answer may cite, also those its text does not tag. */
export function Sources({ citations }: { citations: Record<string, string> }) {
  const entries = Object.entries(citations);
  if (entries.length === 0) return null;
  return (
    <div className="mt-3 border-t border-border pt-2">
      <p className="text-[10px] uppercase tracking-wide text-muted">Sources</p>
      <ul className="mt-1 flex flex-wrap gap-1.5">
        {entries.map(([tag, documentId]) => (
          <li key={tag}>
            <Citation label={tag.replace(/^doc:/, "")} documentId={documentId} />
          </li>
        ))}
      </ul>
    </div>
  );
}
