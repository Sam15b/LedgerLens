"use client";

import { useEffect, useRef, useState } from "react";

import { ACCEPT, checkFile, uploadFile, waitForDocument } from "@/lib/upload";

export type Attachment = {
  key: number;
  name: string;
  state: "uploading" | "processing" | "ready" | "failed";
  progress: number;
  documentId?: string;
  message?: string;
};

/**
 * Files attached to the next chat message: each is uploaded (`POST /documents`) and waited
 * on until the worker has ingested it, so the agents can read it (`document_ids`).
 */
export function useAttachments() {
  const [items, setItems] = useState<Attachment[]>([]);
  const nextKey = useRef(0);
  const controllers = useRef(new Map<number, AbortController>());

  useEffect(() => {
    const all = controllers.current;
    return () => all.forEach((c) => c.abort());
  }, []);

  function update(key: number, change: Partial<Attachment>) {
    setItems((current) => current.map((it) => (it.key === key ? { ...it, ...change } : it)));
  }

  async function attach(file: File) {
    const key = nextKey.current++;
    const problem = checkFile(file);
    setItems((current) => [
      ...current,
      { key, name: file.name, state: problem ? "failed" : "uploading", progress: 0, message: problem ?? undefined },
    ]);
    if (problem) return;
    const controller = new AbortController();
    controllers.current.set(key, controller);
    try {
      const outcome = await uploadFile(file, {
        signal: controller.signal,
        onProgress: (share) => update(key, { progress: share }),
      });
      if (!outcome.ok) {
        update(key, { state: "failed", message: outcome.message });
        return;
      }
      const documentId = outcome.body.document_id;
      update(key, { state: "processing", documentId, progress: 1 });
      const settled = await waitForDocument(documentId, { signal: controller.signal });
      update(
        key,
        settled.status === "ready"
          ? { state: "ready" }
          : { state: "failed", message: settled.error ?? "Ingestion failed." },
      );
    } catch (error) {
      if (!controller.signal.aborted) {
        update(key, { state: "failed", message: error instanceof Error ? error.message : String(error) });
      }
    } finally {
      controllers.current.delete(key);
    }
  }

  function remove(key: number) {
    controllers.current.get(key)?.abort();
    setItems((current) => current.filter((it) => it.key !== key));
  }

  return {
    items,
    add: (files: File[]) => files.forEach((file) => void attach(file)),
    remove,
    clear: () => setItems([]),
    busy: items.some((it) => it.state === "uploading" || it.state === "processing"),
    readyIds: items.flatMap((it) => (it.state === "ready" && it.documentId ? [it.documentId] : [])),
  };
}

const LABELS: Record<Attachment["state"], string> = {
  uploading: "uploading",
  processing: "ingesting…",
  ready: "ready",
  failed: "failed",
};

export function AttachButton({ onFiles, disabled }: { onFiles: (files: File[]) => void; disabled?: boolean }) {
  const input = useRef<HTMLInputElement>(null);
  return (
    <>
      <button
        type="button"
        disabled={disabled}
        onClick={() => input.current?.click()}
        title="Attach an invoice, receipt, purchase order or contract"
        className="rounded-md border border-border px-2.5 py-1.5 text-sm text-muted hover:text-foreground disabled:opacity-50"
      >
        Attach
      </button>
      <input
        ref={input}
        type="file"
        multiple
        accept={ACCEPT}
        aria-label="Attach documents"
        className="sr-only"
        onChange={(event) => {
          const files = Array.from(event.target.files ?? []);
          event.target.value = "";
          if (files.length > 0) onFiles(files);
        }}
      />
    </>
  );
}

export function AttachmentList({ items, onRemove }: { items: Attachment[]; onRemove: (key: number) => void }) {
  if (items.length === 0) return null;
  return (
    <ul aria-label="Attachments" className="flex flex-wrap gap-2">
      {items.map((item) => (
        <li
          key={item.key}
          title={item.message}
          className={`flex items-center gap-2 rounded-full border px-3 py-1 text-xs ${
            item.state === "failed" ? "border-danger/50 text-danger" : "border-border"
          }`}
        >
          <span className="max-w-48 truncate font-medium">{item.name}</span>
          <span className={item.state === "ready" ? "text-success" : "text-muted"}>
            {item.state === "uploading" ? `${Math.round(item.progress * 100)}%` : LABELS[item.state]}
          </span>
          <button
            type="button"
            aria-label={`Remove ${item.name}`}
            onClick={() => onRemove(item.key)}
            className="text-muted hover:text-foreground"
          >
            ×
          </button>
        </li>
      ))}
    </ul>
  );
}
