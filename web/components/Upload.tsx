"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { type DragEvent, useRef, useState } from "react";

import { formatBytes } from "@/lib/format";
import { ACCEPT, checkFile, MAX_UPLOAD_MB, uploadFile } from "@/lib/upload";

type Item = {
  key: number;
  name: string;
  size: number;
  progress: number;
  state: "uploading" | "queued" | "duplicate" | "error";
  documentId?: string;
  message?: string;
};

/**
 * Drop files (or pick them) to upload them one after another, each with its progress.
 * A finished upload links to its document and refreshes the list below.
 */
export function Upload() {
  const router = useRouter();
  const input = useRef<HTMLInputElement>(null);
  const nextKey = useRef(0);
  const [items, setItems] = useState<Item[]>([]);
  const [dragging, setDragging] = useState(false);
  const depth = useRef(0); // dragenter/leave fire for every child the pointer crosses

  function update(key: number, change: Partial<Item>) {
    setItems((current) => current.map((it) => (it.key === key ? { ...it, ...change } : it)));
  }

  async function send(files: File[]) {
    const queued = files.map((file) => {
      const problem = checkFile(file);
      const item: Item = {
        key: nextKey.current++,
        name: file.name,
        size: file.size,
        progress: 0,
        state: problem ? "error" : "uploading",
        message: problem ?? undefined,
      };
      return { file, item, problem };
    });
    setItems((current) => [...queued.map((q) => q.item), ...current]);
    for (const { file, item, problem } of queued) {
      if (problem) continue;
      const outcome = await uploadFile(file, {
        onProgress: (share) => update(item.key, { progress: share }),
      });
      if (outcome.ok) {
        update(item.key, {
          progress: 1,
          state: outcome.body.duplicate ? "duplicate" : "queued",
          documentId: outcome.body.document_id,
        });
        router.refresh();
      } else {
        update(item.key, { state: "error", message: outcome.message });
      }
    }
  }

  function onDrop(event: DragEvent) {
    event.preventDefault();
    depth.current = 0;
    setDragging(false);
    const files = Array.from(event.dataTransfer.files);
    if (files.length > 0) void send(files);
  }

  return (
    <div className="flex flex-col gap-3">
      <div
        data-testid="dropzone"
        onDragEnter={(event) => {
          event.preventDefault();
          depth.current += 1;
          setDragging(true);
        }}
        onDragOver={(event) => event.preventDefault()}
        onDragLeave={() => {
          depth.current = Math.max(0, depth.current - 1);
          if (depth.current === 0) setDragging(false);
        }}
        onDrop={onDrop}
        className={`flex flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed px-6 py-8 text-center transition-colors ${
          dragging ? "border-accent bg-accent-soft" : "border-border bg-surface"
        }`}
      >
        <p className="text-sm font-medium">Drop invoices, receipts, purchase orders or contracts</p>
        <p className="text-xs text-muted">
          PDF, PNG, JPEG or TIFF, up to {MAX_UPLOAD_MB} MB each
        </p>
        <button
          type="button"
          onClick={() => input.current?.click()}
          className="mt-1 rounded-md bg-accent px-3 py-1.5 text-sm font-medium text-surface hover:opacity-90"
        >
          Choose files
        </button>
        <input
          ref={input}
          type="file"
          multiple
          accept={ACCEPT}
          aria-label="Files to upload"
          className="sr-only"
          onChange={(event) => {
            const files = Array.from(event.target.files ?? []);
            event.target.value = ""; // picking the same file again fires change again
            if (files.length > 0) void send(files);
          }}
        />
      </div>

      {items.length > 0 ? (
        <ul aria-label="Uploads" className="flex flex-col gap-2">
          {items.map((item) => (
            <UploadRow key={item.key} item={item} />
          ))}
        </ul>
      ) : null}
    </div>
  );
}

function UploadRow({ item }: { item: Item }) {
  const percent = Math.round(item.progress * 100);
  return (
    <li className="rounded-md border border-border bg-surface px-3 py-2 text-sm">
      <div className="flex items-center justify-between gap-3">
        <span className="min-w-0 truncate font-medium" title={item.name}>
          {item.name}
        </span>
        <span className="shrink-0 text-xs text-muted">{formatBytes(item.size)}</span>
      </div>
      {item.state === "uploading" ? (
        <div className="mt-2 flex items-center gap-2">
          <div
            role="progressbar"
            aria-label={`Uploading ${item.name}`}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={percent}
            className="h-1.5 flex-1 overflow-hidden rounded-full bg-border"
          >
            <div className="h-full rounded-full bg-accent transition-[width]" style={{ width: `${percent}%` }} />
          </div>
          <span className="w-10 text-right text-xs tabular-nums text-muted">{percent}%</span>
        </div>
      ) : null}
      {item.state === "queued" || item.state === "duplicate" ? (
        <p className="mt-1 text-xs text-success">
          {item.state === "queued" ? "Uploaded and queued for ingestion. " : "Already uploaded. "}
          <Link href={`/documents/${item.documentId}`} className="text-accent hover:underline">
            Open the document
          </Link>
        </p>
      ) : null}
      {item.state === "error" ? (
        <p role="alert" className="mt-1 text-xs text-danger">
          {item.message}
        </p>
      ) : null}
    </li>
  );
}
