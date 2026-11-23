import {
  api,
  apiBaseUrl,
  apiErrorMessage,
  load,
  type UploadOut,
  UNREACHABLE,
} from "@/lib/api";

/** File types ingestion takes (`loader.allowed_mime_types` in configs/ingestion.yaml). */
export const ACCEPTED_TYPES = ["application/pdf", "image/png", "image/jpeg", "image/tiff"];
export const ACCEPT = [...ACCEPTED_TYPES, ".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"].join(",");

/** `loader.max_file_mb`: checked here only to fail fast; the API enforces it (413). */
export const MAX_UPLOAD_MB = 25;

export type UploadOutcome =
  | { ok: true; body: UploadOut }
  | { ok: false; status: number | null; message: string };

/** Why ``file`` cannot be uploaded at all, or `null` when it may be sent. */
export function checkFile(file: Pick<File, "size">): string | null {
  if (file.size === 0) return "The file is empty.";
  if (file.size > MAX_UPLOAD_MB * 1024 * 1024) return `The file is over the ${MAX_UPLOAD_MB} MB limit.`;
  return null;
}

/**
 * `POST /documents` with ``file``, reporting the share sent so far (0..1) to
 * ``onProgress``. XMLHttpRequest, because fetch cannot report upload progress. Never
 * rejects: a refusal carries the API's message, a network failure says so.
 */
export function uploadFile(
  file: File,
  {
    onProgress,
    signal,
    baseUrl = apiBaseUrl(),
  }: { onProgress?: (share: number) => void; signal?: AbortSignal; baseUrl?: string } = {},
): Promise<UploadOutcome> {
  return new Promise((resolve) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${baseUrl}/documents`);
    xhr.responseType = "json";
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable && event.total > 0) onProgress?.(event.loaded / event.total);
    };
    xhr.onload = () => {
      const body: unknown = xhr.response;
      if (xhr.status === 200 || xhr.status === 202) {
        onProgress?.(1);
        resolve({ ok: true, body: body as UploadOut });
        return;
      }
      resolve({
        ok: false,
        status: xhr.status,
        message: apiErrorMessage(body) ?? `The upload failed (HTTP ${xhr.status}).`,
      });
    };
    xhr.onerror = () => resolve({ ok: false, status: null, message: UNREACHABLE });
    xhr.onabort = () => resolve({ ok: false, status: null, message: "The upload was cancelled." });
    signal?.addEventListener("abort", () => xhr.abort(), { once: true });
    const form = new FormData();
    form.append("file", file, file.name);
    xhr.send(form);
  });
}

export type Settled = { status: "ready" | "failed"; error: string | null };

/** `GET /documents/{id}`'s status, or `null` when it cannot be read right now. */
async function readStatus(documentId: string): Promise<Settled | { status: string } | null> {
  const result = await load((init) =>
    api.GET("/documents/{document_id}", { ...init, params: { path: { document_id: documentId } } }),
  );
  return result.ok ? { status: result.data.status, error: result.data.error ?? null } : null;
}

/**
 * Polls the document until the worker has finished with it (`ready` or `failed`). A
 * failed read is retried; aborting ``signal`` or running past ``timeoutMs`` rejects.
 */
export async function waitForDocument(
  documentId: string,
  {
    signal,
    everyMs = 2_000,
    timeoutMs = 10 * 60_000,
    read = readStatus,
  }: {
    signal?: AbortSignal;
    everyMs?: number;
    timeoutMs?: number;
    read?: (documentId: string) => Promise<Settled | { status: string } | null>;
  } = {},
): Promise<Settled> {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    signal?.throwIfAborted();
    const found = await read(documentId);
    if (found && (found.status === "ready" || found.status === "failed")) return found as Settled;
    if (Date.now() + everyMs > deadline) throw new Error("The document is still being processed.");
    await new Promise<void>((resolve, reject) => {
      const timer = setTimeout(resolve, everyMs);
      signal?.addEventListener(
        "abort",
        () => {
          clearTimeout(timer);
          reject(signal.reason);
        },
        { once: true },
      );
    });
  }
}
