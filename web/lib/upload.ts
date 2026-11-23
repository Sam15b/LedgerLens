import { apiBaseUrl, apiErrorMessage, type UploadOut, UNREACHABLE } from "@/lib/api";

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
