/**
 * Typed client for the LedgerLens API.
 *
 * Paths, parameters and response bodies come from the API's OpenAPI schema:
 * `python scripts/export_openapi.py` writes `web/openapi.json` and `pnpm gen:api` turns it
 * into `lib/api-schema.ts` (both committed; a Python test fails when they fall behind the
 * API). A call to a path or with a body the API does not have is a type error.
 *
 *     const { data, error } = await api.GET("/documents/{document_id}", {
 *       params: { path: { document_id: id } },
 *     });
 */
import createClient from "openapi-fetch";

import type { components, paths } from "@/lib/api-schema";

export type Schemas = components["schemas"];
export type HealthResponse = Schemas["HealthResponse"];
export type Overview = Schemas["Overview"];
export type DocumentOut = Schemas["DocumentOut"];
export type DocumentDetail = Schemas["DocumentDetail"];
export type DocumentStatus = Schemas["DocumentStatus"];
export type UploadOut = Schemas["UploadOut"];

const DEFAULT_API_URL = "http://localhost:8000";

/**
 * Where the API is. The browser uses `NEXT_PUBLIC_API_URL` (inlined at build time); the
 * server prefers `LEDGERLENS_API_URL`, which can be a private address such as
 * `http://api:8000` inside docker compose.
 */
export function apiBaseUrl(): string {
  const shared = process.env.NEXT_PUBLIC_API_URL;
  if (typeof window === "undefined") {
    return process.env.LEDGERLENS_API_URL ?? shared ?? DEFAULT_API_URL;
  }
  return shared ?? DEFAULT_API_URL;
}

/**
 * The API's address as the browser sees it, for URLs a server component puts in the page
 * (an iframe's `src`): never the private `LEDGERLENS_API_URL`.
 */
export function publicApiUrl(): string {
  return process.env.NEXT_PUBLIC_API_URL ?? DEFAULT_API_URL;
}

export function createApiClient(baseUrl: string = apiBaseUrl()) {
  return createClient<paths>({ baseUrl });
}

export const api = createApiClient();

/** How long the dashboard waits for the API before showing it as unreachable. */
export const API_TIMEOUT_MS = 8_000;

/** The API's `/health`, or `null` when it cannot be reached. */
export async function getHealth(): Promise<HealthResponse | null> {
  try {
    const { data } = await api.GET("/health", {
      cache: "no-store",
      signal: AbortSignal.timeout(API_TIMEOUT_MS),
    });
    return data ?? null;
  } catch {
    return null;
  }
}

/** What a page shows: the body, or why there is none. */
export type ApiResult<T> =
  | { ok: true; data: T }
  | { ok: false; status: number | null; message: string };

/**
 * The message of the API's error body (`{"error": {"code", "message", "request_id"}}`),
 * with the request id when there is one, or `null` for any other body.
 */
export function apiErrorMessage(body: unknown): string | null {
  if (typeof body !== "object" || body === null || !("error" in body)) return null;
  const error = (body as { error: unknown }).error;
  if (typeof error !== "object" || error === null) return null;
  const { message, request_id: requestId } = error as { message?: unknown; request_id?: unknown };
  if (typeof message !== "string") return null;
  return typeof requestId === "string" ? `${message} (request ${requestId})` : message;
}

export const UNREACHABLE = "The API could not be reached. Is it running (make api)?";

type Call<T> = (init: {
  cache: RequestCache;
  signal: AbortSignal;
}) => Promise<{ data?: T; error?: unknown; response: Response }>;

/**
 * Runs one API call for a page, uncached and with a timeout, and turns every outcome into
 * an {@link ApiResult}: never throws. Await `connection()` before calling it from a server
 * component (the timeout reads the clock).
 *
 *     const overview = await load((init) => api.GET("/stats/overview", init));
 */
export async function load<T>(call: Call<T>): Promise<ApiResult<T>> {
  try {
    const { data, error, response } = await call({
      cache: "no-store",
      signal: AbortSignal.timeout(API_TIMEOUT_MS),
    });
    if (response.ok && data !== undefined) return { ok: true, data };
    return {
      ok: false,
      status: response.status,
      message: apiErrorMessage(error) ?? `The API answered ${response.status}.`,
    };
  } catch {
    return { ok: false, status: null, message: UNREACHABLE };
  }
}
