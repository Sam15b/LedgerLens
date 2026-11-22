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
