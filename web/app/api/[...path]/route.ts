/**
 * The browser's way to the LedgerLens API: `/api/<path>` is forwarded to the API with the
 * server's API key (`LEDGERLENS_API_KEY`), which never reaches the browser. Only signed-in
 * sessions may use it (when login is on). Bodies and responses stream through, so uploads
 * and the chat's server-sent events work unchanged.
 */
import { type NextRequest } from "next/server";

import { serverApiUrl } from "@/lib/api";
import { allowed, SESSION_COOKIE } from "@/lib/session";

const REQUEST_HEADERS = ["content-type", "accept", "x-request-id", "last-event-id"];
const RESPONSE_HEADERS = [
  "content-type",
  "content-disposition",
  "cache-control",
  "retry-after",
  "www-authenticate",
  "x-request-id",
  "x-thread-id",
  "x-content-type-options",
];

function refusal(status: number, code: string, message: string): Response {
  return Response.json({ error: { code, message, request_id: null } }, { status });
}

async function forward(request: NextRequest, { params }: RouteContext<"/api/[...path]">): Promise<Response> {
  if (!(await allowed(request.cookies.get(SESSION_COOKIE)?.value))) {
    return refusal(401, "unauthorized", "Sign in to the dashboard again.");
  }
  const { path } = await params;
  if (path.some((part) => part === ".." || part === "." || part.includes("\\"))) {
    return refusal(400, "bad_request", "invalid path");
  }
  const target = `${serverApiUrl()}/${path.map(encodeURIComponent).join("/")}${request.nextUrl.search}`;

  const headers = new Headers();
  for (const name of REQUEST_HEADERS) {
    const value = request.headers.get(name);
    if (value) headers.set(name, value);
  }
  const key = process.env.LEDGERLENS_API_KEY;
  if (key) headers.set("X-API-Key", key);
  const client = request.headers.get("x-forwarded-for")?.split(",")[0]?.trim();
  if (client) headers.set("X-Forwarded-For", client); // per-person rate limits on the API

  const hasBody = request.method !== "GET" && request.method !== "HEAD";
  let upstream: Response;
  try {
    upstream = await fetch(target, {
      method: request.method,
      headers,
      body: hasBody ? request.body : undefined,
      // @ts-expect-error -- Node's fetch needs this to stream a request body
      duplex: hasBody ? "half" : undefined,
      signal: request.signal,
      cache: "no-store",
      redirect: "manual",
    });
  } catch {
    return refusal(502, "bad_gateway", "The dashboard could not reach the API.");
  }

  const out = new Headers();
  for (const name of RESPONSE_HEADERS) {
    const value = upstream.headers.get(name);
    if (value) out.set(name, value);
  }
  return new Response(upstream.body, { status: upstream.status, headers: out });
}

export const GET = forward;
export const POST = forward;
export const PATCH = forward;
