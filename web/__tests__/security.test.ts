import { NextRequest } from "next/server";
import { afterEach, describe, expect, it, vi } from "vitest";

import { GET, POST } from "@/app/api/[...path]/route";
import { proxy } from "@/proxy";
import {
  allowed,
  createSession,
  loginRequired,
  safeNext,
  sameText,
  SESSION_COOKIE,
  SESSION_HOURS,
  validSession,
  type Env,
} from "@/lib/session";

const ENV: Env = { DASHBOARD_PASSWORD: "correct horse battery staple" };
const NOW = Date.UTC(2026, 10, 23, 12);

afterEach(() => {
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
});

describe("session", () => {
  it("is off without a password", async () => {
    expect(loginRequired({})).toBe(false);
    expect(await allowed(undefined, {})).toBe(true);
    expect(await allowed(undefined, ENV)).toBe(false);
  });

  it("accepts its own cookie until it expires", async () => {
    const { value, expires } = await createSession(ENV, NOW);
    expect(expires.getTime()).toBe(NOW + SESSION_HOURS * 3600_000);
    expect(await validSession(value, ENV, NOW + 1000)).toBe(true);
    expect(await validSession(value, ENV, expires.getTime())).toBe(false);
  });

  it("refuses a tampered, foreign or malformed cookie", async () => {
    const { value } = await createSession(ENV, NOW);
    const [expiry, signature] = value.split(".");
    const later = `${Number(expiry) + 3600_000}.${signature}`;
    expect(await validSession(later, ENV, NOW)).toBe(false);
    const other: Env = { DASHBOARD_PASSWORD: "another password" };
    expect(await validSession(value, other, NOW)).toBe(false);
    const secret: Env = { ...ENV, DASHBOARD_SESSION_SECRET: "rotated" };
    expect(await validSession(value, secret, NOW)).toBe(false);
    for (const bad of ["", "abc", `${expiry}`, `${expiry}.${signature}.x`, `-1.${signature}`]) {
      expect(await validSession(bad, ENV, NOW)).toBe(false);
    }
  });

  it("compares strings fully", () => {
    expect(sameText("abc", "abc")).toBe(true);
    expect(sameText("abc", "abd")).toBe(false);
    expect(sameText("abc", "abcd")).toBe(false);
    expect(sameText("", "")).toBe(true);
  });

  it("only returns to paths on this site", () => {
    expect(safeNext("/documents?status=ready")).toBe("/documents?status=ready");
    for (const bad of [null, "", "https://evil.example", "//evil.example", "/\\evil.example", "documents"]) {
      expect(safeNext(bad)).toBe("/");
    }
  });
});

describe("login gate", () => {
  it("sends a page request without a session to the login page", async () => {
    vi.stubEnv("DASHBOARD_PASSWORD", ENV.DASHBOARD_PASSWORD!);
    const response = await proxy(new NextRequest("http://dash.test/anomalies?review=all"));
    expect(response.status).toBe(307);
    const location = new URL(response.headers.get("location")!);
    expect(location.pathname).toBe("/login");
    expect(location.searchParams.get("next")).toBe("/anomalies?review=all");
  });

  it("redirects on the scheme the browser used behind CloudFront", async () => {
    vi.stubEnv("DASHBOARD_PASSWORD", ENV.DASHBOARD_PASSWORD!);
    // the load balancer forwards plain HTTP; CloudFront reports the viewer's HTTPS
    const request = new NextRequest("http://d1.cloudfront.net/eval", {
      headers: { "cloudfront-forwarded-proto": "https" },
    });
    const location = new URL((await proxy(request)).headers.get("location")!);
    expect(location.origin).toBe("https://d1.cloudfront.net");
    expect(location.pathname).toBe("/login");
  });

  it("lets a signed-in request through", async () => {
    vi.stubEnv("DASHBOARD_PASSWORD", ENV.DASHBOARD_PASSWORD!);
    const { value } = await createSession(process.env);
    const request = new NextRequest("http://dash.test/", { headers: { cookie: `${SESSION_COOKIE}=${value}` } });
    const response = await proxy(request);
    expect(response.headers.get("location")).toBeNull();
    expect(response.headers.get("x-middleware-next")).toBe("1");
  });
});

function context(...path: string[]) {
  return { params: Promise.resolve({ path }) };
}

describe("API forwarder", () => {
  it("adds the server's key and streams the answer back", async () => {
    vi.stubEnv("LEDGERLENS_API_URL", "http://api.internal:8000/");
    vi.stubEnv("LEDGERLENS_API_KEY", "server-key");
    const fetcher = vi.fn(
      async () =>
        new Response("event: done\ndata: {}\n\n", {
          status: 200,
          headers: { "content-type": "text/event-stream", "x-thread-id": "chat-1", "set-cookie": "x=1" },
        }),
    );
    vi.stubGlobal("fetch", fetcher);
    const request = new NextRequest("http://dash.test/api/chat?x=1", {
      method: "POST",
      body: JSON.stringify({ message: "hi" }),
      headers: { "content-type": "application/json", "x-api-key": "from-the-browser", cookie: "a=b" },
    });
    const response = await POST(request, context("chat"));

    const [url, init] = fetcher.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe("http://api.internal:8000/chat?x=1");
    const sent = new Headers(init.headers);
    expect(sent.get("x-api-key")).toBe("server-key");
    expect(sent.get("cookie")).toBeNull();
    expect(sent.get("content-type")).toBe("application/json");
    expect(response.headers.get("x-thread-id")).toBe("chat-1");
    expect(response.headers.get("set-cookie")).toBeNull();
    expect(await response.text()).toBe("event: done\ndata: {}\n\n");
  });

  it("refuses a request without a session when login is on", async () => {
    vi.stubEnv("DASHBOARD_PASSWORD", "pw");
    const fetcher = vi.fn();
    vi.stubGlobal("fetch", fetcher);
    const response = await GET(new NextRequest("http://dash.test/api/documents"), context("documents"));
    expect(response.status).toBe(401);
    expect((await response.json()).error.code).toBe("unauthorized");
    expect(fetcher).not.toHaveBeenCalled();
  });

  it("refuses a path that climbs out, and reports an unreachable API", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => Promise.reject(new TypeError("ECONNREFUSED"))));
    const climbing = await GET(new NextRequest("http://dash.test/api/x"), context("..", "admin"));
    expect(climbing.status).toBe(400);
    const down = await GET(new NextRequest("http://dash.test/api/health"), context("health"));
    expect(down.status).toBe(502);
    expect((await down.json()).error.message).toMatch(/could not reach the API/);
  });
});
