import { type NextRequest, NextResponse } from "next/server";

import { allowed, SESSION_COOKIE } from "@/lib/session";

/**
 * The login gate (Next 16 renamed `middleware` to `proxy`). With `DASHBOARD_PASSWORD` set,
 * a page request without a valid session goes to /login. `/api/*` is left to its route
 * handler, which checks the same session itself: the proxy would buffer upload bodies.
 */
export async function proxy(request: NextRequest) {
  if (await allowed(request.cookies.get(SESSION_COOKIE)?.value)) return NextResponse.next();
  const login = new URL("/login", request.url);
  // Behind CloudFront the load balancer is reached over plain HTTP, so request.url says
  // http: keep the browser on the scheme it actually used (CloudFront tells us).
  if (request.headers.get("cloudfront-forwarded-proto") === "https") login.protocol = "https:";
  const next = request.nextUrl.pathname + request.nextUrl.search;
  if (next !== "/") login.searchParams.set("next", next);
  return NextResponse.redirect(login);
}

export const config = {
  // everything but the login page, the API forwarder, the liveness probe and static files
  matcher: ["/((?!login|api/|healthz|_next/static|_next/image|favicon.ico).*)"],
};
