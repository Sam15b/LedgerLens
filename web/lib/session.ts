/**
 * The dashboard's login session: a signed, expiring cookie.
 *
 * Login is off unless `DASHBOARD_PASSWORD` is set (local development). With it, a person
 * signs in once with that password; the cookie `ledgerlens_session` then carries
 * `<expiry>.<signature>`, an HMAC-SHA256 of the expiry under `DASHBOARD_SESSION_SECRET`
 * (or, if unset, the password itself, so changing the password signs everyone out). Web
 * Crypto only, so the same code runs in the proxy and in route handlers.
 */

export const SESSION_COOKIE = "ledgerlens_session";

/** The settings read here (`process.env` by default; tests pass their own). */
export type Env = Readonly<Record<string, string | undefined>>;
export const SESSION_HOURS = 12;

export function loginRequired(env: Env = process.env): boolean {
  return Boolean(env.DASHBOARD_PASSWORD);
}

function secretOf(env: Env): string {
  return env.DASHBOARD_SESSION_SECRET || env.DASHBOARD_PASSWORD || "";
}

const encoder = new TextEncoder();

function base64url(bytes: ArrayBuffer): string {
  let text = "";
  for (const byte of new Uint8Array(bytes)) text += String.fromCharCode(byte);
  return btoa(text).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

async function sign(secret: string, message: string): Promise<string> {
  const key = await crypto.subtle.importKey(
    "raw",
    encoder.encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"],
  );
  return base64url(await crypto.subtle.sign("HMAC", key, encoder.encode(message)));
}

/** Equal strings, compared in time that does not depend on where they differ. */
export function sameText(a: string, b: string): boolean {
  const x = encoder.encode(a);
  const y = encoder.encode(b);
  let diff = x.length ^ y.length;
  for (let i = 0; i < Math.max(x.length, y.length); i++) diff |= (x[i] ?? 0) ^ (y[i] ?? 0);
  return diff === 0;
}

/** A session cookie value valid until ``now`` + {@link SESSION_HOURS}. */
export async function createSession(
  env: Env = process.env,
  now: number = Date.now(),
): Promise<{ value: string; expires: Date }> {
  const expires = now + SESSION_HOURS * 3600_000;
  const payload = `v1.${expires}`;
  return { value: `${expires}.${await sign(secretOf(env), payload)}`, expires: new Date(expires) };
}

/** Whether ``value`` is a session this server signed and that has not expired. */
export async function validSession(
  value: string | undefined,
  env: Env = process.env,
  now: number = Date.now(),
): Promise<boolean> {
  if (!value || !secretOf(env)) return false;
  const [expiry, signature, ...rest] = value.split(".");
  if (!expiry || !signature || rest.length > 0 || !/^\d+$/.test(expiry)) return false;
  if (Number(expiry) <= now) return false;
  return sameText(signature, await sign(secretOf(env), `v1.${expiry}`));
}

/** Whether this request may use the dashboard (always, when login is off). */
export async function allowed(cookie: string | undefined, env: Env = process.env): Promise<boolean> {
  return !loginRequired(env) || validSession(cookie, env);
}

/** Where to go after signing in: a path on this site only (never `//evil.example`). */
export function safeNext(next: string | null | undefined): string {
  if (!next || !next.startsWith("/") || next.startsWith("//") || next.startsWith("/\\")) return "/";
  return next;
}
