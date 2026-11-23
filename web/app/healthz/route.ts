/** Liveness for the container: the Next.js server answers (no session, no API call). */
export function GET(): Response {
  return Response.json({ status: "ok" });
}
