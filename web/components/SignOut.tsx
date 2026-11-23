import { cookies } from "next/headers";

import { logout } from "@/app/login/actions";
import { loginRequired, SESSION_COOKIE, validSession } from "@/lib/session";

/** "Sign out", for a signed-in person when the dashboard has a login. Reads the cookie, so
 * it renders at request time (inside <Suspense>). */
export async function SignOut() {
  const session = (await cookies()).get(SESSION_COOKIE)?.value;
  if (!loginRequired() || !(await validSession(session))) return null;
  return (
    <form action={logout}>
      <button type="submit" className="text-xs text-muted hover:text-foreground">
        Sign out
      </button>
    </form>
  );
}
