import { connection } from "next/server";

import { logout } from "@/app/login/actions";
import { loginRequired } from "@/lib/session";

/** "Sign out", when the dashboard has a login. Read at request time (inside <Suspense>):
 * the password is runtime configuration, not something to bake into a prerendered page. */
export async function SignOut() {
  await connection();
  if (!loginRequired()) return null;
  return (
    <form action={logout}>
      <button type="submit" className="text-xs text-muted hover:text-foreground">
        Sign out
      </button>
    </form>
  );
}
