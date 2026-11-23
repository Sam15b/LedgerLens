import type { Metadata } from "next";
import { Suspense } from "react";

import { login } from "@/app/login/actions";
import { safeNext } from "@/lib/session";

export const metadata: Metadata = { title: "Sign in" };

export default function LoginPage({ searchParams }: PageProps<"/login">) {
  return (
    <section className="mx-auto mt-12 max-w-sm">
      <h1 className="text-2xl font-semibold tracking-tight">Sign in</h1>
      <p className="mt-1 text-sm text-muted">Enter the dashboard password to continue.</p>
      <Suspense>
        <LoginForm searchParams={searchParams} />
      </Suspense>
    </section>
  );
}

async function LoginForm({ searchParams }: { searchParams: PageProps<"/login">["searchParams"] }) {
  const params = await searchParams;
  const next = safeNext(typeof params.next === "string" ? params.next : "/");
  const failed = params.error === "1";
  return (
    <form action={login} className="mt-6 flex flex-col gap-3">
      <input type="hidden" name="next" value={next} />
      <label className="flex flex-col gap-1 text-sm">
        Password
        <input
          type="password"
          name="password"
          required
          autoFocus
          autoComplete="current-password"
          className="rounded-md border border-border bg-surface px-3 py-2 outline-none focus:border-accent"
        />
      </label>
      {failed ? (
        <p role="alert" className="text-sm text-danger">
          That password is not right.
        </p>
      ) : null}
      <button type="submit" className="rounded-md bg-accent px-3 py-2 text-sm font-medium text-surface">
        Sign in
      </button>
    </form>
  );
}
