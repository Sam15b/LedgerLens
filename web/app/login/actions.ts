"use server";

import { cookies } from "next/headers";
import { redirect } from "next/navigation";

import { createSession, loginRequired, safeNext, SESSION_COOKIE, sameText } from "@/lib/session";

const FAILED_DELAY_MS = 750; // slows down guessing; the password should be long anyway

/** Signs in with the dashboard password, then goes back where the person was headed. */
export async function login(formData: FormData): Promise<void> {
  const next = safeNext(String(formData.get("next") ?? "/"));
  if (!loginRequired()) redirect(next);
  const password = String(formData.get("password") ?? "");
  if (!sameText(password, process.env.DASHBOARD_PASSWORD ?? "")) {
    await new Promise((resolve) => setTimeout(resolve, FAILED_DELAY_MS));
    redirect(`/login?${new URLSearchParams({ error: "1", next })}`);
  }
  const session = await createSession();
  (await cookies()).set(SESSION_COOKIE, session.value, {
    httpOnly: true,
    secure: process.env.NODE_ENV === "production",
    sameSite: "lax",
    path: "/",
    expires: session.expires,
  });
  redirect(next);
}

export async function logout(): Promise<void> {
  (await cookies()).delete(SESSION_COOKIE);
  redirect("/login");
}
