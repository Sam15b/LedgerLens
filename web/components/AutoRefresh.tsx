"use client";

import { useRouter } from "next/navigation";
import { useEffect, useTransition } from "react";

export const REFRESH_MS = 3_000;

/**
 * Re-renders the page's server components every few seconds while ``active`` (documents
 * the worker is still processing), so a status reaches ``ready`` without a reload.
 *
 * The next refresh is scheduled only once the last one has landed: a refresh renders the
 * whole layout (the API status check included), which can take longer than the interval,
 * and a new refresh would cancel the one in flight, so nothing would ever update.
 */
export function AutoRefresh({ active, everyMs = REFRESH_MS }: { active: boolean; everyMs?: number }) {
  const router = useRouter();
  const [refreshing, startTransition] = useTransition();
  useEffect(() => {
    if (!active || refreshing) return;
    const timer = setTimeout(() => startTransition(() => router.refresh()), everyMs);
    return () => clearTimeout(timer);
  }, [active, refreshing, everyMs, router]);
  return active ? (
    <p className="text-xs text-muted" aria-live="polite">
      Updating while documents are processed…
    </p>
  ) : null;
}
