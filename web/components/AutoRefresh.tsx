"use client";

import { useRouter } from "next/navigation";
import { useEffect } from "react";

export const REFRESH_MS = 3_000;

/**
 * Re-renders the page's server components every few seconds while ``active`` (documents
 * the worker is still processing), so a status reaches ``ready`` without a reload.
 */
export function AutoRefresh({ active, everyMs = REFRESH_MS }: { active: boolean; everyMs?: number }) {
  const router = useRouter();
  useEffect(() => {
    if (!active) return;
    const timer = setInterval(() => router.refresh(), everyMs);
    return () => clearInterval(timer);
  }, [active, everyMs, router]);
  return active ? (
    <p className="text-xs text-muted" aria-live="polite">
      Updating while documents are processed…
    </p>
  ) : null;
}
