"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import type { ReactNode } from "react";

import { isActive, NAV_ITEMS } from "@/components/nav";

/** The left-hand navigation. ``footer`` holds server-rendered parts (API status). */
export function Sidebar({ footer }: { footer?: ReactNode }) {
  const pathname = usePathname();
  return (
    <aside className="flex w-full shrink-0 flex-col border-b border-border bg-surface md:h-screen md:w-60 md:border-r md:border-b-0">
      <div className="flex items-center gap-2 px-5 py-5">
        <span
          aria-hidden
          className="grid size-8 place-items-center rounded-md bg-accent font-mono text-sm font-bold text-surface"
        >
          LL
        </span>
        <span className="text-lg font-semibold tracking-tight">LedgerLens</span>
      </div>
      <nav aria-label="Main" className="flex-1 px-3">
        <ul className="flex flex-row flex-wrap gap-1 md:flex-col">
          {NAV_ITEMS.map((item) => {
            const active = isActive(item.href, pathname);
            const base = "flex items-center justify-between rounded-md px-3 py-2 text-sm";
            if (!item.ready) {
              return (
                <li key={item.href}>
                  <span
                    aria-disabled="true"
                    title={item.description}
                    className={`${base} cursor-not-allowed text-muted`}
                  >
                    {item.label}
                    <span className="ml-2 rounded bg-border px-1.5 py-0.5 text-[10px] uppercase tracking-wide">
                      soon
                    </span>
                  </span>
                </li>
              );
            }
            return (
              <li key={item.href}>
                <Link
                  href={item.href}
                  title={item.description}
                  aria-current={active ? "page" : undefined}
                  className={`${base} ${
                    active
                      ? "bg-accent-soft font-medium text-accent"
                      : "text-foreground hover:bg-background"
                  }`}
                >
                  {item.label}
                </Link>
              </li>
            );
          })}
        </ul>
      </nav>
      {footer ? <div className="border-t border-border px-5 py-4">{footer}</div> : null}
    </aside>
  );
}
