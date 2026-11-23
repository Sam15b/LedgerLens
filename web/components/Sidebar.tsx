import { type ReactNode, Suspense } from "react";

import { ActiveNavList } from "@/components/ActiveNavList";
import { NavList } from "@/components/NavList";

/** The left-hand navigation. ``footer`` holds server-rendered parts (API status). */
export function Sidebar({ footer }: { footer?: ReactNode }) {
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
        <Suspense fallback={<NavList pathname={null} />}>
          <ActiveNavList />
        </Suspense>
      </nav>
      {footer ? <div className="border-t border-border px-5 py-4">{footer}</div> : null}
    </aside>
  );
}
