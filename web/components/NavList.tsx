import Link from "next/link";

import { isActive, NAV_ITEMS } from "@/components/nav";

/** The page links; ``pathname`` marks the current one (``null`` while it is not known). */
export function NavList({ pathname }: { pathname: string | null }) {
  return (
    <ul className="flex flex-row flex-wrap gap-1 md:flex-col">
      {NAV_ITEMS.map((item) => {
        const active = pathname !== null && isActive(item.href, pathname);
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
  );
}
