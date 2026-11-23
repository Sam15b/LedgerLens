export type NavItem = {
  href: string;
  label: string;
  description: string;
  /** Pages that are not built yet are listed but not linked. */
  ready: boolean;
};

/** The dashboard's pages (docs/layers/10-deployment.md, "Dashboard"). */
export const NAV_ITEMS: NavItem[] = [
  {
    href: "/",
    label: "Overview",
    description: "Documents, unpaid total, overdue and flagged invoices, LLM cost",
    ready: true,
  },
  {
    href: "/documents",
    label: "Documents",
    description: "Upload, status and extracted fields next to the document",
    ready: true,
  },
  {
    href: "/chat",
    label: "Chat",
    description: "Ask the agents; tool calls and citations as they stream",
    ready: true,
  },
  {
    href: "/anomalies",
    label: "Anomalies",
    description: "Flagged invoices with reasons; confirm or dismiss",
    ready: true,
  },
  {
    href: "/reminders",
    label: "Reminders",
    description: "Drafted payment reminders waiting for approval",
    ready: true,
  },
  {
    href: "/eval",
    label: "Evaluation",
    description: "Latest evaluation metrics and their trend",
    ready: true,
  },
];

/** Whether ``href`` is the current page (``/`` only matches itself). */
export function isActive(href: string, pathname: string): boolean {
  return href === "/" ? pathname === "/" : pathname === href || pathname.startsWith(`${href}/`);
}

/** Whether ``href`` is a dashboard page that is built (links to others would 404). */
export function pageReady(href: string): boolean {
  return NAV_ITEMS.some((item) => item.ready && isActive(item.href, href));
}
