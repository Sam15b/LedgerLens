"use client";

import { THEME_KEY } from "@/components/theme";

/** Switches between light and dark and remembers the choice. Both labels are rendered and
 * CSS shows the right one, so the button needs no state and never mismatches on hydration. */
export function ThemeToggle() {
  function toggle() {
    const root = document.documentElement;
    const next = root.dataset.theme === "dark" ? "light" : "dark";
    root.dataset.theme = next;
    try {
      localStorage.setItem(THEME_KEY, next);
    } catch {
      // private mode: the choice lasts for this page only
    }
  }

  return (
    <button
      type="button"
      onClick={toggle}
      aria-label="Switch between light and dark theme"
      className="inline-flex items-center gap-2 rounded-md border border-border px-2.5 py-1.5 text-xs text-muted hover:text-foreground"
    >
      <span className="dark:hidden">Dark theme</span>
      <span className="hidden dark:inline">Light theme</span>
    </button>
  );
}
