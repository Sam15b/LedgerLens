export const THEME_KEY = "ledgerlens-theme";

/**
 * Runs before the first paint (in <head>), so the page never flashes the wrong theme:
 * the saved choice, else the operating system's preference.
 */
export const THEME_SCRIPT = `(() => {
  try {
    const saved = localStorage.getItem("${THEME_KEY}");
    const dark = saved ? saved === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    document.documentElement.dataset.theme = dark ? "dark" : "light";
  } catch {
    document.documentElement.dataset.theme = "light";
  }
})();`;
