/**
 * Display formatting shared by server and client components. Everything is formatted in
 * one locale and in UTC, so the server's render and the browser's hydration agree.
 */

const LOCALE = "en-US";

/** `1234.5` / `"1234.50"` in `currency` -> `$1,234.50`; an unknown code -> `1,234.50 XYZ`. */
export function formatMoney(amount: string | number, currency: string): string {
  const value = typeof amount === "number" ? amount : Number(amount);
  if (!Number.isFinite(value)) return `${amount} ${currency}`;
  try {
    return new Intl.NumberFormat(LOCALE, { style: "currency", currency }).format(value);
  } catch {
    const plain = new Intl.NumberFormat(LOCALE, {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(value);
    return `${plain} ${currency}`;
  }
}

/** LLM spend: cents-level amounts keep four decimals (`$0.0042`), larger ones two. */
export function formatUsd(value: number): string {
  const digits = value !== 0 && Math.abs(value) < 1 ? 4 : 2;
  return new Intl.NumberFormat(LOCALE, {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(value);
}

export function formatCount(value: number): string {
  return new Intl.NumberFormat(LOCALE).format(value);
}

const DAY = new Intl.DateTimeFormat(LOCALE, {
  year: "numeric",
  month: "short",
  day: "numeric",
  timeZone: "UTC",
});
const SHORT_DAY = new Intl.DateTimeFormat(LOCALE, {
  month: "short",
  day: "numeric",
  timeZone: "UTC",
});
const DATETIME = new Intl.DateTimeFormat(LOCALE, {
  year: "numeric",
  month: "short",
  day: "numeric",
  hour: "2-digit",
  minute: "2-digit",
  hourCycle: "h23",
  timeZone: "UTC",
});

function parse(value: string): Date | null {
  // A bare `YYYY-MM-DD` is a calendar day: read it as UTC midnight, not local time.
  const date = new Date(/^\d{4}-\d{2}-\d{2}$/.test(value) ? `${value}T00:00:00Z` : value);
  return Number.isNaN(date.getTime()) ? null : date;
}

/** `2026-11-21` -> `Nov 21, 2026`. */
export function formatDate(value: string): string {
  const date = parse(value);
  return date ? DAY.format(date) : value;
}

/** `2026-11-21` -> `Nov 21`. */
export function formatShortDate(value: string): string {
  const date = parse(value);
  return date ? SHORT_DAY.format(date) : value;
}

/** An ISO timestamp -> `Nov 21, 2026, 14:03 UTC`. */
export function formatDateTime(value: string): string {
  const date = parse(value);
  return date ? `${DATETIME.format(date)} UTC` : value;
}

/** `invoice_number` -> `Invoice number`; `po_number` -> `PO number`. */
export function humanize(key: string): string {
  const words = key.split("_").filter(Boolean);
  const out = words.map((word) => (ACRONYMS.has(word) ? word.toUpperCase() : word));
  const [first = "", ...rest] = out;
  return [first.charAt(0).toUpperCase() + first.slice(1), ...rest].join(" ");
}

const ACRONYMS = new Set(["po", "id", "url", "ocr", "llm", "iban", "vat", "sku"]);

/** `1536` -> `1.5 KB`. */
export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB"];
  let value = bytes / 1024;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(value < 10 ? 1 : 0)} ${units[unit]}`;
}
