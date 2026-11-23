/**
 * Turns an extracted record (`GET /documents/{id}` -> `record`, any of invoice, receipt,
 * purchase order, payment or contract) into what the viewer lists: labelled values, and a
 * table for each list of objects (line items, clauses, allocations, ...).
 */
import { formatDate, formatMoney, humanize } from "@/lib/format";

export type Field = { key: string; label: string; value: string; empty: boolean };
export type FieldTable = {
  key: string;
  label: string;
  columns: { key: string; label: string; numeric: boolean }[];
  rows: string[][];
};
export type FieldView = { fields: Field[]; tables: FieldTable[] };

const MONEY = /^(total|subtotal|tax|amount|unit_price|price|balance|value|total_value)$|_(amount|total)$/;
const DATE = /^\d{4}-\d{2}-\d{2}$/;
const NUMBER = /^-?\d+(\.\d+)?$/;
export const EMPTY = "—";

/** One value as text: money in ``currency``, ISO days as dates, lists joined. */
export function formatValue(key: string, value: unknown, currency?: string): string {
  if (value === null || value === undefined || value === "") return EMPTY;
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value === "number" || typeof value === "string") {
    const text = String(value);
    if (currency && MONEY.test(key) && NUMBER.test(text)) return formatMoney(text, currency);
    if (typeof value === "string" && DATE.test(value)) return formatDate(value);
    if (NUMBER.test(text) && text.includes(".")) return text.replace(/\.?0+$/, ""); // 1.0000 -> 1
    return text;
  }
  if (Array.isArray(value)) {
    return value.length ? value.map((v) => formatValue(key, v, currency)).join(", ") : EMPTY;
  }
  return Object.entries(value as Record<string, unknown>)
    .filter(([, v]) => v !== null && v !== undefined && v !== "")
    .map(([k, v]) => `${humanize(k)}: ${formatValue(k, v, currency)}`)
    .join(" · ") || EMPTY;
}

function isObjectList(value: unknown): value is Record<string, unknown>[] {
  return (
    Array.isArray(value) &&
    value.length > 0 &&
    value.every((v) => typeof v === "object" && v !== null && !Array.isArray(v))
  );
}

export function describeRecord(record: Record<string, unknown>): FieldView {
  const currency = typeof record.currency === "string" ? record.currency : undefined;
  const fields: Field[] = [];
  const tables: FieldTable[] = [];
  for (const [key, value] of Object.entries(record)) {
    if (isObjectList(value)) {
      const keys = [...new Set(value.flatMap((row) => Object.keys(row)))];
      tables.push({
        key,
        label: humanize(key),
        columns: keys.map((k) => ({
          key: k,
          label: humanize(k),
          numeric: value.every((row) => row[k] == null || NUMBER.test(String(row[k]))),
        })),
        rows: value.map((row) => keys.map((k) => formatValue(k, row[k], currency))),
      });
      continue;
    }
    const text = formatValue(key, value, currency);
    fields.push({ key, label: humanize(key), value: text, empty: text === EMPTY });
  }
  return { fields, tables };
}
