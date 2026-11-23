import { describeRecord } from "@/lib/fields";
import { humanize } from "@/lib/format";

/** The fields extracted from a document, as stored, with its line items as tables. */
export function ExtractedFields({
  record,
  recordType,
  recordId,
}: {
  record: Record<string, unknown>;
  recordType?: string | null;
  recordId?: string | null;
}) {
  const { fields, tables } = describeRecord(record);
  return (
    <div className="flex flex-col gap-4">
      <p className="text-xs text-muted">
        {recordType ? humanize(recordType) : "Record"}
        {recordId ? (
          <>
            {" "}
            <span className="font-mono text-foreground">{recordId}</span>
          </>
        ) : null}
      </p>
      <dl className="grid grid-cols-[minmax(8rem,auto)_1fr] gap-x-4 gap-y-1.5 text-sm">
        {fields.map((field) => (
          <div key={field.key} className="contents">
            <dt className="text-muted">{field.label}</dt>
            <dd className={`min-w-0 break-words ${field.empty ? "text-muted" : "font-medium"}`}>
              {field.value}
            </dd>
          </div>
        ))}
      </dl>
      {tables.map((table) => (
        <div key={table.key}>
          <h3 className="mb-1 text-xs font-medium uppercase tracking-wide text-muted">
            {table.label} ({table.rows.length})
          </h3>
          <div className="overflow-x-auto rounded-md border border-border">
            <table className="w-full text-left text-xs" aria-label={table.label}>
              <thead className="bg-background text-muted">
                <tr>
                  {table.columns.map((column) => (
                    <th
                      key={column.key}
                      scope="col"
                      className={`px-2 py-1.5 font-medium ${column.numeric ? "text-right" : ""}`}
                    >
                      {column.label}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {table.rows.map((row, i) => (
                  <tr key={i} className="border-t border-border">
                    {row.map((cell, j) => (
                      <td
                        key={table.columns[j]?.key ?? j}
                        className={`px-2 py-1.5 ${
                          table.columns[j]?.numeric ? "text-right tabular-nums" : ""
                        }`}
                      >
                        {cell}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ))}
    </div>
  );
}
