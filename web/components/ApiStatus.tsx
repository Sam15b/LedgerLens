import { getHealth } from "@/lib/api";

function Dot({ className }: { className: string }) {
  return <span aria-hidden className={`inline-block size-2 rounded-full ${className}`} />;
}

/** Whether the API answers, its version, and which dependencies are down. Fetched on
 * every request: render it inside <Suspense>. */
export async function ApiStatus() {
  const health = await getHealth();
  if (health === null) {
    return (
      <p className="flex items-center gap-2 text-xs text-danger">
        <Dot className="bg-danger" />
        API unreachable
      </p>
    );
  }
  const down = Object.entries(health.dependencies)
    .filter(([, dependency]) => dependency.status !== "ok")
    .map(([name]) => name);
  const ok = health.status === "ok";
  return (
    <div className="text-xs">
      <p className={`flex items-center gap-2 ${ok ? "text-success" : "text-danger"}`}>
        <Dot className={ok ? "bg-success" : "bg-danger"} />
        API {ok ? "online" : "degraded"} · v{health.version}
      </p>
      {down.length > 0 ? <p className="mt-1 text-muted">Down: {down.join(", ")}</p> : null}
    </div>
  );
}

export function ApiStatusPending() {
  return (
    <p className="flex items-center gap-2 text-xs text-muted">
      <Dot className="bg-border" />
      Checking the API…
    </p>
  );
}
