/** Why a section has no data: the API's message, in place of the content. */
export function ErrorPanel({ title, message }: { title: string; message: string }) {
  return (
    <div
      role="alert"
      className="rounded-lg border border-danger/40 bg-danger/5 px-4 py-3 text-sm text-foreground"
    >
      <p className="font-medium text-danger">{title}</p>
      <p className="mt-1 text-muted">{message}</p>
    </div>
  );
}

/** A grey block standing in for content that is still loading. */
export function Placeholder({ className = "" }: { className?: string }) {
  return <div aria-hidden className={`animate-pulse rounded-lg bg-border/60 ${className}`} />;
}
