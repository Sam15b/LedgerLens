# LedgerLens dashboard

Next.js (App Router, TypeScript, Tailwind CSS) front end for the LedgerLens API
(`docs/layers/10-deployment.md`, "Dashboard").

```bash
pnpm install
pnpm dev          # http://localhost:3000 (or `make web` from the repo root)
pnpm lint
pnpm typecheck
pnpm test         # Vitest + Testing Library (__tests__/); pnpm test:watch while editing
pnpm build
```

The API must be running (`make api`, port 8000).

## Configuration (server-side environment)

| Variable | What it does |
|----------|--------------|
| `LEDGERLENS_API_URL` | Where the dashboard's server reaches the API (default `http://localhost:8000`; `http://api:8000` in compose). |
| `LEDGERLENS_API_KEY` | One of the API's `API_KEYS`. Sent by the server only: the browser calls this site's `/api/*`, which forwards with the key. |
| `DASHBOARD_PASSWORD` | Turns the login gate on (`proxy.ts`): every page and `/api/*` needs a signed-in session. Unset = no login (local development). |
| `DASHBOARD_SESSION_SECRET` | Signs the 12-hour session cookie (defaults to the password, so changing it signs everyone out). |
