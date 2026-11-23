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
