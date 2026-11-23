import react from "@vitejs/plugin-react";
import tsconfigPaths from "vite-tsconfig-paths";
import { defineConfig } from "vitest/config";

// Unit and component tests (`pnpm test`). Async server components are not rendered here:
// their markup lives in synchronous components (components/*) that the tests render.
export default defineConfig({
  plugins: [tsconfigPaths(), react()],
  test: {
    environment: "jsdom",
    include: ["__tests__/**/*.test.{ts,tsx}"],
    setupFiles: ["__tests__/setup.ts"],
    env: { TZ: "America/Los_Angeles" }, // dates must not depend on the machine's zone
  },
});
