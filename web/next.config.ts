import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // .next/standalone: server.js plus only the node_modules it needs (infra/docker/web.Dockerfile)
  output: "standalone",
  cacheComponents: true,
  partialPrefetching: true,
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
};

export default nextConfig;
