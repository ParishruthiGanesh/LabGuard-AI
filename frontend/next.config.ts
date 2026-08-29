import type { NextConfig } from "next";

const config: NextConfig = {
  reactStrictMode: true,
  // Pin the trace root to this package so a lockfile elsewhere in the repo
  // cannot make Next infer the wrong workspace root.
  outputFileTracingRoot: import.meta.dirname,
  // The dashboard talks to the FastAPI backend through this rewrite, so the
  // browser only ever sees same-origin requests and no CORS setup is needed
  // for the common deployment.
  async rewrites() {
    const backend = process.env.LABGUARD_API_URL ?? "http://127.0.0.1:8080";
    return [
      { source: "/api/backend/:path*", destination: `${backend}/api/:path*` },
      { source: "/api/health", destination: `${backend}/health` },
    ];
  },
};

export default config;
