/**
 * Plain JavaScript, deliberately.
 *
 * A `next.config.ts` needs the `typescript` package at runtime, and the
 * production image installs with `--omit=dev`. Next then tries to yarn-install
 * TypeScript on container start, which never finishes before Cloud Run's
 * health check and leaves the service restarting forever.
 *
 * @type {import('next').NextConfig}
 */
const config = {
  reactStrictMode: true,
  // Pin the trace root to this package so a lockfile elsewhere in the repo
  // cannot make Next infer the wrong workspace root.
  outputFileTracingRoot: import.meta.dirname,
  // The dashboard talks to the FastAPI backend through this rewrite, so the
  // browser only ever sees same-origin requests and no CORS setup is needed
  // for the common deployment.
  async rewrites() {
    // `||`, not `??`: an env var that is present but empty must fall back too.
    // An empty value makes the destination relative, so the dashboard proxies
    // to itself and every API call 404s.
    const backend = process.env.LABGUARD_API_URL || "http://127.0.0.1:8080";
    // Logged once at startup so a misconfigured deployment is diagnosable from
    // the service logs rather than from a bare 404 in the browser.
    console.log(`[labguard] proxying /api/backend/* to ${backend}`);
    return [
      { source: "/api/backend/:path*", destination: `${backend}/api/:path*` },
      { source: "/api/health", destination: `${backend}/health` },
    ];
  },
};

export default config;
