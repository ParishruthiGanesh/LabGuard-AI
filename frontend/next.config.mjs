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
  // The proxy to the backend is a route handler (app/api/backend/[...path]),
  // not a rewrite. Next resolves rewrites at build time and freezes the
  // destination into the routes manifest, which cannot work for a URL that is
  // only known at deploy time.
};

export default config;
