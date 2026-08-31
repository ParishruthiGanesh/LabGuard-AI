/**
 * Runtime proxy to the LabGuard API.
 *
 * This deliberately is *not* a `rewrites()` entry in next.config. Next resolves
 * rewrites at build time and freezes the destination into
 * `.next/routes-manifest.json`, so a URL that is only known at deploy time —
 * which is exactly the case on Cloud Run — can never be picked up. A route
 * handler reads the environment on every request instead.
 */

export const dynamic = "force-dynamic";

const backendBase = () => (process.env.LABGUARD_API_URL || "http://127.0.0.1:8080").replace(/\/+$/, "");

/** Response headers worth passing back; the rest are hop-by-hop or recomputed. */
const PASS_THROUGH = ["content-type", "content-disposition", "cache-control"];

async function forward(request: Request, path: string[]): Promise<Response> {
  const incoming = new URL(request.url);
  const target = new URL(`${backendBase()}/api/${path.map(encodeURIComponent).join("/")}`);
  incoming.searchParams.forEach((value, key) => target.searchParams.set(key, value));

  const init: RequestInit = {
    method: request.method,
    headers: { accept: request.headers.get("accept") ?? "application/json" },
    cache: "no-store",
  };
  if (request.method !== "GET" && request.method !== "HEAD") {
    init.body = await request.text();
    (init.headers as Record<string, string>)["content-type"] =
      request.headers.get("content-type") ?? "application/json";
  }

  let upstream: Response;
  try {
    upstream = await fetch(target, init);
  } catch (error) {
    // A transport failure is the operator's problem, not the user's mistake,
    // so say which URL failed rather than returning a bare 500.
    return Response.json(
      {
        detail:
          `The LabGuard API at ${backendBase()} did not respond ` +
          `(${error instanceof Error ? error.message : String(error)}).`,
      },
      { status: 502 },
    );
  }

  const headers = new Headers();
  for (const name of PASS_THROUGH) {
    const value = upstream.headers.get(name);
    if (value) headers.set(name, value);
  }
  return new Response(upstream.body, { status: upstream.status, headers });
}

type Context = { params: Promise<{ path: string[] }> };

export async function GET(request: Request, context: Context) {
  return forward(request, (await context.params).path);
}

export async function POST(request: Request, context: Context) {
  return forward(request, (await context.params).path);
}
