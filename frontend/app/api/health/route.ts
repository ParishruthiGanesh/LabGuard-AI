/** Health of the backend the dashboard is actually pointed at. */

export const dynamic = "force-dynamic";

export async function GET(): Promise<Response> {
  const base = (process.env.LABGUARD_API_URL || "http://127.0.0.1:8080").replace(/\/+$/, "");
  try {
    const upstream = await fetch(`${base}/health`, { cache: "no-store" });
    return new Response(upstream.body, {
      status: upstream.status,
      headers: { "content-type": upstream.headers.get("content-type") ?? "application/json" },
    });
  } catch (error) {
    return Response.json(
      { status: "unreachable", backend: base, detail: error instanceof Error ? error.message : String(error) },
      { status: 502 },
    );
  }
}
