import type { AppConfig, Claim, ClaimSnapshot } from "./types";

/** All backend calls go through the Next.js rewrite, so they stay same-origin. */
const BASE = "/api/backend";

class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${BASE}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
      cache: "no-store",
    });
  } catch {
    throw new ApiError("Could not reach the LabGuard API. Is the backend running?", 0);
  }
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (typeof body.detail === "string") detail = body.detail;
      else if (Array.isArray(body.detail)) {
        detail = body.detail
          .map((e) => (typeof e === "object" && e && "msg" in e ? String((e as { msg: unknown }).msg) : String(e)))
          .join("; ");
      }
    } catch {
      /* keep the status line */
    }
    throw new ApiError(detail, response.status);
  }
  return (await response.json()) as T;
}

export const api = {
  config: () => request<AppConfig>("/config"),
  listClaims: () => request<Claim[]>("/claims"),
  snapshot: (claimId: string) => request<ClaimSnapshot>(`/claims/${claimId}`),

  createClaim: (body: Record<string, unknown>) =>
    request<Claim>("/claims", { method: "POST", body: JSON.stringify(body) }),

  decidePlan: (claimId: string, planId: string, approved: boolean, decidedBy = "researcher") =>
    request<Claim>(`/claims/${claimId}/plans/${planId}/decision`, {
      method: "POST",
      body: JSON.stringify({ approved, decided_by: decidedBy }),
    }),

  decideJob: (claimId: string, jobId: string, approved: boolean, decidedBy = "researcher") =>
    request<{ job_id: string; state: string }>(`/claims/${claimId}/jobs/${jobId}/decision`, {
      method: "POST",
      body: JSON.stringify({ approved, decided_by: decidedBy }),
    }),

  reportUrl: (claimId: string) => `${BASE}/claims/${claimId}/report`,
};

export { ApiError };
