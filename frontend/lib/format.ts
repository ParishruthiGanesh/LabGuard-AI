import type { ClaimState, EvidenceStance, HealthStatus, JobState, SubclaimStatus, VerdictStatus } from "./types";

export const titleCase = (value: string) =>
  value.replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());

export const pct = (value: number, digits = 1) => `${(value * 100).toFixed(digits)}%`;

export const signed = (value: number, digits = 4) =>
  `${value >= 0 ? "+" : ""}${value.toFixed(digits)}`;

export const units = (value: number) => `${value.toFixed(2)} units`;

export function relativeTime(iso: string | null): string {
  if (!iso) return "-";
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return "-";
  const seconds = Math.round((Date.now() - then) / 1000);
  if (seconds < 5) return "just now";
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  return `${Math.round(minutes / 60)}h ago`;
}

export type Tone = "neutral" | "ok" | "warn" | "bad" | "info" | "accent";

const TONE_CLASS: Record<Tone, string> = {
  neutral: "bg-ink-100 text-ink-700 ring-ink-200",
  ok: "bg-ok-50 text-ok-700 ring-ok-500/25",
  warn: "bg-warn-50 text-warn-700 ring-warn-500/25",
  bad: "bg-bad-50 text-bad-700 ring-bad-500/25",
  info: "bg-info-50 text-info-700 ring-info-500/25",
  accent: "bg-accent-50 text-accent-700 ring-accent-500/25",
};

export const toneClass = (tone: Tone) => TONE_CLASS[tone];

export const claimStateTone = (state: ClaimState): Tone =>
  state === "verdict"
    ? "accent"
    : state.startsWith("halted")
      ? "warn"
      : state === "awaiting_approval"
        ? "warn"
        : state === "executing"
          ? "info"
          : "neutral";

const JOB_TONE: Record<JobState, Tone> = {
  planned: "neutral",
  awaiting_approval: "warn",
  queued: "neutral",
  running: "info",
  recovering: "warn",
  completed: "ok",
  failed: "bad",
  blocked_loop: "bad",
  rejected: "neutral",
};

export const jobStateTone = (state: JobState): Tone => JOB_TONE[state];

const HEALTH_TONE: Record<HealthStatus, Tone> = {
  healthy: "ok",
  warning: "warn",
  critical: "bad",
  recovered: "info",
  unknown: "neutral",
};

export const healthTone = (status: HealthStatus): Tone => HEALTH_TONE[status];

const SUBCLAIM_TONE: Record<SubclaimStatus, Tone> = {
  untested: "neutral",
  testing: "info",
  supported: "ok",
  contradicted: "bad",
  inconclusive: "warn",
};

export const subclaimTone = (status: SubclaimStatus): Tone => SUBCLAIM_TONE[status];

const STANCE_TONE: Record<EvidenceStance, Tone> = {
  supports: "ok",
  contradicts: "bad",
  neutral: "neutral",
};

export const stanceTone = (stance: EvidenceStance): Tone => STANCE_TONE[stance];

const VERDICT_TONE: Record<VerdictStatus, Tone> = {
  supported: "ok",
  provisionally_supported: "ok",
  fragile: "warn",
  inconclusive: "warn",
  not_sufficiently_supported: "bad",
  refuted: "bad",
};

export const verdictTone = (status: VerdictStatus): Tone => VERDICT_TONE[status];

/** Score bands are fixed, so the same number always reads the same way. */
export const scoreTone = (score: number): Tone =>
  score >= 75 ? "ok" : score >= 50 ? "warn" : "bad";

export const scoreColor = (score: number) =>
  score >= 75 ? "var(--color-ok-500)" : score >= 50 ? "var(--color-warn-500)" : "var(--color-bad-500)";

export const AGENT_LABEL: Record<string, string> = {
  claim_analyst: "Claim Analyst",
  scientific_skeptic: "Scientific Skeptic",
  experiment_planner: "Experiment Planner",
  run_manager: "Run Manager",
  run_medic: "RunMedic",
  evidence_auditor: "Evidence Auditor",
  verdict_agent: "Verdict Agent",
  orchestrator: "Orchestrator",
};
