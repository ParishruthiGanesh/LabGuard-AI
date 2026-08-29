"use client";

import { useState } from "react";
import { Badge, Button, Card, EmptyState, LiveDot } from "../ui";
import { api } from "@/lib/api";
import { healthTone, jobStateTone, relativeTime, titleCase } from "@/lib/format";
import type { ClaimSnapshot, Job, JobState } from "@/lib/types";

const LANES: { state: JobState | "recovered"; label: string; hint: string }[] = [
  { state: "planned", label: "Planned", hint: "chosen but not yet published" },
  { state: "awaiting_approval", label: "Awaiting approval", hint: "held for a human decision" },
  { state: "queued", label: "Queued", hint: "published to the job bus" },
  { state: "running", label: "Running", hint: "executing on the worker" },
  { state: "recovered", label: "Recovered", hint: "repaired and completed" },
  { state: "failed", label: "Failed", hint: "ended without a result" },
  { state: "blocked_loop", label: "Loop blocked", hint: "stopped to avoid burning budget" },
  { state: "completed", label: "Completed", hint: "produced evidence" },
];

function laneOf(job: Job): JobState | "recovered" {
  if (job.state === "completed" && job.recovery_actions.some((r) => r.startsWith("recovery:"))) return "recovered";
  if (job.state === "recovering") return "running";
  return job.state;
}

export function QueueTab({ snapshot, onChange }: { snapshot: ClaimSnapshot; onChange: () => void }) {
  const { claim, jobs } = snapshot;
  const [busy, setBusy] = useState<string | null>(null);

  async function decideJob(jobId: string, approved: boolean) {
    setBusy(jobId);
    try {
      await api.decideJob(claim.id, jobId, approved);
      onChange();
    } finally {
      setBusy(null);
    }
  }

  if (jobs.length === 0) {
    return (
      <Card>
        <EmptyState title="The queue is empty" hint="Jobs appear as soon as the planner has produced a round." />
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      <div className="grid gap-3 sm:grid-cols-4 xl:grid-cols-8">
        {LANES.map((lane) => {
          const count = jobs.filter((j) => laneOf(j) === lane.state).length;
          return (
            <div key={lane.state} className="card px-3 py-2.5">
              <p className="text-[10px] font-medium uppercase tracking-wider text-ink-400">{lane.label}</p>
              <p className="tabular mt-1 text-lg font-semibold text-ink-900">{count}</p>
              <p className="mt-0.5 text-[10px] leading-tight text-ink-400">{lane.hint}</p>
            </div>
          );
        })}
      </div>

      <Card title="Experiment queue" subtitle="Every job, in the order it was created.">
        <ul className="divide-y divide-ink-100">
          {jobs.map((job) => {
            const recovered = job.recovery_actions.filter((r) => r.startsWith("recovery:"));
            const target = typeof job.params.config_name === "string" ? job.params.config_name : null;
            return (
              <li key={job.id} className="py-3.5 first:pt-0 last:pb-0">
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div className="flex min-w-0 flex-1 flex-wrap items-center gap-2">
                    {(job.state === "running" || job.state === "recovering") && <LiveDot />}
                    <span className="font-mono text-sm font-medium text-ink-900">{job.action_type}</span>
                    {target && <span className="text-xs text-ink-500">· {target}</span>}
                    <Badge tone={jobStateTone(job.state)}>{titleCase(job.state)}</Badge>
                    {job.health.status !== "unknown" && (
                      <Badge tone={healthTone(job.health.status)}>{titleCase(job.health.status)}</Badge>
                    )}
                  </div>
                  <div className="flex shrink-0 items-center gap-4 text-xs text-ink-500">
                    <span className="tabular">
                      attempt {job.attempts}/{job.max_retries + 1}
                    </span>
                    <span className="tabular">{job.actual_cost_units.toFixed(2)}u</span>
                    <span>{relativeTime(job.finished_at ?? job.started_at ?? job.created_at)}</span>
                  </div>
                </div>

                <p className="mt-1.5 text-xs leading-relaxed text-ink-600">{job.reason}</p>

                {recovered.length > 0 && (
                  <p className="mt-1.5 text-xs text-info-700">
                    Repaired with {recovered.map((r) => r.replace("recovery:", "")).join(", ")}
                  </p>
                )}
                {job.error && (
                  <p className="mt-1.5 rounded bg-bad-50 px-3 py-2 font-mono text-[11px] leading-relaxed break-words text-bad-700">
                    {job.error}
                  </p>
                )}

                {job.state === "awaiting_approval" && claim.state !== "halted_approval" && (
                  <div className="mt-2.5 flex gap-2">
                    <Button
                      variant="secondary"
                      onClick={() => decideJob(job.id, true)}
                      busy={busy === job.id}
                      disabled={busy !== null}
                    >
                      Approve this run
                    </Button>
                    <Button
                      variant="ghost"
                      onClick={() => decideJob(job.id, false)}
                      disabled={busy !== null}
                    >
                      Decline
                    </Button>
                  </div>
                )}
              </li>
            );
          })}
        </ul>
      </Card>
    </div>
  );
}
