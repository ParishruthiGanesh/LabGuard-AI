"use client";

import { useEffect, useState } from "react";
import { DeltaChart, TrainingCurves } from "../charts";
import { Badge, Card, EmptyState, LiveDot, SimulatedTag } from "../ui";
import { healthTone, jobStateTone, relativeTime, titleCase } from "@/lib/format";
import type { ClaimSnapshot, Job } from "@/lib/types";

interface PairedSummary {
  n_seeds: number;
  mean_delta: number;
  ci_low: number;
  ci_high: number;
  ci_includes_zero: boolean;
  wins_for_b: number;
  losses_for_b: number;
  per_seed_deltas: number[];
}

export function HealthTab({ snapshot }: { snapshot: ClaimSnapshot }) {
  const runs = snapshot.jobs.filter((j) => j.curves.length > 0 || j.health.events.length > 0);
  const live = runs.find((j) => j.state === "running" || j.state === "recovering");
  const [selectedId, setSelectedId] = useState<string | null>(null);

  // Follow whatever is running, until the user picks a run themselves.
  useEffect(() => {
    if (live && selectedId === null) setSelectedId(live.id);
  }, [live, selectedId]);

  const selected = runs.find((j) => j.id === selectedId) ?? live ?? runs[runs.length - 1];

  if (!selected) {
    return (
      <Card>
        <EmptyState
          title="No run to show yet"
          hint="Training curves stream here as soon as the first experiment starts."
        />
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap gap-2">
        {runs.map((job) => {
          const target = typeof job.params.config_name === "string" ? job.params.config_name : null;
          const active = job.id === selected.id;
          return (
            <button
              key={job.id}
              type="button"
              onClick={() => setSelectedId(job.id)}
              className={`flex items-center gap-2 rounded-lg border px-3 py-2 text-left text-xs transition ${
                active
                  ? "border-accent-500 bg-accent-50 text-accent-700"
                  : "border-ink-200 bg-white text-ink-600 hover:border-ink-300"
              }`}
            >
              {(job.state === "running" || job.state === "recovering") && <LiveDot />}
              <span className="font-mono font-medium">{job.action_type}</span>
              {target && <span className="text-ink-400">{target}</span>}
            </button>
          );
        })}
      </div>

      <RunDetail job={selected} />
    </div>
  );
}

function RunDetail({ job }: { job: Job }) {
  const currentRun = typeof job.result.current_run === "string" ? job.result.current_run : null;
  const summary = job.result.paired_summary as Record<string, PairedSummary> | undefined;
  const seeds = (job.result.seeds as number[] | undefined) ?? [];
  const last = job.curves[job.curves.length - 1];

  return (
    <div className="space-y-6">
      <Card
        title={
          <span className="flex flex-wrap items-center gap-2">
            <span className="font-mono">{job.action_type}</span>
            <Badge tone={jobStateTone(job.state)}>{titleCase(job.state)}</Badge>
            <Badge tone={healthTone(job.health.status)}>{titleCase(job.health.status)}</Badge>
          </span>
        }
        subtitle={job.health.summary || job.reason}
      >
        <div className="grid gap-3 sm:grid-cols-4">
          <Metric label="Epochs recorded" value={job.curves.length} />
          <Metric label="Attempts" value={`${job.attempts} / ${job.max_retries + 1}`} />
          <Metric
            label="Peak memory"
            value={job.health.peak_memory_mb ? `${job.health.peak_memory_mb.toFixed(0)} MB` : "-"}
            tag
          />
          <Metric
            label="Mean utilisation"
            value={job.health.mean_gpu_util_pct ? `${job.health.mean_gpu_util_pct.toFixed(0)}%` : "-"}
            tag
          />
        </div>
        {currentRun && (job.state === "running" || job.state === "recovering") && (
          <p className="mt-3 flex items-center gap-2 text-xs text-ink-600">
            <LiveDot />
            Training {currentRun}
            {last ? ` · epoch ${last.epoch}` : ""}
          </p>
        )}
      </Card>

      {job.curves.length > 0 && (
        <Card
          title="Training and validation curves"
          subtitle={
            currentRun ? `Most recent sub-run: ${currentRun}` : "Recorded epochs for the last sub-run of this job."
          }
        >
          <TrainingCurves curves={job.curves} />
        </Card>
      )}

      <Card title="RunMedic timeline" subtitle="What was detected, and what was done about it.">
        {job.health.events.length === 0 ? (
          <EmptyState title="No anomalies detected" hint="This run trained cleanly from start to finish." />
        ) : (
          <ol className="relative space-y-4 border-l border-ink-200 pl-5">
            {job.health.events.map((event) => (
              <li key={event.id} className="relative">
                <span
                  className={`absolute -left-[1.6rem] top-1 h-2.5 w-2.5 rounded-full ring-4 ring-white ${
                    event.status === "critical"
                      ? "bg-bad-500"
                      : event.status === "warning"
                        ? "bg-warn-500"
                        : "bg-info-500"
                  }`}
                  aria-hidden
                />
                <div className="flex flex-wrap items-center gap-2">
                  <Badge tone={healthTone(event.status)}>{titleCase(event.anomaly)}</Badge>
                  {event.epoch != null && <span className="text-[11px] text-ink-500">epoch {event.epoch}</span>}
                  {event.repaired && <Badge tone="info">Repaired</Badge>}
                  {event.requires_approval && <Badge tone="warn">Approval needed</Badge>}
                  <span className="ml-auto text-[11px] text-ink-400">{relativeTime(event.at)}</span>
                </div>
                <p className="mt-1.5 text-sm leading-relaxed break-words text-ink-800">{event.detail}</p>
                {event.action_taken && (
                  <p className="mt-1 text-xs leading-relaxed break-words text-ink-500">{event.action_taken}</p>
                )}
              </li>
            ))}
          </ol>
        )}
      </Card>

      {summary && seeds.length > 0 && (
        <Card
          title="Per-seed comparison"
          subtitle="Candidate minus baseline on every seed, with a 95% interval on the mean difference."
        >
          <div className="grid gap-6 lg:grid-cols-3">
            {Object.entries(summary).map(([metric, stats]) => (
              <div key={metric}>
                <div className="mb-2 flex items-baseline justify-between gap-2">
                  <h3 className="text-sm font-medium text-ink-900">{titleCase(metric)}</h3>
                  <Badge tone={stats.ci_includes_zero ? "warn" : stats.mean_delta > 0 ? "ok" : "bad"}>
                    {stats.ci_includes_zero ? "not separated" : stats.mean_delta > 0 ? "favours candidate" : "favours baseline"}
                  </Badge>
                </div>
                <DeltaChart
                  deltas={stats.per_seed_deltas}
                  seeds={seeds}
                  label={`mean ${stats.mean_delta >= 0 ? "+" : ""}${stats.mean_delta.toFixed(4)}, 95% CI [${stats.ci_low.toFixed(4)}, ${stats.ci_high.toFixed(4)}]`}
                />
                <p className="mt-2 text-[11px] text-ink-500">
                  Candidate wins {stats.wins_for_b} of {stats.n_seeds} seeds.
                </p>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}

function Metric({ label, value, tag }: { label: string; value: string | number; tag?: boolean }) {
  return (
    <div className="rounded-lg bg-ink-50 px-3 py-2.5">
      <p className="flex items-center gap-1.5 text-[10px] font-medium uppercase tracking-wider text-ink-400">
        {label}
        {tag && <SimulatedTag what={label} />}
      </p>
      <p className="tabular mt-1 text-base font-semibold text-ink-900">{value}</p>
    </div>
  );
}
