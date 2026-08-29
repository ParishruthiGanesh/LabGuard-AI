"use client";

import { ScoreBar } from "../charts";
import { Badge, Card, EmptyState, LiveDot, Stat } from "../ui";
import {
  AGENT_LABEL,
  claimStateTone,
  healthTone,
  scoreColor,
  scoreTone,
  titleCase,
  units,
  verdictTone,
} from "@/lib/format";
import type { ClaimSnapshot } from "@/lib/types";

export function Overview({ snapshot, onOpenTab }: { snapshot: ClaimSnapshot; onOpenTab: (tab: string) => void }) {
  const { claim, verdict, score, jobs, evidence, loopholes, subclaims } = snapshot;
  const budgetPct = claim.budget.total_units
    ? Math.min(100, (claim.budget.consumed_units / claim.budget.total_units) * 100)
    : 0;
  const running = jobs.filter((j) => j.state === "running" || j.state === "recovering");
  const completed = jobs.filter((j) => j.state === "completed").length;
  const incidents = jobs.flatMap((j) => j.health.events);
  const contradicting = evidence.filter((e) => e.stance === "contradicts").length;
  const supporting = evidence.filter((e) => e.stance === "supports").length;
  const openLoopholes = loopholes.filter((h) => h.status === "open" || h.status === "investigating").length;

  return (
    <div className="space-y-6">
      <Card>
        <p className="text-[11px] font-medium uppercase tracking-wider text-ink-400">Claim under verification</p>
        <p className="mt-2 text-lg font-semibold leading-snug tracking-tight text-ink-900">{claim.text}</p>
        <div className="mt-4 flex flex-wrap items-center gap-2">
          <Badge tone={claimStateTone(claim.state)}>{titleCase(claim.state)}</Badge>
          <Badge tone="neutral">Autonomy: {titleCase(claim.autonomy_mode)}</Badge>
          <Badge tone="neutral">Round {claim.planning_round + 1}</Badge>
          {verdict && <Badge tone={verdictTone(verdict.status)}>{titleCase(verdict.status)}</Badge>}
        </div>
        {claim.halt_reason && (
          <p className="mt-4 rounded-lg border border-warn-500/25 bg-warn-50 px-4 py-3 text-sm leading-relaxed text-warn-700">
            {claim.halt_reason}
          </p>
        )}
      </Card>

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <Stat
          label="Overall claim confidence"
          value={score ? score.overall : "-"}
          tone={score ? scoreTone(score.overall) : undefined}
          hint={score?.calculation}
        />
        <Stat
          label="Budget consumed"
          value={units(claim.budget.consumed_units)}
          hint={`of ${units(claim.budget.total_units)} · approval above ${units(claim.budget.approval_threshold_units)}`}
        />
        <Stat
          label="Experiments completed"
          value={`${completed} / ${jobs.length}`}
          hint={running.length ? `${running.length} running now` : "nothing running"}
        />
        <Stat
          label="Evidence recorded"
          value={evidence.length}
          hint={`${contradicting} contradicting · ${supporting} supporting`}
        />
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        <Card
          className="lg:col-span-2"
          title="Reliability dimensions"
          subtitle="Each score is a weighted pass rate over named checks. Open the Final report to see every check."
          action={
            <button
              type="button"
              onClick={() => onOpenTab("report")}
              className="shrink-0 text-xs font-medium text-accent-600 hover:text-accent-700"
            >
              See the workings →
            </button>
          }
        >
          {score ? (
            <ul className="space-y-3.5">
              {score.dimensions.map((dimension) => (
                <li key={dimension.dimension}>
                  <div className="flex items-baseline justify-between gap-3">
                    <span className="text-sm font-medium text-ink-800">{titleCase(dimension.dimension)}</span>
                    <span className="tabular text-sm font-semibold text-ink-900">{dimension.score}</span>
                  </div>
                  <div className="mt-1.5">
                    <ScoreBar score={dimension.score} color={scoreColor(dimension.score)} />
                  </div>
                  <p className="mt-1 text-[11px] text-ink-500">{dimension.calculation}</p>
                </li>
              ))}
            </ul>
          ) : (
            <EmptyState title="No scores yet" hint="Scores appear as soon as the first checks have run." />
          )}
        </Card>

        <div className="space-y-6">
          <Card title="Active agent" subtitle="Who is working, and on what.">
            {claim.active_agent ? (
              <div className="flex items-start gap-3">
                <LiveDot />
                <div>
                  <p className="text-sm font-medium text-ink-900">
                    {AGENT_LABEL[claim.active_agent] ?? claim.active_agent}
                  </p>
                  <p className="mt-1 text-xs leading-relaxed text-ink-600">{claim.latest_action}</p>
                </div>
              </div>
            ) : (
              <div>
                <p className="text-sm font-medium text-ink-900">Idle</p>
                <p className="mt-1 text-xs leading-relaxed text-ink-600">{claim.latest_action || "Nothing in flight."}</p>
              </div>
            )}
            <dl className="mt-4 space-y-1.5 border-t border-ink-100 pt-3 text-xs">
              <div className="flex justify-between gap-3">
                <dt className="text-ink-500">Reasoning backend</dt>
                <dd className="font-medium text-ink-800">{claim.reasoning_backend}</dd>
              </div>
              <div className="flex justify-between gap-3">
                <dt className="text-ink-500">Open loopholes</dt>
                <dd className="font-medium text-ink-800">{openLoopholes}</dd>
              </div>
              <div className="flex justify-between gap-3">
                <dt className="text-ink-500">Subclaims</dt>
                <dd className="font-medium text-ink-800">
                  {subclaims.filter((s) => s.status !== "untested").length} of {subclaims.length} tested
                </dd>
              </div>
            </dl>
          </Card>

          <Card title="Budget" subtitle="Compute units spent on verification.">
            <div className="flex items-baseline justify-between">
              <span className="tabular text-2xl font-semibold text-ink-900">
                {claim.budget.consumed_units.toFixed(2)}
              </span>
              <span className="text-xs text-ink-500">of {claim.budget.total_units.toFixed(0)}</span>
            </div>
            <div className="mt-2">
              <ScoreBar score={budgetPct} color="var(--color-accent-500)" />
            </div>
          </Card>
        </div>
      </div>

      <Card
        title="Run-health incidents"
        subtitle="What RunMedic saw while the experiments were running."
        action={
          <button
            type="button"
            onClick={() => onOpenTab("health")}
            className="shrink-0 text-xs font-medium text-accent-600 hover:text-accent-700"
          >
            Live run health →
          </button>
        }
      >
        {incidents.length === 0 ? (
          <EmptyState title="No incidents recorded" hint="Every run so far has trained cleanly." />
        ) : (
          <ul className="divide-y divide-ink-100">
            {incidents.slice(0, 6).map((event) => (
              <li key={event.id} className="flex flex-wrap items-start gap-3 py-2.5 first:pt-0 last:pb-0">
                <Badge tone={healthTone(event.status)}>{titleCase(event.anomaly)}</Badge>
                <div className="min-w-0 flex-1">
                  <p className="text-sm leading-relaxed break-words text-ink-800">{event.detail}</p>
                  {event.action_taken && (
                    <p className="mt-1 text-xs leading-relaxed break-words text-ink-500">{event.action_taken}</p>
                  )}
                </div>
              </li>
            ))}
          </ul>
        )}
      </Card>
    </div>
  );
}
