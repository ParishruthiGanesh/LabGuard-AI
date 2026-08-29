"use client";

import { ScoreBar } from "../charts";
import { Badge, Card, EmptyState } from "../ui";
import { api } from "@/lib/api";
import { scoreColor, stanceTone, titleCase, verdictTone } from "@/lib/format";
import type { ClaimSnapshot, ScoreCheck } from "@/lib/types";

const VERDICT_BLURB: Record<string, string> = {
  supported: "The evidence backs the claim as stated.",
  provisionally_supported: "Nothing contradicts the claim, but some questions are still open.",
  fragile: "The claim survives, but only under narrow conditions.",
  inconclusive: "The evidence gathered cannot settle the claim either way.",
  not_sufficiently_supported: "The evidence does not establish the claim as stated.",
  refuted: "The evidence points the other way.",
};

export function ReportTab({ snapshot }: { snapshot: ClaimSnapshot }) {
  const { claim, verdict, score, jobs } = snapshot;

  if (!verdict) {
    return (
      <Card>
        <EmptyState
          title="No verdict yet"
          hint="The Verdict Agent runs once the Evidence Auditor decides no further experiment would change a conclusion."
        />
      </Card>
    );
  }

  const incidents = verdict.run_health_incidents;
  const repro = verdict.reproducibility as {
    seeds_tested?: number[];
    equalised_training_budget?: boolean;
    checkpoint_selection?: string;
    dataset?: Record<string, unknown>;
    configurations?: Record<string, unknown>[];
  };

  return (
    <div className="space-y-6">
      <div className="card overflow-hidden">
        <div className="header-gradient px-6 py-6 text-white">
          <p className="text-[11px] font-medium uppercase tracking-wider text-white/60">Final verdict</p>
          <h2 className="mt-2 text-2xl font-semibold tracking-tight">{titleCase(verdict.status)}</h2>
          <p className="mt-2 max-w-3xl text-sm leading-relaxed text-white/80">
            {VERDICT_BLURB[verdict.status] ?? ""}
          </p>
        </div>
        <div className="px-6 py-5">
          <p className="text-sm font-medium leading-relaxed text-ink-900">{verdict.headline}</p>
          <p className="mt-3 max-w-4xl text-sm leading-relaxed text-ink-700">{verdict.narrative}</p>
          <div className="mt-4 flex flex-wrap items-center gap-2">
            <Badge tone={verdictTone(verdict.status)}>{titleCase(verdict.status)}</Badge>
            <Badge tone="neutral">Written by {verdict.generated_by}</Badge>
            {verdict.status !== verdict.rule_based_status && (
              <Badge tone="warn">Measured status: {titleCase(verdict.rule_based_status)}</Badge>
            )}
            <a
              href={api.reportUrl(claim.id)}
              className="ml-auto inline-flex items-center gap-2 rounded-md bg-accent-600 px-3.5 py-2 text-sm font-medium text-white transition hover:bg-accent-700"
              download
            >
              Download the full report
            </a>
          </div>
        </div>
      </div>

      {score && (
        <Card
          title="Reliability score"
          subtitle="Every dimension is a weighted pass rate over the named checks below. No number here was written by a language model."
        >
          <div className="space-y-5">
            {score.dimensions.map((dimension) => (
              <div key={dimension.dimension}>
                <div className="flex flex-wrap items-baseline justify-between gap-2">
                  <h3 className="text-sm font-semibold text-ink-900">{titleCase(dimension.dimension)}</h3>
                  <span className="tabular text-sm font-semibold text-ink-900">{dimension.score}</span>
                </div>
                <div className="mt-1.5">
                  <ScoreBar score={dimension.score} color={scoreColor(dimension.score)} />
                </div>
                <p className="mt-1 text-[11px] text-ink-500">{dimension.calculation}</p>
                <ul className="mt-2.5 space-y-1.5">
                  {dimension.checks.map((check) => (
                    <CheckRow key={check.id} check={check} />
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </Card>
      )}

      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="Evidence summary" subtitle="The measurements the verdict rests on.">
          {verdict.evidence_summary.length === 0 ? (
            <EmptyState title="No evidence recorded" />
          ) : (
            <ul className="space-y-2.5">
              {verdict.evidence_summary.map((statement, i) => {
                const item = snapshot.evidence.find((e) => e.statement === statement);
                return (
                  <li key={i} className="flex items-start gap-2.5">
                    <Badge tone={item ? stanceTone(item.stance) : "neutral"}>
                      {item ? titleCase(item.stance) : "Evidence"}
                    </Badge>
                    <p className="min-w-0 flex-1 text-sm leading-relaxed text-ink-800">{statement}</p>
                  </li>
                );
              })}
            </ul>
          )}
        </Card>

        <Card title="Remaining uncertainty" subtitle="What this verification did not settle.">
          {verdict.remaining_uncertainty.length === 0 ? (
            <EmptyState title="Nothing material was left open" />
          ) : (
            <ul className="space-y-2 text-sm leading-relaxed text-ink-800">
              {verdict.remaining_uncertainty.map((item, i) => (
                <li key={i} className="flex gap-2.5">
                  <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-warn-500" aria-hidden />
                  {item}
                </li>
              ))}
            </ul>
          )}
        </Card>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="Run-health incidents" subtitle="Operational problems found while verifying.">
          {incidents.length === 0 ? (
            <EmptyState title="No incidents" hint="Every run trained cleanly." />
          ) : (
            <ul className="space-y-2 text-sm leading-relaxed text-ink-800">
              {incidents.map((incident, i) => (
                <li key={i} className="rounded-lg border border-ink-100 px-3.5 py-2.5 text-xs break-words">
                  {incident}
                </li>
              ))}
            </ul>
          )}
        </Card>

        <Card title="Reproducibility" subtitle="Enough detail to re-run this verification from scratch.">
          <dl className="space-y-2 text-sm">
            <Row label="Seeds tested" value={(repro.seeds_tested ?? []).join(", ") || "none"} />
            <Row label="Training budget equalised" value={String(repro.equalised_training_budget ?? false)} />
            <Row label="Checkpoint selection" value={String(repro.checkpoint_selection ?? "-")} />
            <Row label="Runs executed" value={`${jobs.filter((j) => j.state === "completed").length} completed`} />
          </dl>
          {repro.configurations && (
            <ul className="mt-3 space-y-1.5 border-t border-ink-100 pt-3 font-mono text-[11px] leading-relaxed text-ink-600">
              {repro.configurations.map((config, i) => (
                <li key={i}>
                  {String(config.name)}: {String(config.family)}, {String(config.epochs)} epochs, lr{" "}
                  {String(config.learning_rate)}, batch {String(config.batch_size)}, class weight{" "}
                  {String(config.class_weight)}
                </li>
              ))}
            </ul>
          )}
        </Card>
      </div>
    </div>
  );
}

function CheckRow({ check }: { check: ScoreCheck }) {
  const tone = check.passed === true ? "ok" : check.passed === false ? "bad" : "neutral";
  const symbol = check.passed === true ? "✓" : check.passed === false ? "✕" : "–";
  return (
    <li className="flex items-start gap-2.5 rounded-lg bg-ink-50 px-3 py-2">
      <span
        className={`mt-0.5 grid h-4 w-4 shrink-0 place-items-center rounded-full text-[10px] font-bold ${
          tone === "ok"
            ? "bg-ok-500/15 text-ok-700"
            : tone === "bad"
              ? "bg-bad-500/15 text-bad-700"
              : "bg-ink-200 text-ink-500"
        }`}
        aria-hidden
      >
        {symbol}
      </span>
      <div className="min-w-0 flex-1">
        <p className="text-xs font-medium text-ink-800">
          {check.label}
          <span className="ml-1.5 font-normal text-ink-400">weight {check.weight}</span>
        </p>
        <p className="mt-0.5 text-[11px] leading-relaxed text-ink-500">{check.detail}</p>
      </div>
      <span className="shrink-0 font-mono text-[10px] text-ink-300">{check.id}</span>
    </li>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between gap-3">
      <dt className="text-ink-500">{label}</dt>
      <dd className="text-right font-medium text-ink-800">{value}</dd>
    </div>
  );
}
