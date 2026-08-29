"use client";

import { Badge, Card, Disclosure, EmptyState } from "../ui";
import { pct, stanceTone, subclaimTone, titleCase } from "@/lib/format";
import type { ClaimSnapshot, Evidence } from "@/lib/types";

const LOOPHOLE_TONE = {
  open: "warn",
  investigating: "info",
  confirmed: "bad",
  refuted: "ok",
  unresolved: "warn",
} as const;

export function ClaimMap({ snapshot }: { snapshot: ClaimSnapshot }) {
  const { claim, subclaims, loopholes, alternatives, evidence } = snapshot;
  const byId = new Map<string, Evidence>(evidence.map((e) => [e.id, e]));

  return (
    <div className="space-y-6">
      <Card title="Main claim" subtitle="What the researcher asked LabGuard to verify.">
        <p className="text-base font-medium leading-snug text-ink-900">{claim.text}</p>
        {claim.context.notes && (
          <p className="mt-3 rounded-lg bg-ink-50 px-4 py-3 text-xs leading-relaxed text-ink-600">
            <span className="font-medium text-ink-700">As submitted: </span>
            {claim.context.notes}
          </p>
        )}
        <dl className="mt-4 grid gap-3 sm:grid-cols-3">
          <div>
            <dt className="text-[11px] uppercase tracking-wider text-ink-400">Dataset</dt>
            <dd className="mt-0.5 text-sm text-ink-800">
              {claim.context.dataset.name}
              <span className="block text-xs text-ink-500">
                {claim.context.dataset.n_samples.toLocaleString()} rows ·{" "}
                {pct(claim.context.dataset.positive_rate)} positive
              </span>
            </dd>
          </div>
          {claim.context.models
            .filter((m) => m.role === "primary")
            .map((model) => (
              <div key={model.name}>
                <dt className="text-[11px] uppercase tracking-wider text-ink-400">
                  {model.is_baseline ? "Baseline" : "Candidate"}
                </dt>
                <dd className="mt-0.5 text-sm text-ink-800">
                  {model.name}
                  <span className="block text-xs text-ink-500">
                    {model.family}
                    {model.hidden_units ? ` · ${model.hidden_units} hidden` : ""} · {model.epochs} epochs · lr{" "}
                    {model.learning_rate}
                  </span>
                </dd>
              </div>
            ))}
        </dl>
      </Card>

      <Card
        title="Testable subclaims"
        subtitle="The Claim Analyst turns one broad statement into things that can actually be measured."
      >
        {subclaims.length === 0 ? (
          <EmptyState title="Decomposing the claim" hint="Subclaims appear once the Claim Analyst has run." />
        ) : (
          <ul className="space-y-3">
            {subclaims.map((subclaim) => {
              const items = subclaim.evidence_ids.map((id) => byId.get(id)).filter(Boolean) as Evidence[];
              return (
                <li key={subclaim.id} className="rounded-lg border border-ink-100 px-4 py-3">
                  <div className="flex flex-wrap items-start justify-between gap-3">
                    <p className="min-w-0 flex-1 text-sm font-medium leading-relaxed text-ink-900">
                      {subclaim.statement}
                    </p>
                    <div className="flex items-center gap-2">
                      <Badge tone={subclaimTone(subclaim.status)}>{titleCase(subclaim.status)}</Badge>
                      {subclaim.status !== "untested" && (
                        <span className="tabular text-xs text-ink-500">{pct(subclaim.confidence, 0)}</span>
                      )}
                    </div>
                  </div>
                  <p className="mt-1.5 text-xs text-ink-500">
                    <span className="font-medium text-ink-600">Measured by: </span>
                    {subclaim.measurable_quantity}
                  </p>
                  {items.length > 0 && (
                    <div className="mt-3">
                      <Disclosure summary={`${items.length} piece(s) of evidence`}>
                        <ul className="space-y-2">
                          {items.map((item) => (
                            <li key={item.id} className="flex items-start gap-2.5">
                              <Badge tone={stanceTone(item.stance)}>{titleCase(item.stance)}</Badge>
                              <p className="min-w-0 flex-1 text-xs leading-relaxed text-ink-700">{item.statement}</p>
                            </li>
                          ))}
                        </ul>
                      </Disclosure>
                    </div>
                  )}
                </li>
              );
            })}
          </ul>
        )}
      </Card>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card
          title="Scientific loopholes"
          subtitle="Reasons the reported result might not mean what it appears to mean."
        >
          {loopholes.length === 0 ? (
            <EmptyState title="No loopholes raised yet" />
          ) : (
            <ul className="space-y-3">
              {[...loopholes]
                .sort((a, b) => b.severity - a.severity)
                .map((hole) => (
                  <li key={hole.id} className="rounded-lg border border-ink-100 px-4 py-3">
                    <div className="flex flex-wrap items-start justify-between gap-2">
                      <p className="min-w-0 flex-1 text-sm font-medium text-ink-900">{hole.title}</p>
                      <Badge tone={LOOPHOLE_TONE[hole.status]}>{titleCase(hole.status)}</Badge>
                    </div>
                    <p className="mt-1.5 text-xs leading-relaxed text-ink-600">{hole.rationale}</p>
                    <div className="mt-2.5 flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-ink-500">
                      <span className="font-mono">{hole.kind}</span>
                      <span>severity {hole.severity.toFixed(2)}</span>
                      <span>found by {hole.detected_by}</span>
                    </div>
                    {hole.resolution && (
                      <p className="mt-2 border-t border-ink-100 pt-2 text-xs leading-relaxed text-ink-700">
                        {hole.resolution}
                      </p>
                    )}
                  </li>
                ))}
            </ul>
          )}
        </Card>

        <Card
          title="Alternative explanations"
          subtitle="Rival accounts of the same result that the plan has to rule out."
        >
          {alternatives.length === 0 ? (
            <EmptyState title="None raised yet" />
          ) : (
            <ul className="space-y-2.5">
              {alternatives.map((alt) => (
                <li key={alt.id} className="rounded-lg border border-ink-100 px-4 py-3">
                  <p className="text-sm leading-relaxed text-ink-800">{alt.statement}</p>
                  {alt.tested_by_action && (
                    <p className="mt-1.5 text-[11px] text-ink-500">
                      Tested by <span className="font-mono text-ink-600">{alt.tested_by_action}</span>
                    </p>
                  )}
                </li>
              ))}
            </ul>
          )}
        </Card>
      </div>
    </div>
  );
}
