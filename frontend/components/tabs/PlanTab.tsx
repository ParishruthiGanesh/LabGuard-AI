"use client";

import { useState } from "react";
import { Badge, Button, Card, Disclosure, EmptyState } from "../ui";
import { api } from "@/lib/api";
import { titleCase, units } from "@/lib/format";
import type { ClaimSnapshot } from "@/lib/types";

const CATEGORY_TONE = {
  diagnostic: "info",
  experiment: "accent",
  recovery: "warn",
  report: "neutral",
} as const;

export function PlanTab({ snapshot, onChange }: { snapshot: ClaimSnapshot; onChange: () => void }) {
  const { claim, plans, loopholes } = snapshot;
  const [busy, setBusy] = useState<"approve" | "reject" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const pending = plans.find((p) => p.status === "awaiting_approval");
  const holeById = new Map(loopholes.map((h) => [h.id, h]));

  async function decide(approved: boolean) {
    if (!pending) return;
    setBusy(approved ? "approve" : "reject");
    setError(null);
    try {
      await api.decidePlan(claim.id, pending.id, approved);
      onChange();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(null);
    }
  }

  if (plans.length === 0) {
    return (
      <Card>
        <EmptyState
          title="No plan yet"
          hint="The Experiment Planner runs once the Scientific Skeptic has finished raising loopholes."
        />
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {pending && (
        <div className="card border-warn-500/30 bg-warn-50/50">
          <div className="flex flex-wrap items-start justify-between gap-4 px-5 py-4">
            <div className="min-w-0">
              <h2 className="text-sm font-semibold text-ink-900">Approval required</h2>
              <p className="mt-1 max-w-2xl text-xs leading-relaxed text-ink-600">
                {pending.summary} Nothing in this round has run: every job is held at{" "}
                <span className="font-mono">awaiting_approval</span> until you decide.
              </p>
              {error && <p className="mt-2 text-xs font-medium text-bad-700">{error}</p>}
            </div>
            <div className="flex shrink-0 gap-2">
              <Button variant="danger" onClick={() => decide(false)} busy={busy === "reject"} disabled={busy !== null}>
                Reject
              </Button>
              <Button onClick={() => decide(true)} busy={busy === "approve"} disabled={busy !== null}>
                Approve {units(pending.total_cost_units)}
              </Button>
            </div>
          </div>
        </div>
      )}

      {[...plans].reverse().map((plan) => (
        <Card
          key={plan.id}
          title={`Round ${plan.round_index + 1}`}
          subtitle={plan.summary}
          action={
            <div className="flex shrink-0 items-center gap-2">
              <Badge tone={plan.status === "rejected" ? "bad" : plan.status === "executed" ? "ok" : "neutral"}>
                {titleCase(plan.status)}
              </Badge>
              <span className="tabular text-xs text-ink-500">{units(plan.total_cost_units)}</span>
            </div>
          }
        >
          {plan.approved_by && (
            <p className="mb-3 text-xs text-ink-500">
              {plan.status === "rejected" ? "Rejected" : "Approved"} by{" "}
              <span className="font-medium text-ink-700">{plan.approved_by}</span>
            </p>
          )}
          <ul className="space-y-3">
            {plan.items.map((item) => (
              <li key={item.id} className="rounded-lg border border-ink-100 px-4 py-3">
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div className="flex min-w-0 flex-1 flex-wrap items-center gap-2">
                    <span className="font-mono text-sm font-medium text-ink-900">{item.action_type}</span>
                    <Badge tone={CATEGORY_TONE[item.category as keyof typeof CATEGORY_TONE] ?? "neutral"}>
                      {titleCase(item.category)}
                    </Badge>
                    {item.requires_approval && <Badge tone="warn">Needs approval</Badge>}
                  </div>
                  <dl className="flex shrink-0 items-center gap-4 text-xs">
                    <div className="text-right">
                      <dt className="text-[10px] uppercase tracking-wider text-ink-400">Cost</dt>
                      <dd className="tabular font-medium text-ink-800">{item.estimated_cost_units.toFixed(2)}u</dd>
                    </div>
                    <div className="text-right">
                      <dt className="text-[10px] uppercase tracking-wider text-ink-400">Info gain</dt>
                      <dd className="tabular font-medium text-ink-800">
                        {item.expected_information_gain.toFixed(2)}
                      </dd>
                    </div>
                  </dl>
                </div>
                <p className="mt-2 text-xs leading-relaxed text-ink-600">{item.reason}</p>
                {item.targets_loophole_ids.length > 0 && (
                  <div className="mt-2 flex flex-wrap gap-1.5">
                    {item.targets_loophole_ids.map((id) => (
                      <span
                        key={id}
                        className="rounded bg-ink-50 px-1.5 py-0.5 font-mono text-[10px] text-ink-600"
                        title={holeById.get(id)?.title}
                      >
                        {holeById.get(id)?.kind ?? id}
                      </span>
                    ))}
                  </div>
                )}
                <div className="mt-2.5">
                  <Disclosure summary="Validated parameters">
                    <pre className="overflow-x-auto rounded bg-ink-950 px-3 py-2.5 font-mono text-[11px] leading-relaxed text-ink-100">
                      {JSON.stringify(item.params, null, 2)}
                    </pre>
                  </Disclosure>
                </div>
              </li>
            ))}
          </ul>
        </Card>
      ))}
    </div>
  );
}
