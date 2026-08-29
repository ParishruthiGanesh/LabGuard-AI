"use client";

import { useMemo, useState } from "react";
import { Badge, Card, Disclosure, EmptyState } from "../ui";
import { AGENT_LABEL, relativeTime, stanceTone, titleCase } from "@/lib/format";
import type { ClaimSnapshot } from "@/lib/types";

const AGENT_TONE: Record<string, "accent" | "info" | "warn" | "neutral"> = {
  claim_analyst: "accent",
  scientific_skeptic: "accent",
  experiment_planner: "info",
  run_manager: "info",
  run_medic: "warn",
  evidence_auditor: "accent",
  verdict_agent: "accent",
  orchestrator: "neutral",
};

export function LedgerTab({ snapshot }: { snapshot: ClaimSnapshot }) {
  const { ledger, evidence } = snapshot;
  const [agent, setAgent] = useState<string>("all");

  const agents = useMemo(() => ["all", ...new Set(ledger.map((e) => e.agent))], [ledger]);
  const rows = agent === "all" ? ledger : ledger.filter((e) => e.agent === agent);
  const evidenceByJob = useMemo(() => {
    const map = new Map<string, typeof evidence>();
    for (const item of evidence) {
      const list = map.get(item.job_id) ?? [];
      list.push(item);
      map.set(item.job_id, list);
    }
    return map;
  }, [evidence]);

  if (ledger.length === 0) {
    return (
      <Card>
        <EmptyState title="The ledger is empty" hint="Every agent decision is appended here as it happens." />
      </Card>
    );
  }

  return (
    <Card
      title="Evidence ledger"
      subtitle="Append-only. Each row records who acted, why, what they were given, what came back, and what they decided."
      action={
        <div className="flex shrink-0 flex-wrap gap-1.5">
          {agents.map((name) => (
            <button
              key={name}
              type="button"
              onClick={() => setAgent(name)}
              className={`rounded-full px-2.5 py-1 text-[11px] font-medium transition ${
                agent === name ? "bg-ink-900 text-white" : "bg-ink-50 text-ink-600 hover:bg-ink-100"
              }`}
            >
              {name === "all" ? "All agents" : (AGENT_LABEL[name] ?? name)}
            </button>
          ))}
        </div>
      }
    >
      <ol className="relative space-y-4 border-l border-ink-200 pl-6">
        {rows.map((entry) => {
          const linked = entry.job_id ? evidenceByJob.get(entry.job_id) : undefined;
          const hasDetail =
            Object.keys(entry.input_summary).length > 0 || Object.keys(entry.result_summary).length > 0;
          return (
            <li key={entry.id} className="relative">
              <span
                className="absolute -left-[1.85rem] top-1.5 grid h-5 w-5 place-items-center rounded-full bg-white text-[9px] font-semibold text-ink-500 ring-1 ring-ink-200"
                aria-hidden
              >
                {entry.sequence}
              </span>
              <div className="flex flex-wrap items-center gap-2">
                <Badge tone={AGENT_TONE[entry.agent] ?? "neutral"}>{AGENT_LABEL[entry.agent] ?? entry.agent}</Badge>
                <span className="font-mono text-xs font-medium text-ink-800">{entry.action}</span>
                <span className="ml-auto text-[11px] text-ink-400">{relativeTime(entry.at)}</span>
              </div>
              {entry.reason && <p className="mt-1.5 text-xs leading-relaxed text-ink-600">{entry.reason}</p>}
              {entry.decision && (
                <p className="mt-1.5 text-sm leading-relaxed text-ink-900">
                  <span className="text-[11px] font-medium uppercase tracking-wider text-ink-400">Decision · </span>
                  {entry.decision}
                </p>
              )}
              {linked && linked.length > 0 && (
                <ul className="mt-2 space-y-1.5">
                  {linked.map((item) => (
                    <li key={item.id} className="flex items-start gap-2">
                      <Badge tone={stanceTone(item.stance)}>{titleCase(item.stance)}</Badge>
                      <span className="min-w-0 flex-1 text-xs leading-relaxed text-ink-700">{item.statement}</span>
                    </li>
                  ))}
                </ul>
              )}
              {entry.artifact_uris.length > 0 && (
                <p className="mt-1.5 flex flex-wrap gap-2 text-[11px] text-ink-400">
                  {entry.artifact_uris.map((uri) => (
                    <span key={uri} className="font-mono" title={uri}>
                      {uri.split("/").pop()}
                    </span>
                  ))}
                </p>
              )}
              {hasDetail && (
                <div className="mt-2">
                  <Disclosure summary="Inputs and results">
                    <div className="grid gap-2 lg:grid-cols-2">
                      {Object.keys(entry.input_summary).length > 0 && (
                        <pre className="overflow-x-auto rounded bg-ink-950 px-3 py-2.5 font-mono text-[10px] leading-relaxed text-ink-100">
                          {JSON.stringify(entry.input_summary, null, 2)}
                        </pre>
                      )}
                      {Object.keys(entry.result_summary).length > 0 && (
                        <pre className="overflow-x-auto rounded bg-ink-950 px-3 py-2.5 font-mono text-[10px] leading-relaxed text-ink-100">
                          {JSON.stringify(entry.result_summary, null, 2)}
                        </pre>
                      )}
                    </div>
                  </Disclosure>
                </div>
              )}
            </li>
          );
        })}
      </ol>
    </Card>
  );
}
