"use client";

import Link from "next/link";
import { Badge, LiveDot } from "./ui";
import { AGENT_LABEL, claimStateTone, titleCase } from "@/lib/format";
import type { ClaimSnapshot } from "@/lib/types";

export function AppHeader({ snapshot }: { snapshot?: ClaimSnapshot | null }) {
  const claim = snapshot?.claim;
  const infra = snapshot?.infrastructure;
  const busy = claim ? ["analyzing", "skeptic_review", "planning", "executing", "auditing"].includes(claim.state) : false;

  return (
    <header className="header-gradient text-white">
      <div className="mx-auto flex max-w-[1400px] flex-wrap items-center gap-x-6 gap-y-3 px-6 py-4">
        <Link href="/" className="group flex items-center gap-3">
          <span className="grid h-9 w-9 place-items-center rounded-lg bg-white/10 text-base font-semibold ring-1 ring-white/15">
            LG
          </span>
          <span>
            <span className="block text-sm font-semibold tracking-tight">LabGuard AI</span>
            <span className="block text-[11px] text-white/60">
              Challenge the claim. Protect the run. Trust the result.
            </span>
          </span>
        </Link>

        {claim && (
          <div className="flex flex-wrap items-center gap-2">
            <Badge tone={claimStateTone(claim.state)}>
              {busy && <LiveDot tone="info" />}
              {titleCase(claim.state)}
            </Badge>
            {claim.active_agent && (
              <span className="rounded-full bg-white/10 px-2.5 py-0.5 text-xs text-white/85 ring-1 ring-inset ring-white/15">
                {AGENT_LABEL[claim.active_agent] ?? claim.active_agent}
              </span>
            )}
          </div>
        )}

        {infra && (
          <dl className="ml-auto flex flex-wrap items-center gap-x-5 gap-y-1 text-[11px] text-white/70">
            <InfraItem label="Mode" value={infra.mode} />
            <InfraItem label="State" value={infra.state_store} />
            <InfraItem label="Bus" value={infra.job_bus} />
            <InfraItem label="Reasoning" value={infra.reasoning} />
          </dl>
        )}
      </div>
    </header>
  );
}

function InfraItem({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center gap-1.5">
      <dt className="uppercase tracking-wider text-white/45">{label}</dt>
      <dd className="font-medium text-white/90">{value}</dd>
    </div>
  );
}
