"use client";

import { use, useState } from "react";
import { AppHeader } from "@/components/Header";
import { ClaimMap } from "@/components/tabs/ClaimMap";
import { HealthTab } from "@/components/tabs/HealthTab";
import { LedgerTab } from "@/components/tabs/LedgerTab";
import { Overview } from "@/components/tabs/Overview";
import { PlanTab } from "@/components/tabs/PlanTab";
import { QueueTab } from "@/components/tabs/QueueTab";
import { ReportTab } from "@/components/tabs/ReportTab";
import { ErrorState, Skeleton } from "@/components/ui";
import { useSnapshot } from "@/lib/useSnapshot";

const TABS = [
  { id: "overview", label: "Overview" },
  { id: "map", label: "Claim map" },
  { id: "plan", label: "Experiment plan" },
  { id: "queue", label: "Queue" },
  { id: "health", label: "Live run health" },
  { id: "ledger", label: "Evidence ledger" },
  { id: "report", label: "Final report" },
] as const;

type TabId = (typeof TABS)[number]["id"];

export default function ClaimPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const { snapshot, error, loading, refresh } = useSnapshot(id);
  const [tab, setTab] = useState<TabId>("overview");

  const badge = (id: TabId): number | null => {
    if (!snapshot) return null;
    if (id === "plan") return snapshot.plans.some((p) => p.status === "awaiting_approval") ? 1 : null;
    if (id === "queue") return snapshot.jobs.filter((j) => !["completed", "rejected"].includes(j.state)).length || null;
    if (id === "health") return snapshot.jobs.reduce((n, j) => n + j.health.events.length, 0) || null;
    return null;
  };

  return (
    <div className="min-h-screen">
      <AppHeader snapshot={snapshot} />

      <nav className="sticky top-0 z-10 border-b border-ink-100 bg-white/85 backdrop-blur">
        <div className="mx-auto flex max-w-[1400px] gap-1 overflow-x-auto px-6">
          {TABS.map((entry) => {
            const count = badge(entry.id);
            const active = tab === entry.id;
            return (
              <button
                key={entry.id}
                type="button"
                onClick={() => setTab(entry.id)}
                aria-current={active ? "page" : undefined}
                className={`relative whitespace-nowrap border-b-2 px-3.5 py-3 text-sm font-medium transition ${
                  active
                    ? "border-accent-600 text-ink-900"
                    : "border-transparent text-ink-500 hover:text-ink-800"
                }`}
              >
                {entry.label}
                {count != null && (
                  <span className="ml-2 rounded-full bg-accent-100 px-1.5 py-0.5 text-[10px] font-semibold text-accent-700">
                    {count}
                  </span>
                )}
              </button>
            );
          })}
        </div>
      </nav>

      <main className="mx-auto max-w-[1400px] px-6 py-6">
        {error && !snapshot && <ErrorState message={error} onRetry={() => void refresh()} />}
        {loading && !snapshot && (
          <div className="space-y-4">
            <Skeleton className="h-28 w-full" />
            <div className="grid gap-4 sm:grid-cols-4">
              <Skeleton className="h-20" />
              <Skeleton className="h-20" />
              <Skeleton className="h-20" />
              <Skeleton className="h-20" />
            </div>
            <Skeleton className="h-72 w-full" />
          </div>
        )}
        {snapshot && (
          <>
            {error && (
              <p className="mb-4 rounded-lg border border-warn-500/25 bg-warn-50 px-4 py-2.5 text-xs text-warn-700">
                Lost contact with the API ({error}). Showing the last state received and still retrying.
              </p>
            )}
            {tab === "overview" && <Overview snapshot={snapshot} onOpenTab={(t) => setTab(t as TabId)} />}
            {tab === "map" && <ClaimMap snapshot={snapshot} />}
            {tab === "plan" && <PlanTab snapshot={snapshot} onChange={() => void refresh()} />}
            {tab === "queue" && <QueueTab snapshot={snapshot} onChange={() => void refresh()} />}
            {tab === "health" && <HealthTab snapshot={snapshot} />}
            {tab === "ledger" && <LedgerTab snapshot={snapshot} />}
            {tab === "report" && <ReportTab snapshot={snapshot} />}
          </>
        )}
      </main>
    </div>
  );
}
