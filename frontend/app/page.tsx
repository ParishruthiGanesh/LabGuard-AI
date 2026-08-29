"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { AppHeader } from "@/components/Header";
import { Badge, Button, Card, ErrorState, Skeleton } from "@/components/ui";
import { api } from "@/lib/api";
import { relativeTime, titleCase } from "@/lib/format";
import type { AppConfig, AutonomyMode, Claim } from "@/lib/types";

const AUTONOMY_COPY: Record<AutonomyMode, { label: string; detail: string }> = {
  observe_only: {
    label: "Observe only",
    detail: "Detect problems and recommend actions. Nothing is executed, even if you approve it.",
  },
  safe_repair: {
    label: "Safe repair",
    detail: "Save checkpoints, apply early stopping, retry transient failures and resume safe jobs.",
  },
  managed_autonomy: {
    label: "Managed autonomy",
    detail:
      "Also run inexpensive diagnostics, adjust parameters inside fixed bounds and schedule extra seeds. Expensive experiments still need approval.",
  },
};

const INVOKERS = [
  { key: "planner", label: "Scheduled by the planner", hint: "into a verification round" },
  { key: "runmedic", label: "Applied by RunMedic", hint: "as a repair to a running job" },
  { key: "orchestrator", label: "Run by the orchestrator", hint: "when the claim is finalised" },
] as const;

export default function LauncherPage() {
  const router = useRouter();
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [claims, setClaims] = useState<Claim[]>([]);
  const [autonomy, setAutonomy] = useState<AutonomyMode>("managed_autonomy");
  const [budget, setBudget] = useState(40);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    try {
      const [cfg, existing] = await Promise.all([api.config(), api.listClaims()]);
      setConfig(cfg);
      setClaims(existing);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  useEffect(() => {
    void load();
  }, []);

  async function start() {
    setStarting(true);
    setError(null);
    try {
      const claim = await api.createClaim({
        use_demo_scenario: true,
        autonomy_mode: autonomy,
        budget: { total_units: budget, consumed_units: 0, approval_threshold_units: 6 },
      });
      router.push(`/claims/${claim.id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setStarting(false);
    }
  }

  return (
    <div className="min-h-screen">
      <AppHeader />
      <main className="mx-auto max-w-[1100px] px-6 py-10">
        <div className="max-w-2xl">
          <h1 className="text-3xl font-semibold tracking-tight text-ink-900">
            An experiment that ran correctly can still support the wrong conclusion.
          </h1>
          <p className="mt-3 text-base leading-relaxed text-ink-600">
            LabGuard takes a research claim, breaks it into things that can be measured, looks for the loopholes
            that would explain the result away, and runs the smallest experiments that could settle them. While
            those run, it watches every one for failures and repairs what it safely can.
          </p>
        </div>

        {error && (
          <div className="mt-6">
            <ErrorState message={error} onRetry={load} />
          </div>
        )}

        <div className="mt-8 grid gap-6 lg:grid-cols-[1.4fr_1fr]">
          <Card
            title="Run the bundled scenario"
            subtitle="A synthetic violence-detection benchmark with real, reproducible weaknesses built into it."
          >
            {config ? (
              <>
                <blockquote className="rounded-lg border-l-2 border-accent-500 bg-accent-50/40 px-4 py-3">
                  <p className="text-sm font-medium leading-relaxed text-ink-900">
                    &ldquo;{config.demo_scenario.text}&rdquo;
                  </p>
                  <p className="mt-2 text-xs leading-relaxed text-ink-600">{config.demo_scenario.context.notes}</p>
                </blockquote>

                <div className="mt-5">
                  <p className="text-xs font-medium uppercase tracking-wider text-ink-400">Autonomy policy</p>
                  <div className="mt-2 space-y-2">
                    {(Object.keys(AUTONOMY_COPY) as AutonomyMode[]).map((mode) => (
                      <label
                        key={mode}
                        className={`flex cursor-pointer gap-3 rounded-lg border px-4 py-3 transition ${
                          autonomy === mode
                            ? "border-accent-500 bg-accent-50/50"
                            : "border-ink-200 hover:border-ink-300"
                        }`}
                      >
                        <input
                          type="radio"
                          name="autonomy"
                          value={mode}
                          checked={autonomy === mode}
                          onChange={() => setAutonomy(mode)}
                          className="mt-1 accent-[var(--color-accent-600)]"
                        />
                        <span>
                          <span className="block text-sm font-medium text-ink-900">{AUTONOMY_COPY[mode].label}</span>
                          <span className="mt-0.5 block text-xs leading-relaxed text-ink-600">
                            {AUTONOMY_COPY[mode].detail}
                          </span>
                        </span>
                      </label>
                    ))}
                  </div>
                </div>

                <div className="mt-5">
                  <label htmlFor="budget" className="text-xs font-medium uppercase tracking-wider text-ink-400">
                    Compute budget — {budget} units
                  </label>
                  <input
                    id="budget"
                    type="range"
                    min={12}
                    max={60}
                    step={2}
                    value={budget}
                    onChange={(e) => setBudget(Number(e.target.value))}
                    className="mt-2 w-full accent-[var(--color-accent-600)]"
                  />
                  <p className="mt-1 text-xs text-ink-500">
                    One unit is roughly one short training run. Anything above 6 units needs your approval.
                  </p>
                </div>

                <div className="mt-6">
                  <Button onClick={start} busy={starting} className="w-full">
                    Start verification
                  </Button>
                </div>
              </>
            ) : (
              <div className="space-y-3">
                <Skeleton className="h-20 w-full" />
                <Skeleton className="h-16 w-full" />
                <Skeleton className="h-16 w-full" />
              </div>
            )}
          </Card>

          <div className="space-y-6">
            <Card title="What is deliberately wrong with it" subtitle="Each of these is a genuine property of the data and configuration, not a scripted output.">
              <ul className="space-y-2 text-sm leading-relaxed text-ink-700">
                {[
                  "The benchmark is 8% positive, so accuracy is almost uninformative.",
                  "Model B trained for 90 epochs against Model A's 25.",
                  "Both checkpoints were chosen on the test split.",
                  "The result comes from one favourable seed.",
                  "Model B's reported run overfits well before epoch 90.",
                  "A submitted variant diverges to NaN at its learning rate.",
                  "The reported checkpoint fails its integrity check every time.",
                ].map((item) => (
                  <li key={item} className="flex gap-2.5">
                    <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-warn-500" aria-hidden />
                    {item}
                  </li>
                ))}
              </ul>
            </Card>

            {config && (
              <Card
                title="Safe action registry"
                subtitle={`${config.actions.length} typed actions with validated parameters. An agent may name one; it can never issue a command.`}
              >
                <div className="space-y-3">
                  {INVOKERS.map(({ key, label, hint }) => {
                    const actions = config.actions.filter((a) => a.invoked_by === key);
                    if (actions.length === 0) return null;
                    return (
                      <div key={key}>
                        <p className="text-[11px] font-medium text-ink-600">
                          {label}
                          <span className="ml-1.5 font-normal text-ink-400">{hint}</span>
                        </p>
                        <ul className="mt-1.5 flex flex-wrap gap-1.5">
                          {actions.map((action) => (
                            <li key={action.name}>
                              <span
                                className="inline-block rounded bg-ink-50 px-2 py-1 font-mono text-[10px] text-ink-600"
                                title={`${action.summary} (${action.base_cost_units} units, needs ${action.min_autonomy} autonomy)`}
                              >
                                {action.name}
                              </span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    );
                  })}
                </div>
              </Card>
            )}
          </div>
        </div>

        {claims.length > 0 && (
          <div className="mt-8">
            <Card title="Earlier verifications">
              <ul className="divide-y divide-ink-100">
                {claims.map((claim) => (
                  <li key={claim.id}>
                    <button
                      type="button"
                      onClick={() => router.push(`/claims/${claim.id}`)}
                      className="flex w-full flex-wrap items-center gap-3 py-3 text-left transition hover:opacity-75"
                    >
                      <span className="min-w-0 flex-1 truncate text-sm text-ink-800">{claim.text}</span>
                      <Badge tone="neutral">{titleCase(claim.state)}</Badge>
                      <span className="text-xs text-ink-400">{relativeTime(claim.created_at)}</span>
                    </button>
                  </li>
                ))}
              </ul>
            </Card>
          </div>
        )}

        {config && (
          <p className="mt-8 text-xs leading-relaxed text-ink-500">
            Running against <span className="font-medium text-ink-700">{config.infrastructure.state_store}</span>,{" "}
            <span className="font-medium text-ink-700">{config.infrastructure.job_bus}</span> and{" "}
            <span className="font-medium text-ink-700">{config.infrastructure.reasoning}</span>.
            {config.infrastructure.mode === "demo" && (
              <>
                {" "}
                Demo mode runs the same orchestrator, worker and state machine as the cloud deployment, with
                in-process adapters instead of Firestore and Pub/Sub. Accelerator memory and utilisation figures
                come from an analytic model of the configuration and are labelled as simulated wherever shown;
                every metric, interval and health detection is computed from real training runs.
              </>
            )}
          </p>
        )}
      </main>
    </div>
  );
}
