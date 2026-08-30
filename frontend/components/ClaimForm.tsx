"use client";

import { useState } from "react";
import { Button, Card } from "./ui";
import type { AutonomyMode, ClaimContext } from "@/lib/types";

/**
 * Submit your own claim.
 *
 * A deliberate limitation, stated on the form itself: LabGuard rebuilds a
 * synthetic benchmark with the shape you describe and trains the two
 * configurations you give it. It does not read your real dataset or attach to
 * your training code. What it verifies is the *reasoning* — whether a
 * difference of the size you report survives equal budgets, more seeds,
 * class-balanced metrics and honest checkpointing.
 */

interface ArmState {
  name: string;
  family: "linear" | "mlp";
  hidden_units: number;
  epochs: number;
  learning_rate: number;
  class_weight: "none" | "sqrt_balanced" | "balanced";
}

const BASELINE: ArmState = {
  name: "Baseline",
  family: "linear",
  hidden_units: 0,
  epochs: 25,
  learning_rate: 0.1,
  class_weight: "sqrt_balanced",
};

const CANDIDATE: ArmState = {
  name: "Candidate",
  family: "mlp",
  hidden_units: 24,
  epochs: 90,
  learning_rate: 0.2,
  class_weight: "none",
};

export interface CustomClaim {
  text: string;
  context: ClaimContext;
  autonomy_mode: AutonomyMode;
}

export function ClaimForm({
  autonomy,
  onSubmit,
  submitting,
}: {
  autonomy: AutonomyMode;
  onSubmit: (claim: CustomClaim) => void;
  submitting: boolean;
}) {
  const [text, setText] = useState("");
  const [samples, setSamples] = useState(4000);
  const [features, setFeatures] = useState(24);
  const [positiveRate, setPositiveRate] = useState(8);
  const [baseline, setBaseline] = useState<ArmState>(BASELINE);
  const [candidate, setCandidate] = useState<ArmState>(CANDIDATE);
  const [metric, setMetric] = useState("accuracy");
  const [reported, setReported] = useState(0.91);
  const [seed, setSeed] = useState(11);
  const [selectedOn, setSelectedOn] = useState("test");
  const [error, setError] = useState<string | null>(null);

  function submit() {
    if (text.trim().length < 8) {
      setError("Write the claim you want verified, in a sentence.");
      return;
    }
    if (!baseline.name.trim() || !candidate.name.trim()) {
      setError("Both arms need a name.");
      return;
    }
    if (baseline.name.trim() === candidate.name.trim()) {
      setError("The two arms need different names.");
      return;
    }
    setError(null);
    onSubmit({
      text: text.trim(),
      autonomy_mode: autonomy,
      context: {
        dataset: {
          name: "custom_benchmark",
          n_samples: samples,
          n_features: features,
          positive_rate: positiveRate / 100,
          test_fraction: 0.25,
          inject_train_test_overlap: 0,
          domain_shift_strength: 0,
        },
        models: [
          { ...baseline, batch_size: 64, objective: "bce", is_baseline: true, role: "primary", notes: "" },
          { ...candidate, batch_size: 64, objective: "bce", is_baseline: false, role: "primary", notes: "" },
        ],
        existing_results: [
          {
            model_name: candidate.name.trim(),
            metric,
            value: reported,
            seed,
            checkpoint_selected_on: selectedOn,
            epochs_trained: candidate.epochs,
            checkpoint_uri: "",
          },
        ],
        reported_checkpoint_corrupt: false,
        notes: `Reported on seed ${seed}. Checkpoint selected on ${selectedOn}.`,
      },
    });
  }

  return (
    <Card
      title="Verify your own claim"
      subtitle="Describe the comparison you want challenged. Everything below feeds the loophole detector."
    >
      <p className="mb-4 rounded-lg border border-info-500/25 bg-info-50 px-4 py-3 text-xs leading-relaxed text-info-700">
        <span className="font-semibold">What this does and does not do.</span> LabGuard rebuilds a synthetic
        benchmark with the shape you describe and trains these two configurations itself. It does not read your
        real dataset or attach to your training code. What it verifies is the reasoning: whether a difference of
        the size you report would survive equal budgets, more seeds, class-balanced metrics and honest
        checkpointing.
      </p>

      <Field label="The claim" hint="One falsifiable sentence, as you would write it in a paper.">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={2}
          placeholder="Our re-ranker beats the BM25 baseline on the support-ticket triage set."
          className="w-full rounded-lg border border-ink-200 px-3 py-2 text-sm text-ink-900 outline-none placeholder:text-ink-300 focus:border-accent-500"
        />
      </Field>

      <Group title="Dataset">
        <div className="grid gap-3 sm:grid-cols-3">
          <NumberField label="Rows" value={samples} min={500} max={20000} step={500} onChange={setSamples} />
          <NumberField label="Features" value={features} min={4} max={128} onChange={setFeatures} />
          <NumberField
            label="Positive class %"
            value={positiveRate}
            min={1}
            max={50}
            step={1}
            onChange={setPositiveRate}
            hint="below 25% triggers the imbalance checks"
          />
        </div>
      </Group>

      <Group title="The two arms">
        <div className="grid gap-4 lg:grid-cols-2">
          <Arm label="Baseline" arm={baseline} onChange={setBaseline} />
          <Arm label="Candidate" arm={candidate} onChange={setCandidate} />
        </div>
      </Group>

      <Group title="The result you already have">
        <div className="grid gap-3 sm:grid-cols-4">
          <Select
            label="Metric"
            value={metric}
            onChange={setMetric}
            options={[
              ["accuracy", "accuracy"],
              ["macro_f1", "macro F1"],
              ["balanced_accuracy", "balanced accuracy"],
            ]}
          />
          <NumberField label="Reported value" value={reported} min={0} max={1} step={0.001} onChange={setReported} />
          <NumberField label="Seed" value={seed} min={0} max={9999} onChange={setSeed} />
          <Select
            label="Checkpoint chosen on"
            value={selectedOn}
            onChange={setSelectedOn}
            options={[
              ["test", "test split"],
              ["validation", "validation split"],
              ["last", "last epoch"],
            ]}
          />
        </div>
      </Group>

      {error && <p className="mt-3 text-xs font-medium text-bad-700">{error}</p>}

      <div className="mt-5">
        <Button onClick={submit} busy={submitting} className="w-full">
          Verify this claim
        </Button>
      </div>
    </Card>
  );
}

function Group({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <fieldset className="mt-5 border-t border-ink-100 pt-4">
      <legend className="sr-only">{title}</legend>
      <p className="mb-2.5 text-[11px] font-medium uppercase tracking-wider text-ink-400">{title}</p>
      {children}
    </fieldset>
  );
}

function Field({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <span className="text-[11px] font-medium uppercase tracking-wider text-ink-400">{label}</span>
      {hint && <span className="mt-0.5 block text-[11px] text-ink-400">{hint}</span>}
      <span className="mt-1.5 block">{children}</span>
    </label>
  );
}

const INPUT =
  "w-full rounded-lg border border-ink-200 px-3 py-2 text-sm text-ink-900 outline-none focus:border-accent-500";

function NumberField({
  label,
  value,
  onChange,
  min,
  max,
  step = 1,
  hint,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  step?: number;
  hint?: string;
}) {
  return (
    <Field label={label} hint={hint}>
      <input
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e) => {
          const next = e.target.valueAsNumber;
          if (!Number.isNaN(next)) onChange(Math.min(max, Math.max(min, next)));
        }}
        className={`tabular ${INPUT}`}
      />
    </Field>
  );
}

function Select({
  label,
  value,
  onChange,
  options,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  options: [string, string][];
}) {
  return (
    <Field label={label}>
      <select value={value} onChange={(e) => onChange(e.target.value)} className={INPUT}>
        {options.map(([v, l]) => (
          <option key={v} value={v}>
            {l}
          </option>
        ))}
      </select>
    </Field>
  );
}

function Arm({
  label,
  arm,
  onChange,
}: {
  label: string;
  arm: ArmState;
  onChange: (a: ArmState) => void;
}) {
  const set = <K extends keyof ArmState>(key: K, value: ArmState[K]) => onChange({ ...arm, [key]: value });
  return (
    <div className="rounded-lg border border-ink-100 px-4 py-3.5">
      <p className="mb-2.5 text-xs font-semibold text-ink-800">{label}</p>
      <div className="space-y-3">
        <Field label="Name">
          <input value={arm.name} onChange={(e) => set("name", e.target.value)} className={INPUT} />
        </Field>
        <div className="grid grid-cols-2 gap-3">
          <Select
            label="Family"
            value={arm.family}
            onChange={(v) => set("family", v as ArmState["family"])}
            options={[
              ["linear", "linear"],
              ["mlp", "MLP"],
            ]}
          />
          <NumberField
            label="Hidden units"
            value={arm.hidden_units}
            min={0}
            max={128}
            onChange={(v) => set("hidden_units", v)}
          />
        </div>
        <div className="grid grid-cols-2 gap-3">
          <NumberField label="Epochs" value={arm.epochs} min={5} max={200} step={5} onChange={(v) => set("epochs", v)} />
          <NumberField
            label="Learning rate"
            value={arm.learning_rate}
            min={0.001}
            max={1}
            step={0.01}
            onChange={(v) => set("learning_rate", v)}
          />
        </div>
        <Select
          label="Class weighting"
          value={arm.class_weight}
          onChange={(v) => set("class_weight", v as ArmState["class_weight"])}
          options={[
            ["none", "none"],
            ["sqrt_balanced", "sqrt inverse frequency"],
            ["balanced", "inverse frequency"],
          ]}
        />
      </div>
    </div>
  );
}
