"use client";

import { useId } from "react";
import type { EpochRecord } from "@/lib/types";

interface Series {
  label: string;
  color: string;
  values: number[];
  dashed?: boolean;
}

function scale(values: number[]) {
  const finite = values.filter(Number.isFinite);
  if (finite.length === 0) return { min: 0, max: 1 };
  let min = Math.min(...finite);
  let max = Math.max(...finite);
  if (max - min < 1e-9) {
    min -= 0.5;
    max += 0.5;
  }
  const pad = (max - min) * 0.12;
  return { min: min - pad, max: max + pad };
}

/**
 * A compact multi-series line chart. Hand-rolled SVG so the curve renders
 * identically wherever the report is read, with no chart runtime to load.
 */
export function LineChart({
  series,
  xLabels,
  height = 190,
  yLabel,
  markers = [],
}: {
  series: Series[];
  xLabels: number[];
  height?: number;
  yLabel?: string;
  markers?: { x: number; label: string; color: string }[];
}) {
  const id = useId();
  const width = 640;
  const pad = { top: 14, right: 14, bottom: 26, left: 46 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const all = series.flatMap((s) => s.values);
  const { min, max } = scale(all);
  const n = Math.max(1, xLabels.length - 1);

  const x = (i: number) => pad.left + (i / n) * plotW;
  const y = (v: number) => pad.top + plotH - ((v - min) / (max - min)) * plotH;

  const path = (values: number[]) =>
    values
      .map((v, i) => (Number.isFinite(v) ? `${i === 0 ? "M" : "L"} ${x(i).toFixed(1)} ${y(v).toFixed(1)}` : ""))
      .filter(Boolean)
      .join(" ");

  const ticks = [0, 0.25, 0.5, 0.75, 1].map((t) => min + t * (max - min));
  const xTickEvery = Math.max(1, Math.ceil(xLabels.length / 8));

  return (
    <figure className="w-full">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="h-auto w-full"
        role="img"
        aria-label={`${series.map((s) => s.label).join(" and ")} over ${xLabels.length} epochs`}
      >
        {ticks.map((t, i) => (
          <g key={`${id}-y-${i}`}>
            <line
              x1={pad.left}
              x2={width - pad.right}
              y1={y(t)}
              y2={y(t)}
              stroke="var(--color-ink-100)"
              strokeWidth={1}
            />
            <text x={pad.left - 8} y={y(t) + 3.5} textAnchor="end" className="fill-ink-400 text-[9px]">
              {t.toFixed(t > 10 ? 0 : 2)}
            </text>
          </g>
        ))}

        {xLabels.map((label, i) =>
          i % xTickEvery === 0 || i === xLabels.length - 1 ? (
            <text key={`${id}-x-${i}`} x={x(i)} y={height - 8} textAnchor="middle" className="fill-ink-400 text-[9px]">
              {label}
            </text>
          ) : null,
        )}

        {markers.map((marker) => (
          <g key={`${id}-m-${marker.x}`}>
            <line
              x1={x(marker.x)}
              x2={x(marker.x)}
              y1={pad.top}
              y2={pad.top + plotH}
              stroke={marker.color}
              strokeWidth={1}
              strokeDasharray="3 3"
            />
            <text
              x={x(marker.x) + (x(marker.x) > pad.left + plotW * 0.6 ? -4 : 4)}
              y={pad.top + 10}
              textAnchor={x(marker.x) > pad.left + plotW * 0.6 ? "end" : "start"}
              className="text-[9px]"
              fill={marker.color}
            >
              {marker.label}
            </text>
          </g>
        ))}

        {series.map((s) => (
          <path
            key={`${id}-${s.label}`}
            d={path(s.values)}
            fill="none"
            stroke={s.color}
            strokeWidth={1.9}
            strokeLinejoin="round"
            strokeLinecap="round"
            strokeDasharray={s.dashed ? "4 3" : undefined}
          />
        ))}
      </svg>
      <figcaption className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-ink-500">
        {series.map((s) => (
          <span key={s.label} className="inline-flex items-center gap-1.5">
            <span className="h-0.5 w-4 rounded" style={{ background: s.color }} aria-hidden />
            {s.label}
          </span>
        ))}
        {yLabel && <span className="ml-auto text-ink-400">{yLabel}</span>}
      </figcaption>
    </figure>
  );
}

export function TrainingCurves({ curves }: { curves: EpochRecord[] }) {
  if (curves.length === 0) return null;
  const epochs = curves.map((c) => c.epoch);
  const valLosses = curves.map((c) => c.val_loss);
  const finite = valLosses.filter(Number.isFinite);
  const bestIdx = finite.length ? valLosses.indexOf(Math.min(...finite)) : -1;

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <div>
        <p className="mb-1 text-xs font-medium text-ink-600">Loss</p>
        <LineChart
          xLabels={epochs}
          yLabel="epoch"
          markers={
            bestIdx >= 0
              ? [{ x: bestIdx, label: `best val · epoch ${curves[bestIdx].epoch}`, color: "var(--color-ink-400)" }]
              : []
          }
          series={[
            { label: "train loss", color: "var(--color-accent-500)", values: curves.map((c) => c.train_loss) },
            { label: "validation loss", color: "var(--color-bad-500)", values: valLosses },
          ]}
        />
      </div>
      <div>
        <p className="mb-1 text-xs font-medium text-ink-600">Macro F1</p>
        <LineChart
          xLabels={epochs}
          yLabel="epoch"
          series={[
            { label: "train macro F1", color: "var(--color-accent-500)", values: curves.map((c) => c.train_metric) },
            {
              label: "validation macro F1",
              color: "var(--color-ok-500)",
              values: curves.map((c) => c.val_metric),
            },
          ]}
        />
      </div>
    </div>
  );
}

/** Horizontal paired-delta chart for the per-seed comparison. */
export function DeltaChart({
  deltas,
  seeds,
  label,
}: {
  deltas: number[];
  seeds: number[];
  label: string;
}) {
  const magnitude = Math.max(0.02, ...deltas.map((d) => Math.abs(d))) * 1.15;
  return (
    <div className="space-y-1.5">
      {deltas.map((delta, i) => {
        const widthPct = (Math.abs(delta) / magnitude) * 50;
        const positive = delta >= 0;
        return (
          <div key={seeds[i]} className="flex items-center gap-2">
            <span className="tabular w-14 shrink-0 text-[11px] text-ink-500">seed {seeds[i]}</span>
            <div className="relative h-4 flex-1 rounded bg-ink-50">
              <div className="absolute inset-y-0 left-1/2 w-px bg-ink-200" aria-hidden />
              <div
                className={`absolute inset-y-0.5 rounded ${positive ? "bg-ok-500" : "bg-bad-500"}`}
                style={
                  positive
                    ? { left: "50%", width: `${widthPct}%` }
                    : { right: "50%", width: `${widthPct}%` }
                }
              />
            </div>
            <span
              className={`tabular w-20 shrink-0 text-right text-[11px] font-medium ${positive ? "text-ok-700" : "text-bad-700"}`}
            >
              {delta >= 0 ? "+" : ""}
              {delta.toFixed(4)}
            </span>
          </div>
        );
      })}
      <p className="pt-1 text-[11px] text-ink-400">
        {label} · bars right of the centre line favour the candidate
      </p>
    </div>
  );
}

/** Score meter with a fixed band scale, so a number always reads the same. */
export function ScoreBar({ score, color }: { score: number; color: string }) {
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-ink-100">
      <div
        className="h-full rounded-full transition-[width] duration-500"
        style={{ width: `${Math.max(2, score)}%`, background: color }}
      />
    </div>
  );
}
