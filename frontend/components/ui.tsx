"use client";

import { type ReactNode, useState } from "react";
import { type Tone, toneClass } from "@/lib/format";

export function Badge({ tone = "neutral", children }: { tone?: Tone; children: ReactNode }) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium ring-1 ring-inset ${toneClass(tone)}`}
    >
      {children}
    </span>
  );
}

export function LiveDot({ tone = "info" }: { tone?: Tone }) {
  const color =
    tone === "ok" ? "text-ok-500" : tone === "bad" ? "text-bad-500" : tone === "warn" ? "text-warn-500" : "text-info-500";
  return (
    <span className={`relative inline-flex h-2 w-2 ${color}`} aria-hidden>
      <span className="live-dot absolute inset-0 rounded-full" />
      <span className="relative inline-flex h-2 w-2 rounded-full bg-current" />
    </span>
  );
}

export function Card({
  title,
  subtitle,
  action,
  children,
  className = "",
}: {
  title?: ReactNode;
  subtitle?: ReactNode;
  action?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`card ${className}`}>
      {(title || action) && (
        <header className="flex items-start justify-between gap-4 border-b border-ink-100 px-5 py-4">
          <div className="min-w-0">
            {title && <h2 className="text-sm font-semibold tracking-tight text-ink-900">{title}</h2>}
            {subtitle && <p className="mt-1 text-xs leading-relaxed text-ink-500">{subtitle}</p>}
          </div>
          {action}
        </header>
      )}
      <div className="px-5 py-4">{children}</div>
    </section>
  );
}

export function EmptyState({ title, hint, icon = "·" }: { title: string; hint?: string; icon?: string }) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 rounded-lg border border-dashed border-ink-200 px-6 py-10 text-center">
      <span className="text-2xl text-ink-300" aria-hidden>
        {icon}
      </span>
      <p className="text-sm font-medium text-ink-700">{title}</p>
      {hint && <p className="max-w-sm text-xs leading-relaxed text-ink-500">{hint}</p>}
    </div>
  );
}

export function Skeleton({ className = "" }: { className?: string }) {
  return <div className={`skeleton rounded ${className}`} />;
}

export function ErrorState({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="rounded-lg border border-bad-500/25 bg-bad-50 px-5 py-4">
      <p className="text-sm font-semibold text-bad-700">Something went wrong</p>
      <p className="mt-1 text-xs leading-relaxed text-bad-700/80">{message}</p>
      {onRetry && (
        <button
          type="button"
          onClick={onRetry}
          className="mt-3 rounded-md bg-bad-700 px-3 py-1.5 text-xs font-medium text-white transition hover:bg-bad-700/90"
        >
          Try again
        </button>
      )}
    </div>
  );
}

export function Button({
  children,
  onClick,
  variant = "primary",
  disabled,
  type = "button",
  busy,
  className = "",
}: {
  children: ReactNode;
  onClick?: () => void;
  variant?: "primary" | "secondary" | "danger" | "ghost";
  disabled?: boolean;
  type?: "button" | "submit";
  busy?: boolean;
  className?: string;
}) {
  const styles = {
    primary: "bg-accent-600 text-white hover:bg-accent-700 focus-visible:outline-accent-600",
    secondary: "bg-white text-ink-700 ring-1 ring-inset ring-ink-200 hover:bg-ink-50",
    danger: "bg-white text-bad-700 ring-1 ring-inset ring-bad-500/30 hover:bg-bad-50",
    ghost: "text-ink-600 hover:bg-ink-100",
  }[variant];
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled || busy}
      className={`inline-flex items-center justify-center gap-2 rounded-md px-3.5 py-2 text-sm font-medium transition focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 disabled:cursor-not-allowed disabled:opacity-50 ${styles} ${className}`}
    >
      {busy && (
        <span
          className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-current border-t-transparent"
          aria-hidden
        />
      )}
      {children}
    </button>
  );
}

export function Stat({
  label,
  value,
  hint,
  tone,
}: {
  label: string;
  value: ReactNode;
  hint?: ReactNode;
  tone?: Tone;
}) {
  return (
    <div className="card px-4 py-3.5">
      <p className="text-[11px] font-medium uppercase tracking-wider text-ink-400">{label}</p>
      <div className="mt-1.5 flex items-baseline gap-2">
        <span className="tabular text-xl font-semibold tracking-tight text-ink-900">{value}</span>
        {tone && <span className={`h-1.5 w-1.5 rounded-full ${toneClass(tone).split(" ")[0]}`} aria-hidden />}
      </div>
      {hint && <p className="mt-1 text-xs leading-relaxed text-ink-500">{hint}</p>}
    </div>
  );
}

export function Disclosure({
  summary,
  children,
  defaultOpen = false,
}: {
  summary: ReactNode;
  children: ReactNode;
  defaultOpen?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-2 rounded px-1 py-1 text-left text-xs font-medium text-ink-600 transition hover:text-ink-900"
        aria-expanded={open}
      >
        <span className={`inline-block transition-transform ${open ? "rotate-90" : ""}`} aria-hidden>
          ›
        </span>
        {summary}
      </button>
      {open && <div className="mt-2">{children}</div>}
    </div>
  );
}

/** A labelled note for anything that is simulated rather than measured. */
export function SimulatedTag({ what }: { what: string }) {
  return (
    <span
      className="inline-flex items-center rounded border border-ink-200 bg-ink-50 px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide text-ink-500"
      title={`${what} comes from an analytic model of the configuration, not from a real accelerator.`}
    >
      simulated
    </span>
  );
}
