"""RunMedic's detection logic.

Pure functions over recorded epochs and job history, so every detection is
reproducible and unit-testable.  The language model is not involved in
deciding whether a run is unhealthy — it only narrates what these functions
found.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..models.domain import EpochRecord, Job
from ..models.enums import AnomalyKind, HealthStatus


def _fmt(value: float) -> str:
    """Format a loss for a human. A diverging run produces enormous numbers,
    so fall back to exponential notation rather than printing 200 digits."""
    if value != value:
        return "NaN"
    magnitude = abs(value)
    if magnitude >= 1e4 or (0 < magnitude < 1e-4):
        return f"{value:.3e}"
    return f"{value:.4f}"


#: Consecutive epochs of validation degradation before overfitting is called.
OVERFIT_PATIENCE = 6
#: Relative rise in validation loss from its best that counts as meaningful.
OVERFIT_MIN_RISE = 0.04
#: Relative loss improvement below which training counts as stalled.
STALL_TOLERANCE = 1e-4
STALL_WINDOW = 8
#: A run whose best validation metric never clears this is underfitting.
UNDERFIT_METRIC = 0.35
#: Retries beyond this on one job trigger escalation.
MAX_TOTAL_ATTEMPTS = 3
#: Identical failure signatures in a row that mean recovery is not working.
RECOVERY_LOOP_THRESHOLD = 2


@dataclass
class Finding:
    anomaly: AnomalyKind
    status: HealthStatus
    detail: str
    epoch: int | None = None
    evidence: dict[str, Any] | None = None


def analyse_curve(curves: Sequence[EpochRecord]) -> list[Finding]:
    """Classify a training curve. Returns every finding, most severe first."""
    findings: list[Finding] = []
    if not curves:
        return findings

    val_losses = [c.val_loss for c in curves]
    train_losses = [c.train_loss for c in curves]

    if any(v != v for v in val_losses + train_losses):  # NaN check
        bad = next(c for c in curves if c.val_loss != c.val_loss or c.train_loss != c.train_loss)
        findings.append(
            Finding(
                AnomalyKind.NAN_LOSS,
                HealthStatus.CRITICAL,
                f"loss became non-finite at epoch {bad.epoch}",
                epoch=bad.epoch,
            )
        )
        return findings

    best_idx = min(range(len(val_losses)), key=lambda i: val_losses[i])
    best_val = val_losses[best_idx]
    last_val = val_losses[-1]

    # Exploding loss: validation loss ends far above where it started.
    if last_val > val_losses[0] * 3 and last_val > best_val * 3:
        findings.append(
            Finding(
                AnomalyKind.EXPLODING_LOSS,
                HealthStatus.CRITICAL,
                f"validation loss grew from {_fmt(val_losses[0])} to {_fmt(last_val)}",
                epoch=curves[-1].epoch,
                evidence={"first": val_losses[0], "last": last_val},
            )
        )

    # Overfitting: train loss still falling while validation loss climbs away
    # from its best for a sustained window.
    degrading = len(val_losses) - 1 - best_idx
    rise = (last_val - best_val) / best_val if best_val > 0 else 0.0
    train_improved = train_losses[-1] < train_losses[best_idx] - 1e-6
    if degrading >= OVERFIT_PATIENCE and rise >= OVERFIT_MIN_RISE and train_improved:
        findings.append(
            Finding(
                AnomalyKind.OVERFITTING,
                HealthStatus.WARNING,
                (
                    f"validation loss has risen {rise * 100:.1f}% above its best "
                    f"(epoch {curves[best_idx].epoch}) over {degrading} epochs while "
                    f"training loss kept falling ({_fmt(train_losses[best_idx])} -> {_fmt(train_losses[-1])})"
                ),
                epoch=curves[-1].epoch,
                evidence={
                    "best_epoch": curves[best_idx].epoch,
                    "best_val_loss": round(best_val, 5),
                    "last_val_loss": round(last_val, 5),
                    "epochs_degrading": degrading,
                    "train_loss_at_best": round(train_losses[best_idx], 5),
                    "train_loss_last": round(train_losses[-1], 5),
                },
            )
        )

    # Stalled training: no meaningful movement over a trailing window.
    if len(curves) >= STALL_WINDOW:
        window = train_losses[-STALL_WINDOW:]
        movement = abs(window[0] - window[-1]) / max(abs(window[0]), 1e-9)
        # A flat tail on a run that never learned is a stall; a flat tail on a
        # run that already improved is convergence.
        total_progress = (train_losses[0] - train_losses[-1]) / max(abs(train_losses[0]), 1e-9)
        if movement < STALL_TOLERANCE and total_progress < 0.05:
            findings.append(
                Finding(
                    AnomalyKind.STALLED_TRAINING,
                    HealthStatus.WARNING,
                    f"training loss moved {movement:.2e} over the last {STALL_WINDOW} epochs",
                    epoch=curves[-1].epoch,
                )
            )

    # Underfitting: the model never became useful at all.
    best_metric = max(c.val_metric for c in curves)
    if best_metric < UNDERFIT_METRIC:
        findings.append(
            Finding(
                AnomalyKind.UNDERFITTING,
                HealthStatus.WARNING,
                f"best validation macro F1 across the run was only {best_metric:.3f}",
                epoch=curves[-1].epoch,
            )
        )

    # Unusually slow epochs relative to the run's own median.
    seconds = sorted(c.seconds for c in curves)
    median = seconds[len(seconds) // 2]
    slow = [c for c in curves if median > 0 and c.seconds > median * 4]
    if slow:
        findings.append(
            Finding(
                AnomalyKind.SLOW_EPOCH,
                HealthStatus.WARNING,
                f"{len(slow)} epochs took more than 4x the median epoch time",
                epoch=slow[0].epoch,
            )
        )

    severity = {HealthStatus.CRITICAL: 0, HealthStatus.WARNING: 1, HealthStatus.RECOVERED: 2, HealthStatus.HEALTHY: 3}
    return sorted(findings, key=lambda f: severity.get(f.status, 9))


def should_stop_now(curves: Sequence[EpochRecord], findings: Sequence[Finding]) -> bool:
    """Whether a live run has nothing left to gain by continuing.

    Detecting overfitting is not on its own a reason to stop: validation loss
    can rise while the validation *metric* is still improving. A run is only
    stopped once both have turned, which means the checkpoint worth keeping is
    already behind us.
    """
    if not any(f.anomaly in (AnomalyKind.OVERFITTING, AnomalyKind.EXPLODING_LOSS) for f in findings):
        return False
    if len(curves) < OVERFIT_PATIENCE + 1:
        return False
    metrics = [c.val_metric for c in curves]
    best_idx = max(range(len(metrics)), key=lambda i: metrics[i])
    return (len(metrics) - 1 - best_idx) >= OVERFIT_PATIENCE


def classify_failure(job: Job, signature: str) -> Finding:
    """Decide whether a failed job is repairable or stuck in a loop."""
    repeats = sum(1 for s in job.recovery_actions if s.startswith(f"failure:{signature}"))
    if repeats >= RECOVERY_LOOP_THRESHOLD:
        return Finding(
            AnomalyKind.RECOVERY_LOOP,
            HealthStatus.CRITICAL,
            (
                f"the same failure ({signature.split(':')[0]}) recurred after "
                f"{repeats} recovery attempts; automatic execution paused to "
                f"avoid burning budget on an unchanging failure"
            ),
        )
    if job.attempts > MAX_TOTAL_ATTEMPTS:
        return Finding(
            AnomalyKind.EXCESSIVE_RETRIES,
            HealthStatus.CRITICAL,
            f"job exceeded its retry limit after {job.attempts} attempts",
        )
    kind = signature.split(":")[0]
    try:
        anomaly = AnomalyKind(kind)
    except ValueError:
        anomaly = AnomalyKind.MISSING_ARTIFACT
    return Finding(anomaly, HealthStatus.CRITICAL, str(job.error or kind))


def summarise(findings: Sequence[Finding]) -> tuple[HealthStatus, str]:
    if not findings:
        return HealthStatus.HEALTHY, "no anomalies detected in the recorded curve"
    worst = findings[0]
    return worst.status, worst.detail
