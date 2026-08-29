"""Markdown reliability report.

Rendered from stored state only, so the downloadable report and the dashboard
can never disagree.
"""

from __future__ import annotations

from collections.abc import Sequence

from .models.domain import (
    Claim,
    Evidence,
    Job,
    LedgerEntry,
    Loophole,
    ReliabilityScore,
    Subclaim,
    Verdict,
)
from .models.enums import EvidenceStance, JobState


def _check_mark(passed: bool | None) -> str:
    return {True: "pass", False: "FAIL", None: "not evaluated"}[passed]


def render_report(
    claim: Claim,
    verdict: Verdict,
    score: ReliabilityScore,
    subclaims: Sequence[Subclaim],
    loopholes: Sequence[Loophole],
    jobs: Sequence[Job],
    evidence: Sequence[Evidence],
    ledger: Sequence[LedgerEntry],
) -> str:
    lines: list[str] = []
    add = lines.append

    add("# LabGuard AI reliability report")
    add("")
    add(f"**Claim.** {claim.text}")
    add("")
    add(f"**Verdict: {verdict.status.value.replace('_', ' ')}.** {verdict.headline}")
    add("")
    add(verdict.narrative)
    add("")
    add(
        f"Reasoning backend: `{claim.reasoning_backend}` · "
        f"Autonomy: `{claim.autonomy_mode.value}` · "
        f"Budget: {claim.budget.consumed_units:.2f} of {claim.budget.total_units:.2f} units consumed · "
        f"Planning rounds: {claim.planning_round + 1}"
    )
    if claim.halt_reason:
        add("")
        add(f"> Execution halted early: {claim.halt_reason}")

    add("")
    add("## Reliability score")
    add("")
    add("| Dimension | Score | How it was computed |")
    add("| --- | ---: | --- |")
    for dimension in score.dimensions:
        add(f"| {dimension.dimension.value.replace('_', ' ')} | {dimension.score} | {dimension.calculation} |")

    add("")
    add("### Checks behind each score")
    for dimension in score.dimensions:
        add("")
        add(f"**{dimension.dimension.value.replace('_', ' ')} - {dimension.score}**")
        add("")
        for check in dimension.checks:
            add(
                f"- `{check.id}` (weight {check.weight:g}) - **{_check_mark(check.passed)}** - {check.label}. {check.detail}"
            )

    add("")
    add("## Subclaims")
    add("")
    add("| Subclaim | Status | Confidence | Measured quantity |")
    add("| --- | --- | ---: | --- |")
    for subclaim in subclaims:
        add(
            f"| {subclaim.statement} | {subclaim.status.value} | {subclaim.confidence:.2f} | "
            f"{subclaim.measurable_quantity} |"
        )

    add("")
    add("## Scientific loopholes")
    add("")
    add("| Loophole | Severity | Status | Resolution |")
    add("| --- | ---: | --- | --- |")
    for hole in loopholes:
        add(
            f"| {hole.title} (`{hole.kind.value}`) | {hole.severity:.2f} | {hole.status.value} | {hole.resolution or '-'} |"
        )

    add("")
    add("## Evidence")
    for stance, heading in (
        (EvidenceStance.CONTRADICTS, "Contradicting"),
        (EvidenceStance.SUPPORTS, "Supporting"),
        (EvidenceStance.NEUTRAL, "Contextual"),
    ):
        items = [e for e in evidence if e.stance == stance]
        if not items:
            continue
        add("")
        add(f"### {heading}")
        for item in items:
            add(f"- {item.statement} _(strength {item.strength:.2f})_")

    add("")
    add("## Experiments run")
    add("")
    add("| Action | State | Attempts | Cost (units) | Health |")
    add("| --- | --- | ---: | ---: | --- |")
    for job in jobs:
        add(
            f"| `{job.action_type}` | {job.state.value} | {job.attempts} | {job.actual_cost_units:.2f} | "
            f"{job.health.status.value} |"
        )

    incidents = verdict.run_health_incidents
    add("")
    add("## Run-health incidents")
    if incidents:
        for incident in incidents:
            add(f"- {incident}")
    else:
        add("")
        add("No run-health incidents were recorded.")

    add("")
    add("## Remaining uncertainty")
    if verdict.remaining_uncertainty:
        for item in verdict.remaining_uncertainty:
            add(f"- {item}")
    else:
        add("")
        add("Nothing material was left open.")

    add("")
    add("## Reproducibility")
    add("")
    repro = verdict.reproducibility
    add(f"- Seeds tested: {repro.get('seeds_tested')}")
    add(f"- Training budget equalised: {repro.get('equalised_training_budget')}")
    add(f"- Checkpoint selection: {repro.get('checkpoint_selection')}")
    dataset = repro.get("dataset", {})
    add(
        f"- Dataset: {dataset.get('name')} ({dataset.get('n_samples')} rows, "
        f"{dataset.get('n_features')} features, {float(dataset.get('positive_rate', 0)):.1%} positive)"
    )
    for config in repro.get("configurations", []):
        add(
            f"- Configuration `{config.get('name')}`: {config.get('family')}, "
            f"{config.get('epochs')} epochs, lr {config.get('learning_rate')}, "
            f"batch {config.get('batch_size')}, class weight {config.get('class_weight')}"
        )

    add("")
    add("## Audit trail")
    add("")
    add("| # | Agent | Action | Decision |")
    add("| ---: | --- | --- | --- |")
    for entry in ledger:
        decision = (entry.decision or entry.reason).replace("|", "/")
        add(f"| {entry.sequence} | {entry.agent.value} | `{entry.action}` | {decision} |")

    failed = [j for j in jobs if j.state in (JobState.FAILED, JobState.BLOCKED_LOOP)]
    if failed:
        add("")
        add("## Runs that did not complete")
        for job in failed:
            add(f"- `{job.action_type}` ended as **{job.state.value}** after {job.attempts} attempt(s): {job.error}")

    add("")
    add("---")
    add("")
    add(
        "Every score in this report is a weighted pass rate over the named checks listed above. "
        "Metrics, confidence intervals and health detections are computed in Python; the language "
        "model contributes wording and prioritisation only, and its proposed verdict status is "
        "overridden by the measured one when they disagree."
    )
    return "\n".join(lines)
