"""Reliability scoring.

Scores are computed here and only here.  Each dimension is a weighted set of
named boolean checks whose inputs come from job results, health events and
evidence records.  The score is the weighted pass rate, and the checks that
produced it are returned alongside so the dashboard can show the arithmetic.

A language model is never asked for a number.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..models.domain import (
    Claim,
    DimensionScore,
    Evidence,
    Job,
    Loophole,
    ReliabilityScore,
    ScoreCheck,
    Subclaim,
)
from ..models.enums import (
    AnomalyKind,
    EvidenceStance,
    HealthStatus,
    JobState,
    LoopholeStatus,
    ScoreDimension,
    SubclaimStatus,
)

#: Metric the headline stability checks are read from.
HEADLINE_METRIC = "macro_f1"


def _completed(jobs: Sequence[Job], action: str) -> Job | None:
    for job in jobs:
        if job.action_type == action and job.state == JobState.COMPLETED:
            return job
    return None


def _score_from(checks: list[ScoreCheck]) -> tuple[int, str]:
    """Weighted pass rate over the checks that were actually evaluated."""
    evaluated = [c for c in checks if c.passed is not None]
    if not evaluated:
        return 0, "no checks could be evaluated yet"
    total = sum(c.weight for c in evaluated)
    passed = sum(c.weight for c in evaluated if c.passed)
    score = round(100 * passed / total) if total else 0
    skipped = len(checks) - len(evaluated)
    tail = f"; {skipped} check(s) not yet evaluated" if skipped else ""
    return score, f"{passed:g} of {total:g} weighted checks passed{tail}"


def _reproducibility(claim: Claim, jobs: Sequence[Job]) -> list[ScoreCheck]:
    seed_job = _completed(jobs, "run_seed_comparison")
    seeds = list((seed_job.result.get("seeds") if seed_job else []) or [])
    checkpoint_job = next((j for j in jobs if j.action_type == "resume_from_checkpoint"), None)
    ckpt_ok: bool | None = None
    ckpt_detail = "the reported checkpoint has not been verified"
    if checkpoint_job is not None:
        if checkpoint_job.state == JobState.COMPLETED:
            ckpt_ok = bool(checkpoint_job.result.get("reproduces_within_tolerance"))
            ckpt_detail = (
                f"reported {checkpoint_job.result.get('reported_value')} vs reproduced "
                f"{checkpoint_job.result.get('reproduced_value')}"
            )
        elif checkpoint_job.state in (JobState.FAILED, JobState.BLOCKED_LOOP):
            ckpt_ok = False
            ckpt_detail = f"verification could not complete: {checkpoint_job.error or checkpoint_job.state.value}"

    original_seeds = {r.seed for r in claim.context.existing_results}
    return [
        ScoreCheck(
            id="repro.multiple_seeds",
            label="Result reproduced on 3 or more seeds",
            passed=len(seeds) >= 3 if seed_job else None,
            weight=2.0,
            detail=f"{len(seeds)} seeds retrained" if seed_job else "no seed comparison has run",
            observed={"seeds": seeds},
        ),
        ScoreCheck(
            id="repro.original_not_single_seed",
            label="The original result was not reported from a single seed",
            passed=len(original_seeds) > 1,
            weight=1.5,
            detail=f"the submission reports {len(original_seeds)} distinct seed(s): {sorted(original_seeds)}",
            observed={"reported_seeds": sorted(original_seeds)},
        ),
        ScoreCheck(
            id="repro.checkpoint_verified",
            label="The reported checkpoint reproduces its reported number",
            passed=ckpt_ok,
            weight=1.5,
            detail=ckpt_detail,
        ),
        ScoreCheck(
            id="repro.configs_recorded",
            label="Full training configurations were captured for both arms",
            passed=len(claim.context.models) >= 2,
            weight=1.0,
            detail=f"{len(claim.context.models)} configuration(s) supplied with the claim",
        ),
    ]


def _data_integrity(claim: Claim, jobs: Sequence[Job]) -> list[ScoreCheck]:
    overlap_job = _completed(jobs, "check_data_overlap")
    result = overlap_job.result if overlap_job else {}
    control = (result or {}).get("positive_control", {})
    overlapping = result.get("overlapping_rows")
    return [
        ScoreCheck(
            id="data.no_train_test_overlap",
            label="No test row duplicates a training row",
            passed=(overlapping == 0) if overlap_job else None,
            weight=3.0,
            detail=(
                f"{overlapping} overlapping rows across {result.get('n_test')} test rows"
                if overlap_job
                else "the overlap check has not run"
            ),
            observed={"overlapping_rows": overlapping},
        ),
        ScoreCheck(
            id="data.detector_verified",
            label="The leakage detector was validated on a positive control",
            passed=bool(control.get("detector_working")) if control else None,
            weight=1.0,
            detail=(
                f"{control.get('detected_rows')} of {control.get('injected_rows')} injected duplicates recovered"
                if control
                else "no positive control was run"
            ),
        ),
        ScoreCheck(
            id="data.imbalance_documented",
            label="Class balance is documented with the claim",
            passed=claim.context.dataset.positive_rate > 0,
            weight=1.0,
            detail=f"positive rate {claim.context.dataset.positive_rate:.1%}",
        ),
    ]


def _baseline_fairness(claim: Claim, jobs: Sequence[Job]) -> list[ScoreCheck]:
    config_job = _completed(jobs, "compare_configurations")
    cfg = config_job.result if config_job else {}
    ratio = cfg.get("training_budget_ratio")
    on_test = cfg.get("models_selected_on_test")
    seed_job = _completed(jobs, "run_seed_comparison")
    threshold_job = _completed(jobs, "sweep_decision_threshold")
    return [
        ScoreCheck(
            id="fairness.equal_training_budget",
            label="Both arms were trained under the same budget",
            passed=(abs((ratio or 1.0) - 1.0) < 0.2) if config_job else None,
            weight=2.0,
            detail=(
                f"candidate/baseline epoch ratio is {ratio}" if config_job else "configurations have not been compared"
            ),
            observed={"training_budget_ratio": ratio},
        ),
        ScoreCheck(
            id="fairness.honest_checkpoint_selection",
            label="No checkpoint was selected on the test split",
            passed=(len(on_test) == 0) if config_job else None,
            weight=2.0,
            detail=(
                f"selected on test: {', '.join(on_test) if on_test else 'none'}"
                if config_job
                else "configurations have not been compared"
            ),
            observed={"models_selected_on_test": on_test},
        ),
        ScoreCheck(
            id="fairness.rerun_under_equal_budget",
            label="A corrected, budget-equalised comparison was actually run",
            passed=bool(seed_job and seed_job.result.get("equalised_training_budget")) if seed_job else None,
            weight=1.5,
            detail=(
                f"both arms retrained for {seed_job.result.get('epochs_per_arm')} epochs"
                if seed_job
                else "no equalised comparison has run"
            ),
        ),
        ScoreCheck(
            id="fairness.threshold_per_arm",
            label="Each arm was compared at its own best decision threshold",
            passed=(not threshold_job.result.get("conclusion_changes")) if threshold_job else None,
            weight=1.0,
            detail=(
                "the ordering survives per-arm threshold tuning"
                if threshold_job and not threshold_job.result.get("conclusion_changes")
                else (
                    "the winner changes once each arm gets its own threshold"
                    if threshold_job
                    else "no threshold sweep has run"
                )
            ),
        ),
    ]


def _statistical_stability(jobs: Sequence[Job]) -> list[ScoreCheck]:
    seed_job = _completed(jobs, "run_seed_comparison")
    summaries = (seed_job.result.get("paired_summary") if seed_job else {}) or {}
    headline = summaries.get(HEADLINE_METRIC) or next(iter(summaries.values()), None)
    metrics_job = _completed(jobs, "recalculate_metrics")
    overlapping_cis: bool | None = None
    if metrics_job:
        comparison = metrics_job.result.get("metrics", {})
        overlapping_cis = any(
            v.get("baseline_ci", {}).get("high", 1) >= v.get("candidate_ci", {}).get("low", 0)
            for v in comparison.values()
        )
    consistent = None
    if headline:
        n = headline.get("n_seeds", 0)
        consistent = headline.get("wins_for_b", 0) == n and n >= 3

    return [
        ScoreCheck(
            id="stats.ci_excludes_zero",
            label=f"The {HEADLINE_METRIC} difference has an interval excluding zero in the claim's favour",
            passed=(not headline.get("ci_includes_zero") and headline.get("mean_delta", 0) > 0) if headline else None,
            weight=3.0,
            detail=(
                f"mean delta {headline.get('mean_delta'):+.4f}, 95% interval "
                f"[{headline.get('ci_low'):+.4f}, {headline.get('ci_high'):+.4f}]"
                if headline
                else "no seed comparison has run"
            ),
            observed=headline or {},
        ),
        ScoreCheck(
            id="stats.consistent_across_seeds",
            label="The candidate wins on every seed tested",
            passed=consistent,
            weight=2.0,
            detail=(
                f"won {headline.get('wins_for_b')} of {headline.get('n_seeds')} seeds"
                if headline
                else "no seed comparison has run"
            ),
        ),
        ScoreCheck(
            id="stats.bootstrap_separation",
            label="Single-seed bootstrap intervals separate the two arms",
            passed=(not overlapping_cis) if metrics_job else None,
            weight=1.5,
            detail=(
                "bootstrap intervals overlap on at least one metric"
                if overlapping_cis
                else ("bootstrap intervals are disjoint" if metrics_job else "metrics have not been recomputed")
            ),
        ),
    ]


def _training_health(jobs: Sequence[Job]) -> list[ScoreCheck]:
    events = [e for job in jobs for e in job.health.events]
    critical = [e for e in events if e.status == HealthStatus.CRITICAL]
    loops = [e for e in events if e.anomaly == AnomalyKind.RECOVERY_LOOP]
    # Scoped to the diagnostic replay of the submitted run: overfitting caught
    # and early-stopped inside a verification run is the system working, not a
    # defect in the claim.
    overfit = [
        e
        for j in jobs
        if j.action_type == "inspect_training_curve"
        for e in j.health.events
        if e.anomaly == AnomalyKind.OVERFITTING
    ]
    unrecovered = [j for j in jobs if j.state in (JobState.FAILED, JobState.BLOCKED_LOOP)]
    recovered = [j for j in jobs if j.recovery_actions and j.state == JobState.COMPLETED]
    return [
        ScoreCheck(
            id="health.no_unrecovered_failures",
            label="Every launched run reached a usable end state",
            passed=len(unrecovered) == 0,
            weight=2.0,
            detail=(
                f"{len(unrecovered)} run(s) ended without a result: "
                + ", ".join(f"{j.action_type} ({j.state.value})" for j in unrecovered)
                if unrecovered
                else f"{len(jobs)} run(s) completed"
            ),
        ),
        ScoreCheck(
            id="health.no_overfitting_in_reported_run",
            label="The reported run does not overfit",
            passed=len(overfit) == 0 if jobs else None,
            weight=2.0,
            detail=(overfit[0].detail if overfit else "no overfitting detected in any inspected curve"),
        ),
        ScoreCheck(
            id="health.no_recovery_loops",
            label="No repair loop had to be broken",
            passed=len(loops) == 0,
            weight=1.5,
            detail=(loops[0].detail if loops else "no recovery loops detected"),
        ),
        ScoreCheck(
            id="health.incidents_were_repaired",
            label="Detected incidents were repaired rather than left open",
            passed=(len(recovered) > 0 or len(critical) == 0),
            weight=1.0,
            detail=f"{len(recovered)} run(s) recovered, {len(critical)} critical incident(s) recorded",
        ),
    ]


def _evidence_completeness(
    claim: Claim,
    subclaims: Sequence[Subclaim],
    loopholes: Sequence[Loophole],
    evidence: Sequence[Evidence],
    jobs: Sequence[Job],
) -> list[ScoreCheck]:
    tested = [s for s in subclaims if s.status != SubclaimStatus.UNTESTED]
    resolved = [h for h in loopholes if h.status in (LoopholeStatus.CONFIRMED, LoopholeStatus.REFUTED)]
    classwise = _completed(jobs, "evaluate_classwise")
    budget_left = claim.budget.remaining_units
    return [
        ScoreCheck(
            id="evidence.subclaims_tested",
            label="Every testable subclaim has at least one piece of evidence",
            passed=(len(tested) == len(subclaims)) if subclaims else None,
            weight=2.0,
            detail=f"{len(tested)} of {len(subclaims)} subclaims tested",
            observed={"tested": len(tested), "total": len(subclaims)},
        ),
        ScoreCheck(
            id="evidence.loopholes_investigated",
            label="Every identified loophole was investigated to a conclusion",
            passed=(len(resolved) == len(loopholes)) if loopholes else None,
            weight=2.0,
            detail=f"{len(resolved)} of {len(loopholes)} loopholes resolved",
            observed={"resolved": len(resolved), "total": len(loopholes)},
        ),
        ScoreCheck(
            id="evidence.classwise_present",
            label="A per-class breakdown exists for the headline comparison",
            passed=classwise is not None,
            weight=1.0,
            detail="class-wise evaluation completed" if classwise else "no class-wise evaluation has run",
        ),
        ScoreCheck(
            id="evidence.budget_sufficient",
            label="Verification finished without exhausting the compute budget",
            passed=budget_left > 0,
            weight=1.0,
            detail=f"{budget_left:.2f} of {claim.budget.total_units:.2f} units remaining",
        ),
    ]


def _overall(dimensions: list[DimensionScore], evidence: Sequence[Evidence]) -> tuple[int, list[ScoreCheck], str]:
    """Overall confidence blends the dimension scores with the evidence balance."""
    supporting = sum(e.strength for e in evidence if e.stance == EvidenceStance.SUPPORTS)
    contradicting = sum(e.strength for e in evidence if e.stance == EvidenceStance.CONTRADICTS)
    total = supporting + contradicting
    balance = (supporting / total) if total else 0.5

    dimension_mean = sum(d.score for d in dimensions) / len(dimensions) if dimensions else 0
    # Two thirds process quality, one third which way the evidence actually
    # points. Both terms are shown so the number can be checked by hand.
    overall = round(0.6 * dimension_mean + 0.4 * (balance * 100))
    checks = [
        ScoreCheck(
            id="overall.dimension_mean",
            label="Mean of the six process dimensions",
            passed=dimension_mean >= 60,
            weight=0.6,
            detail=f"mean of {[d.score for d in dimensions]} = {dimension_mean:.1f}",
            observed={"dimension_scores": {d.dimension.value: d.score for d in dimensions}},
        ),
        ScoreCheck(
            id="overall.evidence_balance",
            label="Supporting evidence outweighs contradicting evidence",
            passed=balance > 0.5,
            weight=0.4,
            detail=(
                f"supporting weight {supporting:.2f} vs contradicting weight {contradicting:.2f} "
                f"= {balance:.0%} in favour"
            ),
            observed={"supporting": round(supporting, 3), "contradicting": round(contradicting, 3)},
        ),
    ]
    calculation = (
        f"0.6 x {dimension_mean:.1f} (dimension mean) + 0.4 x {balance * 100:.1f} (evidence balance) = {overall}"
    )
    return overall, checks, calculation


def compute_reliability(
    claim: Claim,
    subclaims: Sequence[Subclaim],
    loopholes: Sequence[Loophole],
    jobs: Sequence[Job],
    evidence: Sequence[Evidence],
) -> ReliabilityScore:
    """Build the full reliability score from observed state."""
    groups: list[tuple[ScoreDimension, list[ScoreCheck]]] = [
        (ScoreDimension.REPRODUCIBILITY, _reproducibility(claim, jobs)),
        (ScoreDimension.DATA_INTEGRITY, _data_integrity(claim, jobs)),
        (ScoreDimension.BASELINE_FAIRNESS, _baseline_fairness(claim, jobs)),
        (ScoreDimension.STATISTICAL_STABILITY, _statistical_stability(jobs)),
        (ScoreDimension.TRAINING_HEALTH, _training_health(jobs)),
        (
            ScoreDimension.EVIDENCE_COMPLETENESS,
            _evidence_completeness(claim, subclaims, loopholes, evidence, jobs),
        ),
    ]

    dimensions: list[DimensionScore] = []
    for dimension, checks in groups:
        score, calculation = _score_from(checks)
        dimensions.append(DimensionScore(dimension=dimension, score=score, checks=checks, calculation=calculation))

    overall, overall_checks, overall_calc = _overall(dimensions, evidence)
    dimensions.append(
        DimensionScore(
            dimension=ScoreDimension.OVERALL_CLAIM_CONFIDENCE,
            score=overall,
            checks=overall_checks,
            calculation=overall_calc,
        )
    )
    return ReliabilityScore(
        claim_id=claim.id,
        dimensions=dimensions,
        overall=overall,
        calculation=overall_calc,
    )


def as_table(score: ReliabilityScore) -> list[dict[str, Any]]:
    """Flatten for the report renderer."""
    return [
        {
            "dimension": d.dimension.value,
            "score": d.score,
            "calculation": d.calculation,
            "checks": [
                {"id": c.id, "label": c.label, "passed": c.passed, "weight": c.weight, "detail": c.detail}
                for c in d.checks
            ],
        }
        for d in score.dimensions
    ]
