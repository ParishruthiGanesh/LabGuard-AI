"""The deterministic rule engine.

This is both the demo-mode reasoning backend *and* the safety layer under the
Gemini backend: the Gemini runtime derives every status, score and plan
feasibility decision from these same functions, and uses the model only for
wording, ranking and narrative.  Nothing here calls out to a network.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

from ..actions.registry import (
    REGISTRY,
    get_action,
    requires_approval,
)
from ..experiments.scenario import arms
from ..models.domain import (
    AlternativeExplanation,
    Claim,
    Evidence,
    ExperimentPlan,
    Job,
    Loophole,
    PlanItem,
    ReliabilityScore,
    Subclaim,
    Verdict,
)
from ..models.enums import (
    EvidenceStance,
    JobState,
    LoopholeKind,
    LoopholeStatus,
    SubclaimStatus,
    VerdictStatus,
)
from .base import AuditOutcome

#: Subclaim templates. Each maps onto loopholes and onto registry actions, so
#: the planner can always find a test for anything left untested.
SUBCLAIM_TEMPLATES: list[dict[str, object]] = [
    {
        "key": "equal_budget",
        "statement": "The advantage holds when both models are trained under the same budget.",
        "quantity": "accuracy delta at equal epochs",
        "rationale": "A longer-trained candidate confounds architecture with training budget.",
        "loopholes": [LoopholeKind.UNFAIR_BASELINE],
    },
    {
        "key": "seed_stability",
        "statement": "The advantage is stable across independent random seeds.",
        "quantity": "mean per-seed delta with a 95% interval",
        "rationale": "A single-seed result can be noise rather than a real difference.",
        "loopholes": [LoopholeKind.SEED_SENSITIVITY, LoopholeKind.STATISTICAL_UNCERTAINTY],
    },
    {
        "key": "balanced_metrics",
        "statement": "The advantage survives class-balanced metrics, not only raw accuracy.",
        "quantity": "macro F1 and balanced accuracy deltas",
        "rationale": "At an 8% positive rate accuracy is dominated by the majority class.",
        "loopholes": [LoopholeKind.MISLEADING_METRIC, LoopholeKind.CLASS_IMBALANCE],
    },
    {
        "key": "honest_checkpoint",
        "statement": "The advantage is not produced by selecting checkpoints on the test split.",
        "quantity": "delta with validation-selected checkpoints",
        "rationale": "Choosing the checkpoint that scores best on test leaks the test set.",
        "loopholes": [LoopholeKind.CHERRY_PICKED_CHECKPOINT],
    },
    {
        "key": "no_leakage",
        "statement": "The advantage is not an artefact of train/test leakage.",
        "quantity": "count of duplicated rows across the split",
        "rationale": "Duplicated rows inflate both arms unpredictably.",
        "loopholes": [LoopholeKind.DATA_LEAKAGE, LoopholeKind.TRAIN_TEST_OVERLAP],
    },
    {
        "key": "minority_class",
        "statement": "The advantage appears in the positive class the task is about, not only the majority class.",
        "quantity": "per-class F1 and recall deltas",
        "rationale": "A detector that gains only on negatives has not become a better detector.",
        "loopholes": [LoopholeKind.CLASS_IMBALANCE],
    },
    {
        "key": "threshold",
        "statement": "The advantage is not an artefact of the default 0.5 decision threshold.",
        "quantity": "delta with each arm at its own best threshold",
        "rationale": "A fixed threshold can favour whichever model happens to be better calibrated.",
        "loopholes": [LoopholeKind.IMPROPER_THRESHOLD],
    },
    {
        "key": "reproducible",
        "statement": "The reported checkpoint reproduces the reported number.",
        "quantity": "reproduced metric versus reported metric",
        "rationale": "A result that cannot be re-derived from its own artefacts is not yet evidence.",
        "loopholes": [LoopholeKind.IRREPRODUCIBLE],
    },
]

#: Which registry action settles which subclaim.
ACTION_FOR_SUBCLAIM: dict[str, list[str]] = {
    "equal_budget": ["compare_configurations", "run_seed_comparison"],
    "seed_stability": ["run_seed_comparison"],
    "balanced_metrics": ["recalculate_metrics", "run_seed_comparison"],
    "honest_checkpoint": ["compare_configurations", "inspect_training_curve"],
    "no_leakage": ["check_data_overlap"],
    "minority_class": ["evaluate_classwise"],
    "threshold": ["sweep_decision_threshold"],
    "reproducible": ["resume_from_checkpoint"],
}

#: Planner ordering: cheap, decisive checks first. Expected information gain is
#: a fixed prior per action, scaled down when nothing open depends on it.
#: `follow_up` actions are second-line diagnostics whose value depends on how
#: the first round turns out, so they are held back until a round has been
#: audited and the question is still open. This is what makes the loop
#: recursive rather than one big batch.
ACTION_PRIORITY: list[tuple[str, float, bool]] = [
    ("compare_configurations", 0.62, False),
    ("check_data_overlap", 0.55, False),
    ("resume_from_checkpoint", 0.50, False),
    ("inspect_training_curve", 0.68, False),
    ("run_seed_comparison", 0.95, False),
    ("evaluate_classwise", 0.80, False),
    ("recalculate_metrics", 0.74, False),
    ("sweep_decision_threshold", 0.52, True),
    ("test_domain_shift", 0.30, True),
]


def fingerprint(action_type: str, params: dict) -> str:
    """Stable identity of an action invocation, for duplicate detection."""
    blob = json.dumps({"a": action_type, "p": params}, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------
# Claim Analyst
# --------------------------------------------------------------------------


def build_subclaims(claim: Claim) -> list[Subclaim]:
    baseline, candidate = arms(claim.context)
    out: list[Subclaim] = []
    for template in SUBCLAIM_TEMPLATES:
        statement = str(template["statement"])
        statement = statement.replace("the candidate", candidate.name).replace(
            "both models", f"{baseline.name} and {candidate.name}"
        )
        sub = Subclaim(
            claim_id=claim.id,
            statement=statement,
            measurable_quantity=str(template["quantity"]),
            rationale=str(template["rationale"]),
        )
        sub.id = f"sub_{template['key']}"
        out.append(sub)
    return out


def subclaim_key(subclaim: Subclaim) -> str:
    return subclaim.id.removeprefix("sub_")


# --------------------------------------------------------------------------
# Scientific Skeptic
# --------------------------------------------------------------------------


def detect_loopholes(
    claim: Claim, subclaims: Sequence[Subclaim]
) -> tuple[list[Loophole], list[AlternativeExplanation]]:
    """Read the submitted context and enumerate what could be wrong with it."""
    ctx = claim.context
    baseline, candidate = arms(ctx)
    by_key = {subclaim_key(s): s for s in subclaims}
    found: list[Loophole] = []

    def add(kind: LoopholeKind, title: str, rationale: str, severity: float, keys: list[str]) -> None:
        hole = Loophole(
            claim_id=claim.id,
            kind=kind,
            title=title,
            rationale=rationale,
            severity=severity,
            subclaim_ids=[by_key[k].id for k in keys if k in by_key],
        )
        hole.id = f"hole_{kind.value}"
        found.append(hole)

    seeds = {r.seed for r in ctx.existing_results}
    if len(seeds) <= 1:
        add(
            LoopholeKind.SEED_SENSITIVITY,
            "Reported on a single random seed",
            f"Every submitted result comes from seed {sorted(seeds) or ['unspecified']}. A one-seed gap "
            f"cannot be separated from run-to-run variance.",
            0.85,
            ["seed_stability"],
        )

    positive_rate = ctx.dataset.positive_rate
    if positive_rate < 0.25:
        add(
            LoopholeKind.CLASS_IMBALANCE,
            f"Severe class imbalance ({positive_rate:.1%} positive)",
            "A trivial majority-class predictor already reaches "
            f"{1 - positive_rate:.1%} accuracy, so accuracy carries almost no information here.",
            0.80,
            ["balanced_metrics", "minority_class"],
        )
        if any(r.metric == "accuracy" for r in ctx.existing_results):
            add(
                LoopholeKind.MISLEADING_METRIC,
                "Conclusion rests on raw accuracy",
                "The claim is argued from accuracy on an imbalanced benchmark. Macro F1 and balanced "
                "accuracy can reverse the ordering.",
                0.85,
                ["balanced_metrics"],
            )

    if baseline.epochs and candidate.epochs / baseline.epochs > 1.2:
        add(
            LoopholeKind.UNFAIR_BASELINE,
            "Unequal training budget between the arms",
            f"{candidate.name} trained for {candidate.epochs} epochs against {baseline.name}'s "
            f"{baseline.epochs}. The comparison confounds architecture with budget.",
            0.75,
            ["equal_budget"],
        )

    if any(r.checkpoint_selected_on == "test" for r in ctx.existing_results):
        add(
            LoopholeKind.CHERRY_PICKED_CHECKPOINT,
            "Checkpoints selected on the test split",
            "Selecting the epoch that scores best on test leaks the test set into model selection and "
            "inflates the reported number.",
            0.78,
            ["honest_checkpoint"],
        )

    add(
        LoopholeKind.TRAIN_TEST_OVERLAP,
        "Train/test overlap has not been verified",
        "No leakage check was submitted with the result, so duplication across the split cannot be ruled out.",
        0.45,
        ["no_leakage"],
    )
    add(
        LoopholeKind.STATISTICAL_UNCERTAINTY,
        "No uncertainty reported with the difference",
        "The submission reports point estimates with no confidence interval, so the size of the "
        "difference cannot be compared against its noise.",
        0.70,
        ["seed_stability"],
    )
    add(
        LoopholeKind.IMPROPER_THRESHOLD,
        "Both arms scored at a fixed 0.5 threshold",
        "On imbalanced data the default threshold can favour whichever arm happens to be better "
        "calibrated rather than better ranked.",
        0.45,
        ["threshold"],
    )

    if any(r.checkpoint_uri for r in ctx.existing_results):
        add(
            LoopholeKind.IRREPRODUCIBLE,
            "Reported checkpoint has not been re-verified",
            "A checkpoint URI was supplied but the reported number has not been re-derived from it.",
            0.55,
            ["reproducible"],
        )

    differing = [
        f
        for f in ("hidden_units", "epochs", "class_weight", "learning_rate")
        if getattr(baseline, f) != getattr(candidate, f)
    ]
    if len(differing) > 1:
        add(
            LoopholeKind.CONFOUNDING_VARIABLE,
            "Several factors change at once between the arms",
            "The arms differ in " + ", ".join(differing) + ", so no single factor can be credited "
            "for any difference without an ablation.",
            0.60,
            ["equal_budget"],
        )

    n_test_positives = int(ctx.dataset.n_samples * ctx.dataset.test_fraction * positive_rate)
    if n_test_positives < 150:
        add(
            LoopholeKind.INSUFFICIENT_SAMPLE_SIZE,
            f"Only about {n_test_positives} positive test examples",
            "Minority-class metrics computed on this few positives carry wide intervals.",
            0.50,
            ["balanced_metrics"],
        )

    alternatives = [
        AlternativeExplanation(
            claim_id=claim.id,
            statement=f"{candidate.name} looks better only because it was trained {candidate.epochs // max(1, baseline.epochs)}x longer.",
            tested_by_action="run_seed_comparison",
        ),
        AlternativeExplanation(
            claim_id=claim.id,
            statement=f"{candidate.name} predicts the majority class more often, which raises accuracy while lowering minority recall.",
            tested_by_action="evaluate_classwise",
        ),
        AlternativeExplanation(
            claim_id=claim.id,
            statement="The reported gap is within seed-to-seed variance and would vanish on a different seed.",
            tested_by_action="run_seed_comparison",
        ),
        AlternativeExplanation(
            claim_id=claim.id,
            statement="The gap comes from selecting the reported checkpoint on the test split rather than on validation.",
            tested_by_action="inspect_training_curve",
        ),
    ]
    return found, alternatives


# --------------------------------------------------------------------------
# Experiment Planner
# --------------------------------------------------------------------------


def candidate_params(action: str, claim: Claim) -> list[dict]:
    """Every parameterisation of `action` worth running for this claim.

    Most actions have exactly one; `inspect_training_curve` gets one job per
    submitted configuration, because each run has its own health story.
    """
    from ..experiments.scenario import VERIFICATION_SEEDS

    original_seed = next((r.seed for r in claim.context.existing_results), 11)
    if action == "run_seed_comparison":
        return [{"seeds": VERIFICATION_SEEDS, "equalise_training_budget": True, "equal_epochs": 45}]
    if action == "inspect_training_curve":
        _, candidate = arms(claim.context)
        configs = [candidate] + [m for m in claim.context.models if m.role == "variant"]
        return [{"config_name": c.name, "seed": original_seed, "epochs": c.epochs} for c in configs]
    if action in {
        "recalculate_metrics",
        "evaluate_classwise",
        "check_data_overlap",
        "sweep_decision_threshold",
        "test_domain_shift",
    }:
        return [{"seed": original_seed}]
    return [{}]


def plan_next_round(
    claim: Claim,
    subclaims: Sequence[Subclaim],
    loopholes: Sequence[Loophole],
    jobs: Sequence[Job],
    round_index: int,
) -> ExperimentPlan:
    """Pick the next batch of actions: unresolved questions, cheapest first."""
    already = {j.fingerprint for j in jobs if j.fingerprint}

    open_keys = {subclaim_key(s) for s in subclaims if s.status in (SubclaimStatus.UNTESTED, SubclaimStatus.TESTING)}
    open_holes = [h for h in loopholes if h.status in (LoopholeStatus.OPEN, LoopholeStatus.INVESTIGATING)]

    plan = ExperimentPlan(claim_id=claim.id, round_index=round_index)
    remaining = claim.budget.remaining_units

    for action_name, gain, follow_up in ACTION_PRIORITY:
        if follow_up and round_index == 0:
            continue
        spec = get_action(action_name)
        # Only schedule work that answers something still open.
        answers_subclaim = [
            k for k, actions in ACTION_FOR_SUBCLAIM.items() if action_name in actions and k in open_keys
        ]
        answers_hole = [h for h in open_holes if h.kind.value in spec.addresses]
        if not answers_subclaim and not answers_hole:
            continue

        for raw in candidate_params(action_name, claim):
            params = spec.validate_params(raw)
            if fingerprint(action_name, params) in already:
                continue
            cost = spec.cost(params)
            if cost > remaining:
                continue

            needs_approval, approval_reason = requires_approval(
                spec, params, claim.autonomy_mode, remaining, claim.budget.approval_threshold_units
            )
            targets = [s.id for s in subclaims if subclaim_key(s) in answers_subclaim]
            reason = _plan_reason(action_name, answers_subclaim, answers_hole, params)
            item = PlanItem(
                action_type=action_name,
                params=params,
                reason=reason,
                targets_loophole_ids=[h.id for h in answers_hole],
                targets_subclaim_ids=targets,
                estimated_cost_units=cost,
                expected_information_gain=round(min(0.99, gain * (1.0 if answers_hole else 0.7)), 2),
                requires_approval=needs_approval,
                category=spec.category,
            )
            if needs_approval:
                item.reason += f" Approval required: {approval_reason}."
            plan.items.append(item)
            remaining -= cost

    plan.total_cost_units = round(sum(i.estimated_cost_units for i in plan.items), 2)
    plan.requires_approval = any(i.requires_approval for i in plan.items)
    plan.summary = _plan_summary(plan, round_index)
    return plan


def _plan_reason(action: str, subclaim_keys: list[str], holes: Sequence[Loophole], params: dict) -> str:
    spec = REGISTRY[action]
    parts = [spec.summary]
    if params.get("config_name"):
        parts[0] = f"{spec.summary} Target run: {params['config_name']}."
    if holes:
        parts.append("Targets: " + ", ".join(h.title for h in holes) + ".")
    if subclaim_keys:
        parts.append("Settles subclaim(s): " + ", ".join(subclaim_keys) + ".")
    return " ".join(parts)


def _plan_summary(plan: ExperimentPlan, round_index: int) -> str:
    if not plan.items:
        return "No further experiment would reduce the remaining uncertainty within budget."
    names = ", ".join(i.action_type for i in plan.items)
    return f"Round {round_index + 1}: {len(plan.items)} action(s) for {plan.total_cost_units} compute units - {names}."


# --------------------------------------------------------------------------
# Evidence Auditor
# --------------------------------------------------------------------------


def audit_evidence(
    claim: Claim,
    subclaims: Sequence[Subclaim],
    loopholes: Sequence[Loophole],
    jobs: Sequence[Job],
    evidence: Sequence[Evidence],
) -> AuditOutcome:
    """Map completed jobs onto subclaim conclusions and loophole resolutions."""
    outcome = AuditOutcome()
    completed = {j.action_type: j for j in jobs if j.state == JobState.COMPLETED}
    failed = {j.action_type: j for j in jobs if j.state in (JobState.FAILED, JobState.BLOCKED_LOOP)}
    by_job: dict[str, list[Evidence]] = {}
    for ev in evidence:
        by_job.setdefault(ev.job_id, []).append(ev)

    for subclaim in subclaims:
        key = subclaim_key(subclaim)
        actions = ACTION_FOR_SUBCLAIM.get(key, [])
        relevant = [completed[a] for a in actions if a in completed]
        blocked = [failed[a] for a in actions if a in failed]
        if not relevant:
            if blocked:
                outcome.subclaim_status[subclaim.id] = SubclaimStatus.INCONCLUSIVE
                outcome.subclaim_confidence[subclaim.id] = 0.15
                outcome.open_questions.append(
                    f"{subclaim.statement} could not be settled: "
                    f"{blocked[0].action_type} ended as {blocked[0].state.value}."
                )
            continue

        items = [e for job in relevant for e in by_job.get(job.id, [])]
        supporting = sum(e.strength for e in items if e.stance == EvidenceStance.SUPPORTS)
        contradicting = sum(e.strength for e in items if e.stance == EvidenceStance.CONTRADICTS)
        outcome.subclaim_evidence[subclaim.id] = [e.id for e in items]

        if contradicting > supporting * 1.15:
            status, confidence = SubclaimStatus.CONTRADICTED, contradicting / max(1e-9, supporting + contradicting)
        elif supporting > contradicting * 1.15:
            status, confidence = SubclaimStatus.SUPPORTED, supporting / max(1e-9, supporting + contradicting)
        else:
            status, confidence = SubclaimStatus.INCONCLUSIVE, 0.5
        outcome.subclaim_status[subclaim.id] = status
        outcome.subclaim_confidence[subclaim.id] = round(float(confidence), 3)

    for hole in loopholes:
        actions = [name for name, spec in REGISTRY.items() if hole.kind.value in spec.addresses]
        ran = [completed[a] for a in actions if a in completed]
        if not ran:
            blocked = [failed[a] for a in actions if a in failed]
            if blocked:
                outcome.loophole_status[hole.id] = LoopholeStatus.UNRESOLVED
                outcome.loophole_resolution[hole.id] = (
                    f"Could not be settled: {blocked[0].action_type} ended as {blocked[0].state.value}."
                )
            continue
        linked = [s for s in subclaims if s.id in hole.subclaim_ids]
        statuses = {outcome.subclaim_status.get(s.id, s.status) for s in linked}
        if SubclaimStatus.CONTRADICTED in statuses:
            outcome.loophole_status[hole.id] = LoopholeStatus.CONFIRMED
            outcome.loophole_resolution[hole.id] = "Investigated and confirmed: it materially affects the claim."
        elif statuses and statuses <= {SubclaimStatus.SUPPORTED}:
            outcome.loophole_status[hole.id] = LoopholeStatus.REFUTED
            outcome.loophole_resolution[hole.id] = "Investigated and ruled out by direct measurement."
        else:
            outcome.loophole_status[hole.id] = LoopholeStatus.UNRESOLVED
            outcome.loophole_resolution[hole.id] = "Investigated; the measurements were not decisive."

    untested = [
        s
        for s in subclaims
        if outcome.subclaim_status.get(s.id, s.status) in (SubclaimStatus.UNTESTED, SubclaimStatus.TESTING)
    ]
    severe_open = [
        h
        for h in loopholes
        if outcome.loophole_status.get(h.id, h.status) in (LoopholeStatus.OPEN, LoopholeStatus.INVESTIGATING)
        and h.severity >= 0.6
    ]
    outcome.needs_more_testing = bool(untested or severe_open)
    outcome.rationale = _audit_rationale(untested, severe_open, outcome)
    return outcome


def _audit_rationale(untested: Sequence[Subclaim], severe_open: Sequence[Loophole], outcome: AuditOutcome) -> str:
    contradicted = sum(1 for s in outcome.subclaim_status.values() if s == SubclaimStatus.CONTRADICTED)
    supported = sum(1 for s in outcome.subclaim_status.values() if s == SubclaimStatus.SUPPORTED)
    parts = [f"{supported} subclaim(s) supported, {contradicted} contradicted."]
    if untested:
        parts.append(
            f"{len(untested)} subclaim(s) still untested: " + ", ".join(subclaim_key(s) for s in untested) + "."
        )
    if severe_open:
        parts.append("High-severity loopholes still open: " + ", ".join(h.kind.value for h in severe_open) + ".")
    if not untested and not severe_open:
        parts.append("Every subclaim has been tested and no high-severity loophole remains open.")
    return " ".join(parts)


# --------------------------------------------------------------------------
# Verdict Agent
# --------------------------------------------------------------------------


def rule_based_status(
    claim: Claim,
    subclaims: Sequence[Subclaim],
    jobs: Sequence[Job],
    evidence: Sequence[Evidence],
) -> tuple[VerdictStatus, list[str]]:
    """Decide the verdict from measurements alone, with its reasons."""
    reasons: list[str] = []
    supported = [s for s in subclaims if s.status == SubclaimStatus.SUPPORTED]
    contradicted = [s for s in subclaims if s.status == SubclaimStatus.CONTRADICTED]
    inconclusive = [s for s in subclaims if s.status in (SubclaimStatus.INCONCLUSIVE, SubclaimStatus.UNTESTED)]

    seed_job = next((j for j in jobs if j.action_type == "run_seed_comparison" and j.state == JobState.COMPLETED), None)
    summaries = (seed_job.result.get("paired_summary") if seed_job else {}) or {}
    claimed_metric = next((r.metric for r in claim.context.existing_results), "accuracy")
    claimed = summaries.get(claimed_metric)

    reasons.append(
        f"{len(supported)} subclaim(s) supported, {len(contradicted)} contradicted, {len(inconclusive)} inconclusive."
    )

    claimed_reversed = bool(claimed and claimed["mean_delta"] < 0 and not claimed["ci_includes_zero"])
    claimed_unstable = bool(claimed and claimed["ci_includes_zero"])
    if claimed:
        reasons.append(
            f"On the claimed metric ({claimed_metric}) the mean per-seed difference is "
            f"{claimed['mean_delta']:+.4f} with a 95% interval of "
            f"[{claimed['ci_low']:+.4f}, {claimed['ci_high']:+.4f}] over {claimed['n_seeds']} seeds."
        )

    if claimed_reversed:
        reasons.append("The claimed metric reverses direction with an interval that excludes zero.")
        return VerdictStatus.REFUTED, reasons
    if not seed_job:
        reasons.append("No multi-seed comparison completed, so stability was never established.")
        return VerdictStatus.INCONCLUSIVE, reasons
    if len(contradicted) >= 2:
        reasons.append("Two or more subclaims are contradicted by direct measurement.")
        return VerdictStatus.NOT_SUFFICIENTLY_SUPPORTED, reasons
    if contradicted and claimed_unstable:
        reasons.append("The claimed metric is not separated from zero and a subclaim is contradicted.")
        return VerdictStatus.FRAGILE, reasons
    if not contradicted and not inconclusive and not claimed_unstable:
        reasons.append("Every subclaim is supported and the claimed metric is separated from zero.")
        return VerdictStatus.SUPPORTED, reasons
    if not contradicted and claimed and not claimed_unstable:
        reasons.append("No subclaim is contradicted, but some remain inconclusive.")
        return VerdictStatus.PROVISIONALLY_SUPPORTED, reasons
    if claimed_unstable:
        reasons.append("The claimed advantage is not separated from run-to-run variance.")
        return VerdictStatus.FRAGILE, reasons
    return VerdictStatus.INCONCLUSIVE, reasons


def build_verdict(
    claim: Claim,
    subclaims: Sequence[Subclaim],
    loopholes: Sequence[Loophole],
    jobs: Sequence[Job],
    evidence: Sequence[Evidence],
    score: ReliabilityScore,
) -> Verdict:
    status, reasons = rule_based_status(claim, subclaims, jobs, evidence)
    contradicting = [e for e in evidence if e.stance == EvidenceStance.CONTRADICTS]
    supporting = [e for e in evidence if e.stance == EvidenceStance.SUPPORTS]

    incidents: list[str] = []
    for job in jobs:
        for event in job.health.events:
            incidents.append(
                f"{job.action_type}: {event.anomaly.value} - {event.detail}"
                + (f" Action taken: {event.action_taken}." if event.action_taken else "")
            )

    uncertainty: list[str] = []
    for subclaim in subclaims:
        if subclaim.status in (SubclaimStatus.INCONCLUSIVE, SubclaimStatus.UNTESTED):
            uncertainty.append(subclaim.statement)
    for hole in loopholes:
        if hole.status in (LoopholeStatus.OPEN, LoopholeStatus.UNRESOLVED):
            uncertainty.append(f"{hole.title} was not settled ({hole.kind.value}).")
    if claim.budget.remaining_units <= 0:
        uncertainty.append("The compute budget was exhausted before every question was closed.")

    seed_job = next((j for j in jobs if j.action_type == "run_seed_comparison" and j.state == JobState.COMPLETED), None)
    reproducibility = {
        "seeds_tested": (seed_job.result.get("seeds") if seed_job else []),
        "equalised_training_budget": bool(seed_job and seed_job.result.get("equalised_training_budget")),
        "checkpoint_selection": (seed_job.result.get("checkpoint_selection") if seed_job else None),
        "dataset": claim.context.dataset.model_dump(mode="json"),
        "configurations": [m.model_dump(mode="json") for m in claim.context.models],
    }

    headline = _headline(status, claim, seed_job)
    narrative = " ".join(reasons)
    return Verdict(
        claim_id=claim.id,
        status=status,
        rule_based_status=status,
        headline=headline,
        narrative=narrative,
        evidence_summary=[e.statement for e in (contradicting + supporting)][:10],
        remaining_uncertainty=uncertainty,
        run_health_incidents=incidents,
        reproducibility=reproducibility,
        score=score,
        generated_by="deterministic",
    )


def _headline(status: VerdictStatus, claim: Claim, seed_job: Job | None) -> str:
    label = status.value.replace("_", " ")
    if seed_job:
        n = len(seed_job.result.get("seeds") or [])
        return f"Claim status: {label} after {n}-seed verification under an equalised training budget."
    return f"Claim status: {label}."
