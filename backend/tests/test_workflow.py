"""End-to-end: claim submission through to a verdict, on the real workflow."""

from __future__ import annotations

import pytest
import pytest_asyncio

from labguard.experiments.scenario import demo_claim
from labguard.models.enums import (
    AutonomyMode,
    ClaimState,
    JobState,
    LoopholeKind,
    SubclaimStatus,
    VerdictStatus,
)
from labguard.scoring.reliability import compute_reliability

#: The full workflow takes a few seconds of real training, so the tests that
#: only inspect the finished state share a single run.
pytestmark = pytest.mark.asyncio(loop_scope="module")


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def completed(tmp_path_factory):
    from labguard.config import Settings
    from labguard.services import Services

    svc = Services(
        Settings(
            LABGUARD_MODE="demo",
            SIMULATED_QUEUE_LATENCY=0.0,
            ARTIFACT_DIR=str(tmp_path_factory.mktemp("artifacts")),
        )
    )
    await svc.start()
    claim = await _run_to_verdict(svc)
    try:
        yield svc, claim
    finally:
        await svc.close()


async def _run_to_verdict(services, autonomy=AutonomyMode.MANAGED_AUTONOMY, approve=True):
    claim = await services.orchestrator.create_claim(demo_claim(autonomy))
    for _ in range(40):
        await services.orchestrator.advance(claim.id)
        current = await services.store.get_claim(claim.id)
        if current.state == ClaimState.AWAITING_APPROVAL:
            plan = await services.orchestrator.pending_plan(claim.id)
            if plan is None:
                break
            await services.orchestrator.decide_plan(claim.id, plan.id, approve, "test")
            if not approve:
                break
        if current.state == ClaimState.EXECUTING:
            await services.bus.drain(timeout=180)
            continue
        if current.state.is_terminal:
            break
    await services.bus.drain(timeout=180)
    return await services.store.get_claim(claim.id)


class TestClaimAnalysis:
    async def test_claim_is_decomposed_and_audited_for_loopholes(self, services):
        claim = await services.orchestrator.create_claim(demo_claim())
        await services.orchestrator.advance(claim.id)

        subclaims = await services.store.list_subclaims(claim.id)
        loopholes = await services.store.list_loopholes(claim.id)
        kinds = {h.kind for h in loopholes}

        assert len(subclaims) >= 6
        # The weaknesses that are actually present in the submission.
        assert LoopholeKind.SEED_SENSITIVITY in kinds
        assert LoopholeKind.CLASS_IMBALANCE in kinds
        assert LoopholeKind.UNFAIR_BASELINE in kinds
        assert LoopholeKind.CHERRY_PICKED_CHECKPOINT in kinds
        assert LoopholeKind.MISLEADING_METRIC in kinds


class TestApprovalGate:
    async def test_expensive_experiment_blocks_on_approval(self, services):
        claim = await services.orchestrator.create_claim(demo_claim())
        await services.orchestrator.advance(claim.id)
        current = await services.store.get_claim(claim.id)

        assert current.state == ClaimState.AWAITING_APPROVAL
        plan = await services.orchestrator.pending_plan(claim.id)
        assert plan is not None and plan.requires_approval
        expensive = [i for i in plan.items if i.requires_approval]
        assert [i.action_type for i in expensive] == ["run_seed_comparison"]

        jobs = await services.store.list_jobs(claim.id)
        assert jobs and all(j.state == JobState.AWAITING_APPROVAL for j in jobs), "nothing may run before approval"

    async def test_rejecting_a_plan_halts_the_claim(self, services):
        claim = await _run_to_verdict(services, approve=False)
        assert claim.state == ClaimState.HALTED_APPROVAL
        jobs = await services.store.list_jobs(claim.id)
        assert all(j.state == JobState.REJECTED for j in jobs)

    async def test_observe_only_recommends_but_never_executes(self, services):
        claim = await services.orchestrator.create_claim(demo_claim(AutonomyMode.OBSERVE_ONLY))
        for _ in range(6):
            await services.orchestrator.advance(claim.id)
            current = await services.store.get_claim(claim.id)
            if current.state == ClaimState.AWAITING_APPROVAL:
                plan = await services.orchestrator.pending_plan(claim.id)
                await services.orchestrator.decide_plan(claim.id, plan.id, True, "test")
            if current.state.is_terminal:
                break
        await services.bus.drain(timeout=30)

        jobs = await services.store.list_jobs(claim.id)
        assert all(j.attempts == 0 for j in jobs), "observe-only must not execute anything"


class TestFullWorkflow:
    async def test_reaches_a_verdict_with_traceable_evidence(self, completed):
        services, claim = completed
        assert claim.state == ClaimState.VERDICT

        verdict = await services.store.get_verdict(claim.id)
        assert verdict is not None
        # The submitted claim is genuinely not supported by the corrected
        # comparison, so anything stronger would be wrong.
        assert verdict.status in (
            VerdictStatus.NOT_SUFFICIENTLY_SUPPORTED,
            VerdictStatus.FRAGILE,
            VerdictStatus.REFUTED,
        )
        assert verdict.evidence_summary
        assert verdict.remaining_uncertainty

    async def test_recursion_happens_and_is_bounded(self, completed):
        services, claim = completed
        plans = await services.store.list_plans(claim.id)
        assert len(plans) >= 2, "the auditor should have triggered at least one further round"
        assert claim.planning_round + 1 <= services.settings.max_planning_rounds

    async def test_seed_comparison_contradicts_the_claim_on_balanced_metrics(self, completed):
        services, claim = completed
        jobs = await services.store.list_jobs(claim.id)
        seed_job = next(j for j in jobs if j.action_type == "run_seed_comparison")

        assert seed_job.state == JobState.COMPLETED
        summary = seed_job.result["paired_summary"]
        assert summary["accuracy"]["ci_includes_zero"], "the headline metric should not be separated from zero"
        assert summary["macro_f1"]["mean_delta"] < 0
        assert summary["balanced_accuracy"]["wins_for_b"] == 0

    async def test_no_train_test_leakage_is_found_and_the_detector_is_proven(self, completed):
        services, claim = completed
        jobs = await services.store.list_jobs(claim.id)
        overlap = next(j for j in jobs if j.action_type == "check_data_overlap")
        assert overlap.result["overlapping_rows"] == 0
        assert overlap.result["positive_control"]["detector_working"] is True


class TestRunHealth:
    async def test_overfitting_is_detected_in_the_reported_run(self, completed):
        services, claim = completed
        jobs = await services.store.list_jobs(claim.id)
        replay = next(
            j for j in jobs if j.action_type == "inspect_training_curve" and j.params.get("config_name") == "Model B"
        )
        assert any(e.anomaly.value == "overfitting" for e in replay.health.events)

    async def test_a_diverging_run_is_repaired_and_completes(self, completed):
        services, claim = completed
        jobs = await services.store.list_jobs(claim.id)
        variant = next(
            j
            for j in jobs
            if j.action_type == "inspect_training_curve" and "variant" in str(j.params.get("config_name"))
        )
        assert any(e.anomaly.value == "nan_loss" for e in variant.health.events)
        assert any(r.startswith("recovery:adjust_learning_rate") for r in variant.recovery_actions)
        assert variant.state == JobState.COMPLETED and variant.attempts == 2

    async def test_an_unchanging_failure_stops_instead_of_retrying_forever(self, completed):
        services, claim = completed
        jobs = await services.store.list_jobs(claim.id)
        stuck = next(j for j in jobs if j.action_type == "resume_from_checkpoint")

        assert stuck.state == JobState.BLOCKED_LOOP
        assert stuck.attempts == 3, "three identical failures, then escalation"
        assert any(e.anomaly.value == "recovery_loop" for e in stuck.health.events)

    async def test_recovery_never_leaves_the_declared_bounds(self, completed):
        services, claim = completed
        jobs = await services.store.list_jobs(claim.id)
        variant = next(
            j
            for j in jobs
            if j.action_type == "inspect_training_curve" and "variant" in str(j.params.get("config_name"))
        )
        applied = variant.params["_recovery_overrides"]["learning_rate"]
        assert 0.001 <= applied <= 0.5


class TestScoring:
    async def test_every_score_is_backed_by_named_checks(self, completed):
        services, claim = completed
        subclaims = await services.store.list_subclaims(claim.id)
        loopholes = await services.store.list_loopholes(claim.id)
        jobs = await services.store.list_jobs(claim.id)
        evidence = await services.store.list_evidence(claim.id)
        score = compute_reliability(claim, subclaims, loopholes, jobs, evidence)

        assert len(score.dimensions) == 7
        for dimension in score.dimensions:
            assert 0 <= dimension.score <= 100
            assert dimension.checks, f"{dimension.dimension} has no checks behind it"
            assert dimension.calculation, f"{dimension.dimension} does not show its arithmetic"
            for check in dimension.checks:
                assert check.detail, f"{check.id} does not explain itself"

    async def test_weak_evidence_produces_a_low_confidence_score(self, completed):
        services, claim = completed
        verdict = await services.store.get_verdict(claim.id)
        assert verdict.score.overall < 60

    async def test_data_integrity_scores_full_marks_on_a_clean_split(self, completed):
        services, claim = completed
        verdict = await services.store.get_verdict(claim.id)
        integrity = next(d for d in verdict.score.dimensions if d.dimension.value == "data_integrity")
        assert integrity.score == 100


class TestLedgerAndReport:
    async def test_the_ledger_is_ordered_and_complete(self, completed):
        services, claim = completed
        ledger = await services.store.list_ledger(claim.id)

        assert [e.sequence for e in ledger] == list(range(1, len(ledger) + 1))
        actions = {e.action for e in ledger}
        assert "claim_submitted" in actions
        assert "detect_loopholes" in actions
        assert "final_verdict" in actions
        assert any(a.startswith("loop_detected") for a in actions)
        assert all(e.decision or e.reason for e in ledger), "every ledger row must explain itself"

    async def test_the_report_is_written_and_shows_its_arithmetic(self, completed):
        services, claim = completed
        report = await services.artifacts.read_text(f"{claim.id}/reliability_report.md")

        assert report is not None
        assert "# LabGuard AI reliability report" in report
        assert "## Reliability score" in report
        assert "### Checks behind each score" in report
        assert "## Audit trail" in report
        assert "weighted checks passed" in report


class TestSubclaims:
    async def test_conclusions_are_recorded_against_each_subclaim(self, completed):
        services, claim = completed
        subclaims = await services.store.list_subclaims(claim.id)
        statuses = {s.id: s.status for s in subclaims}

        assert statuses["sub_seed_stability"] == SubclaimStatus.CONTRADICTED
        assert statuses["sub_balanced_metrics"] == SubclaimStatus.CONTRADICTED
        assert statuses["sub_minority_class"] == SubclaimStatus.CONTRADICTED
        assert statuses["sub_no_leakage"] == SubclaimStatus.SUPPORTED
        # The checkpoint could never be read, so this one cannot be settled.
        assert statuses["sub_reproducible"] == SubclaimStatus.INCONCLUSIVE
