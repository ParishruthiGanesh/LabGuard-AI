"""The claim state machine.

The orchestrator is the only component that moves a claim between states.  It
calls the agent runtime for reasoning, writes everything it learns to the
shared store, and publishes approved work to the job bus.  It never executes an
experiment itself: that is the worker's job, reached asynchronously.

    CREATED -> ANALYZING -> SKEPTIC_REVIEW -> PLANNING -> AWAITING_APPROVAL
            -> EXECUTING -> AUDITING -> (PLANNING | VERDICT)

with terminal HALTED_BUDGET, HALTED_LOOP and HALTED_APPROVAL.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any

from .actions.registry import get_action
from .agents.base import AgentRuntime
from .agents.deterministic import fingerprint
from .config import Settings
from .infra.artifacts import ArtifactStore
from .infra.bus import JobBus
from .infra.store import StateStore
from .models.domain import (
    Claim,
    ExperimentPlan,
    Job,
    LedgerEntry,
    PlanItem,
    Verdict,
)
from .models.enums import (
    AgentName,
    AutonomyMode,
    ClaimState,
    JobState,
)
from .reporting import render_report
from .scoring.reliability import compute_reliability

log = logging.getLogger("labguard.orchestrator")


class Orchestrator:
    def __init__(
        self,
        store: StateStore,
        bus: JobBus,
        artifacts: ArtifactStore,
        runtime: AgentRuntime,
        settings: Settings,
    ) -> None:
        self.store = store
        self.bus = bus
        self.artifacts = artifacts
        self.runtime = runtime
        self.settings = settings
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    # -- ledger ----------------------------------------------------------

    async def log(
        self,
        claim: Claim,
        agent: AgentName,
        action: str,
        *,
        reason: str = "",
        inputs: dict[str, Any] | None = None,
        results: dict[str, Any] | None = None,
        decision: str = "",
        job_id: str | None = None,
        artifacts: list[str] | None = None,
    ) -> LedgerEntry:
        entry = LedgerEntry(
            claim_id=claim.id,
            agent=agent,
            action=action,
            reason=reason,
            input_summary=inputs or {},
            result_summary=results or {},
            decision=decision,
            job_id=job_id,
            artifact_uris=artifacts or [],
        )
        await self.store.append_ledger(entry)
        await self.bus.publish_event({"type": "ledger", "claim_id": claim.id, "agent": agent.value, "action": action})
        return entry

    async def _set_state(self, claim: Claim, state: ClaimState, agent: AgentName | None, latest: str) -> None:
        claim.state = state
        claim.active_agent = agent
        claim.latest_action = latest
        await self.store.save_claim(claim)

    # -- entry points ----------------------------------------------------

    async def create_claim(self, claim: Claim) -> Claim:
        claim.reasoning_backend = self.runtime.name
        await self.store.save_claim(claim)
        await self.log(
            claim,
            AgentName.ORCHESTRATOR,
            "claim_submitted",
            reason="A researcher submitted a claim for verification.",
            inputs={
                "claim": claim.text,
                "autonomy_mode": claim.autonomy_mode.value,
                "budget_units": claim.budget.total_units,
                "models": [m.name for m in claim.context.models],
            },
            decision="Begin claim analysis.",
        )
        return claim

    async def advance(self, claim_id: str) -> Claim | None:
        """Drive the state machine as far as it can go without waiting."""
        async with self._locks[claim_id]:
            return await self._advance_locked(claim_id)

    async def _advance_locked(self, claim_id: str) -> Claim | None:
        for _ in range(12):  # bounded: every branch either advances or returns
            claim = await self.store.get_claim(claim_id)
            if claim is None or claim.state.is_terminal:
                return claim

            if claim.state == ClaimState.CREATED:
                await self._run_claim_analyst(claim)
            elif claim.state == ClaimState.ANALYZING:
                await self._run_skeptic(claim)
            elif claim.state == ClaimState.SKEPTIC_REVIEW:
                await self._run_planner(claim)
            elif claim.state == ClaimState.PLANNING:
                await self._dispatch_current_plan(claim)
            elif claim.state in (ClaimState.AWAITING_APPROVAL, ClaimState.EXECUTING):
                return claim  # waiting on a human or on the worker
            elif claim.state == ClaimState.AUDITING:
                await self._run_auditor(claim)
            else:
                return claim
        return await self.store.get_claim(claim_id)

    # -- state handlers --------------------------------------------------

    async def _run_claim_analyst(self, claim: Claim) -> None:
        await self._set_state(claim, ClaimState.ANALYZING, AgentName.CLAIM_ANALYST, "Decomposing the claim")
        subclaims = await self.runtime.analyse_claim(claim)
        await self.store.save_subclaims(subclaims)
        await self.log(
            claim,
            AgentName.CLAIM_ANALYST,
            "decompose_claim",
            reason="A broad claim cannot be tested; measurable subclaims can.",
            inputs={"claim": claim.text},
            results={"subclaims": [s.statement for s in subclaims]},
            decision=f"Produced {len(subclaims)} testable subclaims.",
        )
        claim.state = ClaimState.ANALYZING
        await self.store.save_claim(claim)

    async def _run_skeptic(self, claim: Claim) -> None:
        await self._set_state(claim, ClaimState.SKEPTIC_REVIEW, AgentName.SCIENTIFIC_SKEPTIC, "Searching for loopholes")
        subclaims = await self.store.list_subclaims(claim.id)
        loopholes, alternatives = await self.runtime.find_loopholes(claim, subclaims)
        await self.store.save_loopholes(loopholes)
        await self.store.save_alternatives(alternatives)
        await self.log(
            claim,
            AgentName.SCIENTIFIC_SKEPTIC,
            "detect_loopholes",
            reason="Even a correctly executed experiment can support a wrong conclusion.",
            inputs={
                "positive_rate": claim.context.dataset.positive_rate,
                "reported_seeds": sorted({r.seed for r in claim.context.existing_results}),
                "checkpoint_selection": sorted({r.checkpoint_selected_on for r in claim.context.existing_results}),
            },
            results={
                "loopholes": [{"kind": h.kind.value, "severity": h.severity, "title": h.title} for h in loopholes],
                "alternative_explanations": [a.statement for a in alternatives],
            },
            decision=f"Identified {len(loopholes)} loopholes and {len(alternatives)} alternative explanations.",
        )
        claim.state = ClaimState.SKEPTIC_REVIEW
        await self.store.save_claim(claim)

    async def _run_planner(self, claim: Claim) -> None:
        await self._set_state(
            claim, ClaimState.PLANNING, AgentName.EXPERIMENT_PLANNER, "Planning verification experiments"
        )
        subclaims = await self.store.list_subclaims(claim.id)
        loopholes = await self.store.list_loopholes(claim.id)
        jobs = await self.store.list_jobs(claim.id)
        evidence = await self.store.list_evidence(claim.id)

        plan = await self.runtime.plan_round(claim, subclaims, loopholes, jobs, evidence, claim.planning_round)
        if not plan.items:
            await self.log(
                claim,
                AgentName.EXPERIMENT_PLANNER,
                "plan_round",
                reason="Looked for a further experiment that could change a conclusion.",
                results={"items": []},
                decision="No remaining experiment would reduce uncertainty within budget; going to verdict.",
            )
            await self._finalise(claim)
            return

        plan.status = "awaiting_approval" if plan.requires_approval else "approved"
        await self.store.save_plan(plan)
        await self.log(
            claim,
            AgentName.EXPERIMENT_PLANNER,
            "plan_round",
            reason=f"Round {plan.round_index + 1} planning against the open loopholes and untested subclaims.",
            inputs={"remaining_budget_units": claim.budget.remaining_units},
            results={
                "summary": plan.summary,
                "items": [
                    {
                        "action": i.action_type,
                        "cost": i.estimated_cost_units,
                        "expected_information_gain": i.expected_information_gain,
                        "requires_approval": i.requires_approval,
                    }
                    for i in plan.items
                ],
            },
            decision=(
                "Plan needs human approval before any of it runs."
                if plan.requires_approval
                else "Plan is inside the autonomy policy; queueing it now."
            ),
        )
        claim.state = ClaimState.PLANNING
        await self.store.save_claim(claim)

    async def _current_plan(self, claim: Claim) -> ExperimentPlan | None:
        plans = await self.store.list_plans(claim.id)
        return plans[-1] if plans else None

    async def _dispatch_current_plan(self, claim: Claim) -> None:
        plan = await self._current_plan(claim)
        if plan is None:
            await self._finalise(claim)
            return

        if plan.requires_approval and plan.status == "awaiting_approval":
            await self._set_state(
                claim,
                ClaimState.AWAITING_APPROVAL,
                AgentName.RUN_MANAGER,
                f"Waiting for approval of round {plan.round_index + 1}",
            )
            await self._create_jobs(claim, plan, queue_now=False)
            return

        if claim.autonomy_mode == AutonomyMode.OBSERVE_ONLY:
            await self._create_jobs(claim, plan, queue_now=False)
            await self._set_state(
                claim,
                ClaimState.HALTED_APPROVAL,
                AgentName.RUN_MANAGER,
                "Observe-only mode: actions recommended, nothing executed",
            )
            claim.halt_reason = (
                "Autonomy is set to observe-only, so LabGuard recommended the plan but executed nothing. "
                "Raise the autonomy level or approve the plan to continue."
            )
            await self.store.save_claim(claim)
            return

        await self._create_jobs(claim, plan, queue_now=True)
        await self._set_state(
            claim, ClaimState.EXECUTING, AgentName.RUN_MANAGER, f"Executing round {plan.round_index + 1}"
        )

    async def _create_jobs(self, claim: Claim, plan: ExperimentPlan, queue_now: bool) -> list[Job]:
        """Materialise plan items as jobs, and publish the approved ones."""
        created: list[Job] = []
        for item in plan.items:
            if item.job_id:
                continue
            spec = get_action(item.action_type)
            job = Job(
                claim_id=claim.id,
                plan_id=plan.id,
                plan_item_id=item.id,
                action_type=item.action_type,
                params=dict(item.params),
                category=item.category,
                reason=item.reason,
                estimated_cost_units=item.estimated_cost_units,
                max_retries=spec.max_retries,
                fingerprint=fingerprint(item.action_type, item.params),
                state=JobState.AWAITING_APPROVAL if not queue_now else JobState.PLANNED,
            )
            item.job_id = job.id
            await self.store.save_job(job)
            created.append(job)
        await self.store.save_plan(plan)

        if queue_now:
            for job in created:
                await self.queue_job(claim, job)
        return created

    async def queue_job(self, claim: Claim, job: Job) -> None:
        from .models.domain import utcnow

        job.state = JobState.QUEUED
        job.queued_at = utcnow()
        await self.store.save_job(job)
        await self.log(
            claim,
            AgentName.RUN_MANAGER,
            f"queue:{job.action_type}",
            reason=job.reason,
            inputs={"params": job.params, "attempt": job.attempts + 1},
            decision=f"Published to the job bus (estimated {job.estimated_cost_units} units).",
            job_id=job.id,
        )
        await self.bus.publish_job({"claim_id": claim.id, "job_id": job.id, "action_type": job.action_type})

    # -- approvals -------------------------------------------------------

    async def decide_plan(self, claim_id: str, plan_id: str, approved: bool, decided_by: str) -> Claim | None:
        from .models.domain import utcnow

        async with self._locks[claim_id]:
            claim = await self.store.get_claim(claim_id)
            plan = await self.store.get_plan(claim_id, plan_id) if claim else None
            if claim is None or plan is None:
                return None

            plan.status = "approved" if approved else "rejected"
            plan.approved_by = decided_by
            plan.decided_at = utcnow()
            await self.store.save_plan(plan)

            jobs = {j.plan_item_id: j for j in await self.store.list_jobs(claim_id)}
            await self.log(
                claim,
                AgentName.RUN_MANAGER,
                "plan_decision",
                reason=f"Round {plan.round_index + 1} required human approval.",
                inputs={"plan_id": plan.id, "cost_units": plan.total_cost_units},
                decision=f"{'Approved' if approved else 'Rejected'} by {decided_by}.",
            )

            if not approved:
                for item in plan.items:
                    job = jobs.get(item.id)
                    if job and not job.state.is_terminal:
                        job.state = JobState.REJECTED
                        await self.store.save_job(job)
                claim.state = ClaimState.HALTED_APPROVAL
                claim.halt_reason = f"Round {plan.round_index + 1} was rejected by {decided_by}."
                claim.active_agent = None
                claim.latest_action = "Plan rejected; execution stopped."
                await self.store.save_claim(claim)
                return claim

            if claim.autonomy_mode == AutonomyMode.OBSERVE_ONLY:
                # Approval does not override the mode: observe-only means the
                # platform recommends and never executes. The researcher has to
                # raise the autonomy level deliberately.
                claim.state = ClaimState.HALTED_APPROVAL
                claim.halt_reason = (
                    "The plan was approved, but autonomy is set to observe-only so nothing was executed. "
                    "Raise the autonomy level to safe-repair or managed autonomy to run it."
                )
                claim.active_agent = None
                claim.latest_action = "Plan approved but withheld by the observe-only policy."
                await self.store.save_claim(claim)
                await self.log(
                    claim,
                    AgentName.RUN_MANAGER,
                    "policy_block",
                    reason="Autonomy mode is observe-only.",
                    decision=claim.halt_reason,
                )
                return claim

            for item in plan.items:
                job = jobs.get(item.id)
                if job and job.state in (JobState.AWAITING_APPROVAL, JobState.PLANNED):
                    await self.queue_job(claim, job)
            await self._set_state(
                claim, ClaimState.EXECUTING, AgentName.RUN_MANAGER, f"Executing round {plan.round_index + 1}"
            )
            return claim

    # -- completion ------------------------------------------------------

    async def on_job_finished(self, claim_id: str) -> None:
        """Called by the worker; moves to auditing once the round is done."""
        async with self._locks[claim_id]:
            claim = await self.store.get_claim(claim_id)
            if claim is None or claim.state != ClaimState.EXECUTING:
                return
            plan = await self._current_plan(claim)
            if plan is None:
                return
            job_ids = {i.job_id for i in plan.items if i.job_id}
            jobs = [j for j in await self.store.list_jobs(claim.id) if j.id in job_ids]
            if any(not j.state.is_terminal for j in jobs):
                return
            plan.status = "executed"
            await self.store.save_plan(plan)
            claim.state = ClaimState.AUDITING
            await self.store.save_claim(claim)
        await self.advance(claim_id)

    async def _run_auditor(self, claim: Claim) -> None:
        await self._set_state(claim, ClaimState.AUDITING, AgentName.EVIDENCE_AUDITOR, "Auditing the evidence")
        subclaims = await self.store.list_subclaims(claim.id)
        loopholes = await self.store.list_loopholes(claim.id)
        jobs = await self.store.list_jobs(claim.id)
        evidence = await self.store.list_evidence(claim.id)

        outcome = await self.runtime.audit(claim, subclaims, loopholes, jobs, evidence)

        for subclaim in subclaims:
            if subclaim.id in outcome.subclaim_status:
                subclaim.status = outcome.subclaim_status[subclaim.id]
                subclaim.confidence = outcome.subclaim_confidence.get(subclaim.id, subclaim.confidence)
                subclaim.evidence_ids = outcome.subclaim_evidence.get(subclaim.id, subclaim.evidence_ids)
        await self.store.save_subclaims(subclaims)

        for hole in loopholes:
            if hole.id in outcome.loophole_status:
                hole.status = outcome.loophole_status[hole.id]
                hole.resolution = outcome.loophole_resolution.get(hole.id, hole.resolution)
        await self.store.save_loopholes(loopholes)

        await self.log(
            claim,
            AgentName.EVIDENCE_AUDITOR,
            "audit_evidence",
            reason="Turn the measurements from this round into conclusions about each subclaim.",
            inputs={"completed_jobs": [j.action_type for j in jobs if j.state == JobState.COMPLETED]},
            results={
                "subclaims": {s.statement: s.status.value for s in subclaims},
                "open_questions": outcome.open_questions,
            },
            decision=outcome.rationale,
        )

        rounds_left = claim.planning_round + 1 < self.settings.max_planning_rounds
        budget_left = claim.budget.remaining_units > 0
        if outcome.needs_more_testing and rounds_left and budget_left:
            claim.planning_round += 1
            claim.state = ClaimState.SKEPTIC_REVIEW  # re-enter planning on the next tick
            await self.store.save_claim(claim)
            await self.log(
                claim,
                AgentName.EXPERIMENT_PLANNER,
                "recurse",
                reason=outcome.rationale,
                decision=f"Uncertainty remains; planning round {claim.planning_round + 1}.",
            )
            return  # the advance loop re-enters the planner from SKEPTIC_REVIEW

        if outcome.needs_more_testing and not budget_left:
            claim.halt_reason = "The compute budget was exhausted before every question was closed."
        elif outcome.needs_more_testing and not rounds_left:
            claim.halt_reason = (
                f"Reached the {self.settings.max_planning_rounds}-round planning limit with questions still open."
            )
        await self._finalise(claim)

    async def _finalise(self, claim: Claim) -> Verdict:
        # Stay in `auditing` until the verdict is written and stored: clients
        # stop polling once a claim reports a terminal state, so publishing
        # `verdict` early would freeze them on a snapshot with no verdict in it.
        await self._set_state(claim, ClaimState.AUDITING, AgentName.VERDICT_AGENT, "Writing the final verdict")
        subclaims = await self.store.list_subclaims(claim.id)
        loopholes = await self.store.list_loopholes(claim.id)
        jobs = await self.store.list_jobs(claim.id)
        evidence = await self.store.list_evidence(claim.id)

        score = compute_reliability(claim, subclaims, loopholes, jobs, evidence)
        verdict = await self.runtime.write_verdict(claim, subclaims, loopholes, jobs, evidence, score)
        verdict.score = score
        await self.store.save_verdict(verdict)

        ledger = await self.store.list_ledger(claim.id)
        report = render_report(claim, verdict, score, subclaims, loopholes, jobs, evidence, ledger)
        uri = await self.artifacts.write_text(f"{claim.id}/reliability_report.md", report, "text/markdown")

        overrides = getattr(self.runtime, "overrides", [])
        await self.log(
            claim,
            AgentName.VERDICT_AGENT,
            "final_verdict",
            reason="All planned verification finished; issuing the reviewable conclusion.",
            inputs={
                "evidence_items": len(evidence),
                "completed_jobs": sum(1 for j in jobs if j.state == JobState.COMPLETED),
            },
            results={
                "status": verdict.status.value,
                "overall_confidence": score.overall,
                "dimensions": {d.dimension.value: d.score for d in score.dimensions},
                "model_overrides": overrides,
            },
            decision=verdict.headline,
            artifacts=[uri],
        )

        claim.state = ClaimState.VERDICT
        claim.active_agent = None
        claim.latest_action = verdict.headline
        await self.store.save_claim(claim)
        return verdict

    # -- helpers used by the API ----------------------------------------

    async def pending_plan(self, claim_id: str) -> ExperimentPlan | None:
        for plan in await self.store.list_plans(claim_id):
            if plan.status == "awaiting_approval":
                return plan
        return None

    async def plan_items_by_job(self, claim_id: str) -> dict[str, PlanItem]:
        out: dict[str, PlanItem] = {}
        for plan in await self.store.list_plans(claim_id):
            for item in plan.items:
                if item.job_id:
                    out[item.job_id] = item
        return out
