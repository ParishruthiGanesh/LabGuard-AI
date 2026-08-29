"""Experiment worker and RunMedic.

Consumes job messages, runs the requested registry action, watches the run
while it happens, and applies approved repairs.  In cloud mode this module is
deployed as its own Cloud Run service behind a Pub/Sub push subscription; in
demo mode the identical handler is driven by the in-process bus.
"""

from __future__ import annotations

import logging
from typing import Any

from .actions.executors import EXECUTORS, ActionOutcome, ExecutionContext
from .actions.registry import (
    RECOVERY_FOR_ANOMALY,
    ActionValidationError,
    UnknownActionError,
    get_action,
    requires_approval,
)
from .agents.health import (
    MAX_TOTAL_ATTEMPTS,
    Finding,
    analyse_curve,
    classify_failure,
    should_stop_now,
    summarise,
)
from .experiments.trainer import TrainingFailure
from .infra.artifacts import ArtifactStore
from .infra.store import StateStore
from .models.domain import Claim, EpochRecord, HealthEvent, Job, RunHealth, utcnow
from .models.enums import (
    ActionCategory,
    AgentName,
    AnomalyKind,
    HealthStatus,
    JobState,
)
from .orchestrator import Orchestrator

log = logging.getLogger("labguard.worker")

#: How often, in epochs, RunMedic re-examines a live curve.
LIVE_CHECK_EVERY = 4


class ExperimentWorker:
    def __init__(self, store: StateStore, artifacts: ArtifactStore, orchestrator: Orchestrator) -> None:
        self.store = store
        self.artifacts = artifacts
        self.orchestrator = orchestrator

    # -- bus entry point -------------------------------------------------

    async def handle_message(self, message: dict[str, Any]) -> None:
        claim_id, job_id = message.get("claim_id"), message.get("job_id")
        if not claim_id or not job_id:
            log.warning("ignoring malformed job message: %s", message)
            return
        claim = await self.store.get_claim(claim_id)
        job = await self.store.get_job(claim_id, job_id)
        if claim is None or job is None:
            log.warning("job %s/%s no longer exists", claim_id, job_id)
            return
        if job.state.is_terminal:
            return
        await self._execute(claim, job)
        await self.orchestrator.on_job_finished(claim_id)

    # -- execution -------------------------------------------------------

    async def _execute(self, claim: Claim, job: Job) -> None:
        job.attempts += 1
        job.state = JobState.RUNNING
        job.started_at = job.started_at or utcnow()
        job.curves = []
        await self.store.save_job(job)
        await self.orchestrator.log(
            claim,
            AgentName.RUN_MANAGER,
            f"start:{job.action_type}",
            reason=job.reason,
            inputs={"params": job.params, "attempt": job.attempts},
            decision="Run started on the experiment worker.",
            job_id=job.id,
        )

        try:
            spec = get_action(job.action_type)
            params = spec.validate_params(job.params)
        except (UnknownActionError, ActionValidationError) as exc:
            await self._fail_permanently(claim, job, str(exc))
            return

        executor = EXECUTORS.get(job.action_type)
        if executor is None:
            await self._fail_permanently(
                claim, job, f"'{job.action_type}' has no executor; recovery actions are applied to a parent job"
            )
            return

        ctx = self._context(claim, job)
        try:
            outcome: ActionOutcome = await executor(ctx, params)
        except TrainingFailure as failure:
            await self._handle_failure(claim, job, failure)
            return
        except Exception as exc:  # unexpected: record it, do not crash the worker
            log.exception("job %s raised", job.id)
            await self._handle_failure(
                claim, job, TrainingFailure(AnomalyKind.MISSING_ARTIFACT, f"unhandled worker error: {exc}")
            )
            return

        await self._complete(claim, job, outcome)

    def _context(self, claim: Claim, job: Job) -> ExecutionContext:
        """Wire the live-monitoring hooks into an executor."""
        stop_requested = {"value": False}

        async def on_run_start(label: str) -> None:
            job.curves = []
            stop_requested["value"] = False
            job.result = {**job.result, "current_run": label}
            await self.store.save_job(job)

        async def on_epoch(record: EpochRecord) -> None:
            job.curves.append(record)
            if record.epoch % LIVE_CHECK_EVERY == 0 and await self._live_check(claim, job):
                stop_requested["value"] = True
            await self.store.save_job(job)

        async def write_artifact(path: str, payload: Any) -> str:
            return await self.artifacts.write_json(f"{claim.id}/{job.id}/{path}", payload)

        return ExecutionContext(
            claim=claim,
            job=job,
            on_epoch=on_epoch,
            on_run_start=on_run_start,
            write_artifact=write_artifact,
            should_stop=lambda: stop_requested["value"],
        )

    # -- RunMedic --------------------------------------------------------

    async def _live_check(self, claim: Claim, job: Job) -> bool:
        """Watch a live run. Returns True when the run should be stopped."""
        findings = analyse_curve(job.curves)
        if not findings:
            job.health.status = HealthStatus.HEALTHY
            job.health.summary = f"training healthy through epoch {job.curves[-1].epoch}"
            return False
        status, summary = summarise(findings)
        job.health.status = status
        job.health.summary = summary

        # A diagnostic replay exists to characterise the reported run, so it is
        # observed rather than interrupted.
        may_stop = job.category != ActionCategory.DIAGNOSTIC and should_stop_now(job.curves, findings)
        if may_stop:
            spec = get_action("apply_early_stopping")
            needs_approval, _ = requires_approval(
                spec, {}, claim.autonomy_mode, claim.budget.remaining_units, claim.budget.approval_threshold_units
            )
            may_stop = not needs_approval

        for finding in findings:
            if any(e.anomaly == finding.anomaly for e in job.health.events):
                continue  # already reported for this run
            event = await self._record_health_event(claim, job, finding, live=True, stopping=may_stop)
            job.health.events.append(event)
        return may_stop

    async def _record_health_event(
        self, claim: Claim, job: Job, finding: Finding, *, live: bool, stopping: bool = False
    ) -> HealthEvent:
        """Log a detection and the repair that was actually applied."""
        if stopping:
            action_taken, needs_approval = (
                "Applied 'apply_early_stopping': the run was stopped and the best validated checkpoint kept.",
                False,
            )
        else:
            action_taken, needs_approval = self._recovery_decision(claim, job, finding, live=live)
        event = HealthEvent(
            job_id=job.id,
            anomaly=finding.anomaly,
            status=finding.status,
            detail=finding.detail,
            epoch=finding.epoch,
            action_taken=action_taken,
            repaired=stopping,
            requires_approval=needs_approval,
        )
        await self.orchestrator.log(
            claim,
            AgentName.RUN_MEDIC,
            f"detect:{finding.anomaly.value}",
            reason=finding.detail,
            inputs={"epoch": finding.epoch, "observed": finding.evidence or {}},
            results={"health_status": finding.status.value},
            decision=action_taken,
            job_id=job.id,
        )
        return event

    def _recovery_decision(self, claim: Claim, job: Job, finding: Finding, *, live: bool) -> tuple[str, bool]:
        """Decide what, if anything, RunMedic may do about a finding."""
        recovery = RECOVERY_FOR_ANOMALY.get(finding.anomaly.value)
        if recovery is None:
            return "Recorded for the evidence ledger; no repair applies.", False
        if live and job.category == ActionCategory.DIAGNOSTIC:
            # A diagnostic replay exists precisely to characterise the reported
            # run, so stopping it early would destroy the measurement. The
            # repair is recorded and applied to the runs that follow instead.
            return (
                f"Recorded without intervening: this run is a diagnostic replay of the reported "
                f"configuration. '{recovery}' is applied to the verification runs that follow.",
                False,
            )
        spec = get_action(recovery)
        needs_approval, reason = requires_approval(
            spec,
            {},
            claim.autonomy_mode,
            claim.budget.remaining_units,
            claim.budget.approval_threshold_units,
        )
        if needs_approval:
            return f"'{recovery}' requires approval: {reason}", True
        if live:
            # Detected, but the stop condition has not been met yet: the
            # validation metric is still improving, so the run continues.
            return (
                f"Detected and being watched. '{recovery}' fires once the validation metric also stops improving.",
                False,
            )
        return f"Applying approved recovery '{recovery}'.", False

    async def _handle_failure(self, claim: Claim, job: Job, failure: TrainingFailure) -> None:
        """Classify a failed run, then repair, retry or stop."""
        signature = failure.signature
        job.error = str(failure)
        job.failure_signature = signature

        finding = classify_failure(job, signature)
        job.recovery_actions.append(f"failure:{signature}")

        if finding.anomaly in (AnomalyKind.RECOVERY_LOOP, AnomalyKind.EXCESSIVE_RETRIES):
            event = HealthEvent(
                job_id=job.id,
                anomaly=finding.anomaly,
                status=HealthStatus.CRITICAL,
                detail=finding.detail,
                action_taken="Automatic execution paused for this job to stop it burning budget.",
            )
            job.health.events.append(event)
            job.health.status = HealthStatus.CRITICAL
            job.health.summary = finding.detail
            job.state = JobState.BLOCKED_LOOP
            job.finished_at = utcnow()
            await self.store.save_job(job)
            await self.orchestrator.log(
                claim,
                AgentName.RUN_MEDIC,
                f"loop_detected:{job.action_type}",
                reason=finding.detail,
                inputs={"attempts": job.attempts, "failure_signature": signature},
                results={"state": job.state.value},
                decision=(
                    f"Stopped after {job.attempts} attempts producing the identical failure. "
                    f"Escalated to the researcher rather than retrying again."
                ),
                job_id=job.id,
            )
            return

        recovery = RECOVERY_FOR_ANOMALY.get(finding.anomaly.value)
        action_taken, needs_approval = self._recovery_decision(claim, job, finding, live=False)
        event = HealthEvent(
            job_id=job.id,
            anomaly=finding.anomaly,
            status=HealthStatus.CRITICAL,
            detail=finding.detail,
            epoch=failure.epoch,
            action_taken=action_taken,
            requires_approval=needs_approval,
        )
        job.health.events.append(event)
        job.health.status = HealthStatus.CRITICAL
        job.health.summary = finding.detail
        await self.orchestrator.log(
            claim,
            AgentName.RUN_MEDIC,
            f"detect:{finding.anomaly.value}",
            reason=finding.detail,
            inputs={"attempt": job.attempts, "epoch": failure.epoch},
            results={"proposed_recovery": recovery},
            decision=action_taken,
            job_id=job.id,
        )

        if recovery is None or needs_approval or job.attempts > MAX_TOTAL_ATTEMPTS:
            job.state = JobState.AWAITING_APPROVAL if needs_approval else JobState.FAILED
            job.finished_at = utcnow()
            await self.store.save_job(job)
            return

        applied = self._apply_recovery(claim, job, recovery)
        job.recovery_actions.append(f"recovery:{recovery}")
        job.state = JobState.RECOVERING
        await self.store.save_job(job)
        await self.orchestrator.log(
            claim,
            AgentName.RUN_MEDIC,
            f"recover:{recovery}",
            reason=finding.detail,
            inputs=applied,
            results={"attempt": job.attempts + 1},
            decision=f"Applied '{recovery}' and re-queued the run.",
            job_id=job.id,
        )
        await self.orchestrator.queue_job(claim, job)

    def _apply_recovery(self, claim: Claim, job: Job, recovery: str) -> dict[str, Any]:
        """Mutate the job's parameters inside the registry's declared bounds."""
        from .experiments.scenario import arms, find_config

        spec = get_action(recovery)
        overrides = dict(job.params.get("_recovery_overrides") or {})
        config = find_config(claim.context, str(job.params.get("config_name") or "")) or arms(claim.context)[1]

        if recovery == "adjust_learning_rate_within_bounds":
            bounds = spec.params_model()
            current = float(overrides.get("learning_rate") or config.learning_rate)
            proposed = current * float(bounds.factor)
            applied = min(max(proposed, float(bounds.min_lr)), float(bounds.max_lr))
            overrides["learning_rate"] = applied
            detail = {
                "previous_learning_rate": current,
                "new_learning_rate": applied,
                "bounds": [bounds.min_lr, bounds.max_lr],
            }
        elif recovery == "reduce_batch_size":
            bounds = spec.params_model()
            current = int(overrides.get("batch_size") or config.batch_size)
            applied = max(int(bounds.min_batch_size), int(current * float(bounds.factor)))
            overrides["batch_size"] = applied
            detail = {"previous_batch_size": current, "new_batch_size": applied}
        elif recovery == "apply_early_stopping":
            bounds = spec.params_model()
            overrides["early_stopping_patience"] = int(bounds.patience)
            detail = {"early_stopping_patience": int(bounds.patience)}
        else:
            detail = {"recovery": recovery, "change": "re-running with the same configuration"}

        job.params = {**job.params, "_recovery_overrides": overrides}
        return detail

    # -- terminal states -------------------------------------------------

    async def _complete(self, claim: Claim, job: Job, outcome: ActionOutcome) -> None:
        findings = analyse_curve(job.curves)
        for finding in findings:
            if any(e.anomaly == finding.anomaly for e in job.health.events):
                continue
            job.health.events.append(await self._record_health_event(claim, job, finding, live=False))
        status, summary = summarise(findings)
        repairs = len([r for r in job.recovery_actions if r.startswith("recovery:")]) + sum(
            1 for e in job.health.events if e.repaired
        )
        if repairs:
            status = HealthStatus.RECOVERED
            summary = f"recovered after {repairs} repair(s); {summary}"
        job.health = RunHealth(
            status=status,
            summary=summary,
            events=job.health.events,
            peak_memory_mb=max((c.memory_mb for c in job.curves), default=0.0),
            mean_gpu_util_pct=round(sum(c.gpu_util_pct for c in job.curves) / len(job.curves), 1)
            if job.curves
            else 0.0,
        )
        if outcome.curves:
            job.curves = outcome.curves
        job.result = dict(outcome.result)
        job.artifact_uris = outcome.artifacts
        job.actual_cost_units = job.estimated_cost_units
        job.state = JobState.COMPLETED
        job.finished_at = utcnow()
        await self.store.save_job(job)

        for item in outcome.evidence:
            item.claim_id = claim.id
            item.job_id = job.id
            item.artifact_uris = outcome.artifacts
        await self.store.save_evidence(outcome.evidence)

        claim.budget.consumed_units = round(claim.budget.consumed_units + job.actual_cost_units, 3)
        await self.store.save_claim(claim)

        await self.orchestrator.log(
            claim,
            AgentName.EVIDENCE_AUDITOR if outcome.evidence else AgentName.RUN_MANAGER,
            f"complete:{job.action_type}",
            reason=job.reason,
            inputs={"params": {k: v for k, v in job.params.items() if not k.startswith("_")}},
            results={
                "health": job.health.status.value,
                "evidence": [e.statement for e in outcome.evidence],
                "cost_units": job.actual_cost_units,
            },
            decision=f"Completed; {len(outcome.evidence)} evidence item(s) recorded.",
            job_id=job.id,
            artifacts=outcome.artifacts,
        )

    async def _fail_permanently(self, claim: Claim, job: Job, error: str) -> None:
        job.state = JobState.FAILED
        job.error = error
        job.finished_at = utcnow()
        job.health.status = HealthStatus.CRITICAL
        job.health.summary = error
        await self.store.save_job(job)
        await self.orchestrator.log(
            claim,
            AgentName.RUN_MANAGER,
            f"reject:{job.action_type}",
            reason=error,
            decision="Job rejected before execution; nothing was run.",
            job_id=job.id,
        )
