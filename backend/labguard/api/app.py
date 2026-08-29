"""LabGuard AI HTTP API.

One FastAPI application serves both Cloud Run services.  The API service
handles the dashboard; the worker service is the same image with a Pub/Sub
push subscription pointed at `/internal/pubsub/push`.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import BackgroundTasks, Body, FastAPI, Header, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware

from ..actions.registry import REGISTRY
from ..config import Settings, get_settings
from ..experiments import scenario
from ..models.domain import BudgetPolicy, Claim
from ..models.enums import ClaimState, JobState
from ..scoring.reliability import compute_reliability
from ..services import Services
from .schemas import (
    ActionSpecView,
    ClaimSnapshot,
    ConfigResponse,
    CreateClaimRequest,
    JobApprovalRequest,
    PlanDecisionRequest,
)

log = logging.getLogger("labguard.api")

_services: Services | None = None
#: Strong references to in-flight push handlers; without these the event loop
#: may garbage-collect a running task mid-job.
_push_tasks: set[asyncio.Task[None]] = set()


def get_services() -> Services:
    if _services is None:  # pragma: no cover - set during lifespan
        raise RuntimeError("services are not initialised")
    return _services


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _services
    _services = Services()
    await _services.start()
    log.info("LabGuard started: %s", _services.infrastructure)
    try:
        yield
    finally:
        await _services.close()
        _services = None


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(
        title="LabGuard AI",
        version="1.0.0",
        summary="Autonomous research reliability platform: challenge the claim, protect the run.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -- meta ------------------------------------------------------------

    @app.get("/health", tags=["meta"])
    async def health() -> dict[str, Any]:
        services = get_services()
        return {"status": "ok", "infrastructure": services.infrastructure}

    @app.get("/api/config", response_model=ConfigResponse, tags=["meta"])
    async def config() -> ConfigResponse:
        services = get_services()
        demo = scenario.demo_claim()
        return ConfigResponse(
            infrastructure=services.infrastructure,
            autonomy_modes=["observe_only", "safe_repair", "managed_autonomy"],
            actions=[
                ActionSpecView(
                    name=spec.name,
                    category=spec.category.value,
                    summary=spec.summary,
                    base_cost_units=spec.base_cost_units,
                    max_retries=spec.max_retries,
                    min_autonomy=spec.min_autonomy.value,
                    invoked_by=spec.invoked_by,
                    addresses=list(spec.addresses),
                    parameters=spec.params_model.model_json_schema(),
                )
                for spec in sorted(REGISTRY.values(), key=lambda s: (s.category.value, s.name))
            ],
            demo_scenario={
                "text": demo.text,
                "context": demo.context.model_dump(mode="json"),
                "budget": demo.budget.model_dump(mode="json"),
                "autonomy_mode": demo.autonomy_mode.value,
            },
        )

    # -- claims ----------------------------------------------------------

    @app.post("/api/claims", tags=["claims"], status_code=201)
    async def create_claim(request: CreateClaimRequest, background: BackgroundTasks) -> Claim:
        services = get_services()
        if request.use_demo_scenario:
            claim = scenario.demo_claim(request.autonomy_mode)
            if request.budget is not None:
                claim.budget = request.budget
        else:
            claim = Claim(
                text=request.text,
                context=request.context,
                autonomy_mode=request.autonomy_mode,
                budget=request.budget or BudgetPolicy(),
                demo_mode=not services.settings.is_cloud,
            )
        claim = await services.orchestrator.create_claim(claim)
        background.add_task(_drive, services, claim.id)
        return claim

    @app.get("/api/claims", tags=["claims"])
    async def list_claims() -> list[Claim]:
        return await get_services().store.list_claims()

    @app.get("/api/claims/{claim_id}", response_model=ClaimSnapshot, tags=["claims"])
    async def get_claim(claim_id: str) -> ClaimSnapshot:
        return await _snapshot(get_services(), claim_id)

    @app.post("/api/claims/{claim_id}/plans/{plan_id}/decision", tags=["approvals"])
    async def decide_plan(
        claim_id: str, plan_id: str, request: PlanDecisionRequest, background: BackgroundTasks
    ) -> Claim:
        services = get_services()
        claim = await services.orchestrator.decide_plan(claim_id, plan_id, request.approved, request.decided_by)
        if claim is None:
            raise HTTPException(status_code=404, detail="claim or plan not found")
        if request.approved:
            background.add_task(_drive, services, claim_id)
        return claim

    @app.post("/api/claims/{claim_id}/jobs/{job_id}/decision", tags=["approvals"])
    async def decide_job(
        claim_id: str, job_id: str, request: JobApprovalRequest, background: BackgroundTasks
    ) -> dict[str, Any]:
        """Approve or decline a recovery that needed a human decision."""
        services = get_services()
        claim = await services.store.get_claim(claim_id)
        job = await services.store.get_job(claim_id, job_id)
        if claim is None or job is None:
            raise HTTPException(status_code=404, detail="claim or job not found")
        if job.state != JobState.AWAITING_APPROVAL:
            raise HTTPException(status_code=409, detail=f"job is {job.state.value}, not awaiting approval")

        if request.approved:
            await services.orchestrator.queue_job(claim, job)
        else:
            job.state = JobState.REJECTED
            await services.store.save_job(job)
            background.add_task(services.orchestrator.on_job_finished, claim_id)
        return {"job_id": job.id, "state": job.state.value}

    @app.get("/api/claims/{claim_id}/report", tags=["claims"])
    async def get_report(claim_id: str) -> Response:
        services = get_services()
        text = await services.artifacts.read_text(f"{claim_id}/reliability_report.md")
        if text is None:
            raise HTTPException(status_code=404, detail="the report is generated once a verdict is reached")
        return Response(
            content=text,
            media_type="text/markdown",
            headers={"Content-Disposition": f'attachment; filename="labguard_report_{claim_id}.md"'},
        )

    @app.get("/api/claims/{claim_id}/artifacts/{path:path}", tags=["claims"])
    async def get_artifact(claim_id: str, path: str) -> Response:
        services = get_services()
        text = await services.artifacts.read_text(f"{claim_id}/{path}")
        if text is None:
            raise HTTPException(status_code=404, detail="artifact not found")
        media = "application/json" if path.endswith(".json") else "text/plain"
        return Response(content=text, media_type=media)

    # -- worker push endpoint --------------------------------------------

    @app.post("/internal/pubsub/push", tags=["internal"], status_code=204)
    async def pubsub_push(
        envelope: dict[str, Any] = Body(...),
        authorization: str | None = Header(default=None),
    ) -> Response:
        """Pub/Sub push delivery for the worker Cloud Run service.

        Always answers 204 once the message is accepted: a non-2xx would make
        Pub/Sub redeliver, and job retries are RunMedic's decision, not the
        transport's.
        """
        services = get_services()
        secret = services.settings.worker_shared_secret
        if secret and authorization != f"Bearer {secret}":
            raise HTTPException(status_code=401, detail="invalid worker credentials")

        message = envelope.get("message") or {}
        raw = message.get("data")
        if not raw:
            return Response(status_code=204)
        try:
            payload = json.loads(base64.b64decode(raw).decode("utf-8"))
        except Exception as exc:
            log.error("undecodable Pub/Sub payload: %s", exc)
            return Response(status_code=204)

        task = asyncio.create_task(services.worker.handle_message(payload))
        _push_tasks.add(task)
        task.add_done_callback(_push_tasks.discard)
        return Response(status_code=204)

    return app


async def _drive(services: Services, claim_id: str) -> None:
    try:
        await services.orchestrator.advance(claim_id)
    except Exception:  # pragma: no cover - background task safety net
        log.exception("failed to advance claim %s", claim_id)


async def _snapshot(services: Services, claim_id: str) -> ClaimSnapshot:
    claim = await services.store.get_claim(claim_id)
    if claim is None:
        raise HTTPException(status_code=404, detail="claim not found")

    subclaims = await services.store.list_subclaims(claim_id)
    loopholes = await services.store.list_loopholes(claim_id)
    alternatives = await services.store.list_alternatives(claim_id)
    plans = await services.store.list_plans(claim_id)
    jobs = await services.store.list_jobs(claim_id)
    evidence = await services.store.list_evidence(claim_id)
    ledger = await services.store.list_ledger(claim_id)
    verdict = await services.store.get_verdict(claim_id)

    # The curve is already on the job; drop the duplicate copy from the result
    # payload so a poll stays small.
    for job in jobs:
        if "curve" in job.result:
            job.result = {k: v for k, v in job.result.items() if k != "curve"}

    score = verdict.score if verdict else compute_reliability(claim, subclaims, loopholes, jobs, evidence)
    revision = len(ledger) * 1000 + sum(len(j.curves) for j in jobs) + len(evidence)
    return ClaimSnapshot(
        claim=claim,
        subclaims=subclaims,
        loopholes=loopholes,
        alternatives=alternatives,
        plans=plans,
        jobs=jobs,
        evidence=evidence,
        ledger=ledger,
        verdict=verdict,
        score=score,
        revision=revision,
        infrastructure=services.infrastructure,
        report_available=claim.state == ClaimState.VERDICT and verdict is not None,
    )


app = create_app()
