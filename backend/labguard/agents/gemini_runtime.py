"""Gemini + Google ADK reasoning backend.

Each logical agent is an ADK `LlmAgent` with a strict `output_schema`, run
through an ADK `Runner` so the framework owns the session, the invocation and
the event stream.  The agents are deliberately given **no tools**: LabGuard's
side effects all go through the typed action registry and the experiment
worker, never through model-issued calls.

Every model response is treated as a proposal.  Statuses, scores, costs,
approval requirements and the verdict itself are re-derived from the
deterministic rule engine, and any disagreement is recorded rather than
silently accepted.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, Field

from ..config import Settings
from ..models.domain import (
    AlternativeExplanation,
    Claim,
    Evidence,
    ExperimentPlan,
    Job,
    Loophole,
    ReliabilityScore,
    Subclaim,
    Verdict,
)
from ..models.enums import LoopholeKind, LoopholeStatus, VerdictStatus
from . import deterministic as rules
from .base import AuditOutcome

log = logging.getLogger("labguard.gemini")

APP_NAME = "labguard"


# --------------------------------------------------------------------------
# Structured response schemas
# --------------------------------------------------------------------------


class SubclaimDraft(BaseModel):
    key: str = Field(description="The exact key supplied in the input; do not invent new keys.")
    statement: str = Field(description="The subclaim restated in the researcher's own terms.")
    measurable_quantity: str = Field(description="The quantity that would settle it.")
    rationale: str = Field(description="Why this subclaim matters for the parent claim.")


class SubclaimsResponse(BaseModel):
    subclaims: list[SubclaimDraft]


class LoopholeDraft(BaseModel):
    kind: str = Field(description="One of the allowed loophole kinds, exactly as spelled in the input.")
    title: str
    rationale: str
    severity: float = Field(ge=0.0, le=1.0)


class LoopholesResponse(BaseModel):
    loopholes: list[LoopholeDraft]
    alternative_explanations: list[str]


class PlanChoice(BaseModel):
    action_type: str = Field(description="An action name from the supplied candidate list.")
    reason: str = Field(description="Why this action is the most informative next step.")
    expected_information_gain: float = Field(ge=0.0, le=1.0)


class PlanResponse(BaseModel):
    chosen: list[PlanChoice]
    summary: str


class AuditResponse(BaseModel):
    rationale: str
    open_questions: list[str]


class VerdictResponse(BaseModel):
    status: str = Field(description="One of the allowed verdict statuses.")
    headline: str
    narrative: str
    evidence_summary: list[str]
    remaining_uncertainty: list[str]


# --------------------------------------------------------------------------
# Model resolution
# --------------------------------------------------------------------------


def resolve_model(settings: Settings) -> str:
    """Pick the first configured model id the deployment actually exposes.

    `GEMINI_MODEL` may name a model this project has not been granted; rather
    than failing the whole run, fall through the configured candidates.
    """
    candidates = settings.model_candidates
    try:
        from google import genai

        client = (
            genai.Client(vertexai=True, project=settings.google_cloud_project, location=settings.google_cloud_region)
            if settings.use_vertex_ai
            else genai.Client(api_key=settings.gemini_api_key)
        )
        available = {m.name.split("/")[-1] for m in client.models.list() if m.name}
        for candidate in candidates:
            if candidate in available:
                return candidate
        log.warning("none of %s were listed by the API; using %s", candidates, candidates[0])
    except Exception as exc:  # pragma: no cover - network dependent
        log.warning("could not list Gemini models (%s); using %s", exc, candidates[0])
    return candidates[0]


class AdkAgentRuntime:
    """The seven agents, backed by Gemini through Google ADK."""

    name = "gemini-adk"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model = resolve_model(settings)
        self.name = f"gemini-adk:{self.model}"
        self._agents: dict[str, Any] = {}
        self._runners: dict[str, Any] = {}
        #: Disagreements between the model and the rule engine, surfaced in
        #: the ledger so a reviewer can see where the model was overruled.
        self.overrides: list[dict[str, Any]] = []

    # -- ADK plumbing ----------------------------------------------------

    def _agent(self, name: str, instruction: str, schema: type[BaseModel]) -> Any:
        from google.adk.agents import LlmAgent

        if name not in self._agents:
            self._agents[name] = LlmAgent(
                name=name,
                model=self.model,
                description=instruction.strip().splitlines()[0],
                instruction=instruction,
                output_schema=schema,
                output_key=name,
                # No tools by design: the model proposes, the registry disposes.
                disallow_transfer_to_parent=True,
                disallow_transfer_to_peers=True,
            )
        return self._agents[name]

    async def _invoke(
        self, name: str, instruction: str, schema: type[BaseModel], payload: dict[str, Any]
    ) -> BaseModel | None:
        """Run one ADK agent and parse its structured response."""
        from google.adk.runners import InMemoryRunner
        from google.genai import types

        agent = self._agent(name, instruction, schema)
        if name not in self._runners:
            self._runners[name] = InMemoryRunner(agent=agent, app_name=APP_NAME)
        runner = self._runners[name]

        session = await runner.session_service.create_session(app_name=APP_NAME, user_id="labguard")
        message = types.Content(role="user", parts=[types.Part(text=json.dumps(payload, default=str))])
        text = ""
        try:
            async for event in runner.run_async(user_id="labguard", session_id=session.id, new_message=message):
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if part.text:
                            text = part.text
        except Exception as exc:  # pragma: no cover - network dependent
            log.warning("ADK agent %s failed (%s); falling back to the rule engine", name, exc)
            return None
        try:
            return schema.model_validate_json(text)
        except Exception as exc:
            log.warning("agent %s returned unparseable output (%s)", name, exc)
            return None

    # -- Claim Analyst ---------------------------------------------------

    async def analyse_claim(self, claim: Claim) -> list[Subclaim]:
        base = rules.build_subclaims(claim)
        payload = {
            "claim": claim.text,
            "dataset": claim.context.dataset.model_dump(mode="json"),
            "models": [m.model_dump(mode="json") for m in claim.context.models],
            "existing_results": [r.model_dump(mode="json") for r in claim.context.existing_results],
            "subclaims_to_restate": [
                {"key": rules.subclaim_key(s), "statement": s.statement, "measurable_quantity": s.measurable_quantity}
                for s in base
            ],
        }
        instruction = """
You are the Claim Analyst on a research reliability platform.
Restate each supplied subclaim in the concrete terms of THIS claim, dataset and
model configurations, so a reviewer can see exactly what would be measured.

Rules:
- Return exactly the keys you were given. Do not add, drop or rename keys.
- Keep each statement to one sentence, falsifiable, and tied to a number.
- `measurable_quantity` must name a quantity that can actually be computed from
  a train/test split, not a vague notion of quality.
"""
        response = await self._invoke("claim_analyst", instruction, SubclaimsResponse, payload)
        if response is None:
            return base

        by_key = {rules.subclaim_key(s): s for s in base}
        for draft in response.subclaims:
            target = by_key.get(draft.key)
            if target is None:  # model invented a key: ignore it
                self.overrides.append(
                    {"agent": "claim_analyst", "reason": f"unknown subclaim key '{draft.key}' dropped"}
                )
                continue
            target.statement = draft.statement.strip() or target.statement
            target.measurable_quantity = draft.measurable_quantity.strip() or target.measurable_quantity
            target.rationale = draft.rationale.strip() or target.rationale
        return base

    # -- Scientific Skeptic ----------------------------------------------

    async def find_loopholes(
        self, claim: Claim, subclaims: Sequence[Subclaim]
    ) -> tuple[list[Loophole], list[AlternativeExplanation]]:
        detected, alternatives = rules.detect_loopholes(claim, subclaims)
        payload = {
            "claim": claim.text,
            "dataset": claim.context.dataset.model_dump(mode="json"),
            "models": [m.model_dump(mode="json") for m in claim.context.models],
            "existing_results": [r.model_dump(mode="json") for r in claim.context.existing_results],
            "already_detected": [{"kind": h.kind.value, "title": h.title, "severity": h.severity} for h in detected],
            "allowed_kinds": [k.value for k in LoopholeKind],
        }
        instruction = """
You are the Scientific Skeptic. Your job is to find reasons the reported result
might not mean what it appears to mean.

Rules:
- `kind` MUST be one of `allowed_kinds`, spelled exactly.
- The entries in `already_detected` were found by static analysis of the
  submission. You may sharpen their wording, and you may add kinds that were
  missed. Do not contradict a detected fact.
- `severity` is how much the claim would be damaged if the weakness is real.
- Ground every rationale in the supplied configuration or dataset numbers. Do
  not speculate about data you were not given.
- Alternative explanations must be concrete rival accounts of the same result.
"""
        response = await self._invoke("scientific_skeptic", instruction, LoopholesResponse, payload)
        if response is None:
            return detected, alternatives

        by_kind = {h.kind: h for h in detected}
        for draft in response.loopholes:
            try:
                kind = LoopholeKind(draft.kind)
            except ValueError:
                self.overrides.append(
                    {"agent": "scientific_skeptic", "reason": f"unknown loophole kind '{draft.kind}' dropped"}
                )
                continue
            if kind in by_kind:
                # Keep the rule engine's severity: it is grounded in the data.
                by_kind[kind].rationale = draft.rationale.strip() or by_kind[kind].rationale
                by_kind[kind].title = draft.title.strip() or by_kind[kind].title
                by_kind[kind].detected_by = "heuristic+gemini"
            else:
                extra = Loophole(
                    claim_id=claim.id,
                    kind=kind,
                    title=draft.title.strip(),
                    rationale=draft.rationale.strip(),
                    # A model-proposed weakness is capped below the threshold
                    # that can block a verdict on its own.
                    severity=min(0.6, float(draft.severity)),
                    status=LoopholeStatus.OPEN,
                    detected_by="gemini",
                )
                extra.id = f"hole_{kind.value}"
                detected.append(extra)
                by_kind[kind] = extra

        for statement in response.alternative_explanations[:6]:
            if statement.strip():
                alternatives.append(AlternativeExplanation(claim_id=claim.id, statement=statement.strip()))
        return detected, alternatives

    # -- Experiment Planner ----------------------------------------------

    async def plan_round(
        self,
        claim: Claim,
        subclaims: Sequence[Subclaim],
        loopholes: Sequence[Loophole],
        jobs: Sequence[Job],
        evidence: Sequence[Evidence],
        round_index: int,
    ) -> ExperimentPlan:
        # The rule engine produces the feasible set: deduplicated, inside
        # budget, with costs and approval requirements already decided.
        plan = rules.plan_next_round(claim, subclaims, loopholes, jobs, round_index)
        if len(plan.items) <= 1:
            return plan

        payload = {
            "claim": claim.text,
            "round": round_index + 1,
            "remaining_budget_units": claim.budget.remaining_units,
            "open_loopholes": [
                {"kind": h.kind.value, "title": h.title, "severity": h.severity}
                for h in loopholes
                if h.status in (LoopholeStatus.OPEN, LoopholeStatus.INVESTIGATING)
            ],
            "untested_subclaims": [s.statement for s in subclaims if s.status.value in ("untested", "testing")],
            "evidence_so_far": [{"stance": e.stance.value, "statement": e.statement} for e in evidence][-12:],
            "candidate_actions": [
                {
                    "action_type": i.action_type,
                    "summary": i.reason,
                    "estimated_cost_units": i.estimated_cost_units,
                    "requires_approval": i.requires_approval,
                }
                for i in plan.items
            ],
        }
        instruction = """
You are the Experiment Planner. Choose which of the candidate actions to run
next and in what order, to remove the most uncertainty per compute unit.

Rules:
- `action_type` MUST come from `candidate_actions`. You cannot invent actions,
  change their parameters, or propose shell commands.
- Prefer cheap actions that can settle a high-severity loophole outright.
- Order the list so that the results of earlier actions inform later ones.
- Drop an action only if it cannot change any conclusion; say so in `summary`.
- `reason` must state which loophole or subclaim the action settles.
"""
        response = await self._invoke("experiment_planner", instruction, PlanResponse, payload)
        if response is None:
            return plan

        by_action = {i.action_type: i for i in plan.items}
        ordered = []
        for choice in response.chosen:
            item = by_action.pop(choice.action_type, None)
            if item is None:
                self.overrides.append(
                    {
                        "agent": "experiment_planner",
                        "reason": f"action '{choice.action_type}' is not in the candidate set; ignored",
                    }
                )
                continue
            item.reason = choice.reason.strip() or item.reason
            item.expected_information_gain = round(float(choice.expected_information_gain), 2)
            ordered.append(item)
        # Anything the model dropped stays available for a later round rather
        # than being lost, but does not run now.
        if not ordered:
            return plan
        plan.items = ordered
        plan.total_cost_units = round(sum(i.estimated_cost_units for i in plan.items), 2)
        plan.requires_approval = any(i.requires_approval for i in plan.items)
        plan.summary = response.summary.strip() or plan.summary
        return plan

    # -- Evidence Auditor ------------------------------------------------

    async def audit(
        self,
        claim: Claim,
        subclaims: Sequence[Subclaim],
        loopholes: Sequence[Loophole],
        jobs: Sequence[Job],
        evidence: Sequence[Evidence],
    ) -> AuditOutcome:
        outcome = rules.audit_evidence(claim, subclaims, loopholes, jobs, evidence)
        payload = {
            "claim": claim.text,
            "measurements": [
                {"action": j.action_type, "state": j.state.value, "result": j.result} for j in jobs if j.result
            ],
            "evidence": [
                {"stance": e.stance.value, "statement": e.statement, "measurements": e.measurements} for e in evidence
            ],
            "computed_subclaim_status": {
                s.statement: outcome.subclaim_status.get(s.id, s.status).value for s in subclaims
            },
            "run_health_incidents": [
                {
                    "action": j.action_type,
                    "anomaly": ev.anomaly.value,
                    "detail": ev.detail,
                    "action_taken": ev.action_taken,
                }
                for j in jobs
                for ev in j.health.events
            ],
        }
        instruction = """
You are the Evidence Auditor. The statuses in `computed_subclaim_status` were
derived arithmetically from the measurements and are FINAL - do not restate,
dispute or recompute them.

Your job is to explain, in two to four sentences, what the measurements
collectively show, and to list the questions the evidence still leaves open.

Rules:
- Quote real numbers from `measurements`; never invent or round-trip a figure
  that is not there.
- `open_questions` should each name something a further experiment could settle.
- Mention any run-health incident that affects how the numbers should be read.
"""
        response = await self._invoke("evidence_auditor", instruction, AuditResponse, payload)
        if response is not None:
            # The narrative comes from the model; the decision does not.
            outcome.rationale = f"{outcome.rationale} {response.rationale.strip()}".strip()
            outcome.open_questions.extend(q.strip() for q in response.open_questions if q.strip())
        return outcome

    # -- Verdict Agent ---------------------------------------------------

    async def write_verdict(
        self,
        claim: Claim,
        subclaims: Sequence[Subclaim],
        loopholes: Sequence[Loophole],
        jobs: Sequence[Job],
        evidence: Sequence[Evidence],
        score: ReliabilityScore,
    ) -> Verdict:
        verdict = rules.build_verdict(claim, subclaims, loopholes, jobs, evidence, score)
        payload = {
            "claim": claim.text,
            "rule_based_status": verdict.status.value,
            "rule_based_reasons": verdict.narrative,
            "allowed_statuses": [v.value for v in VerdictStatus],
            "evidence": [{"stance": e.stance.value, "statement": e.statement} for e in evidence],
            "reliability_dimensions": [
                {"dimension": d.dimension.value, "score": d.score, "calculation": d.calculation}
                for d in score.dimensions
            ],
            "run_health_incidents": verdict.run_health_incidents,
            "remaining_uncertainty": verdict.remaining_uncertainty,
        }
        instruction = """
You are the Verdict Agent. Write the final, reviewable conclusion.

Rules:
- `status` must be one of `allowed_statuses`.
- Do NOT output any numeric score. The reliability scores are already computed
  and are supplied to you only as context.
- Every sentence of `narrative` must be traceable to an item in `evidence` or
  `reliability_dimensions`. Quote the actual figures.
- Name what would change the verdict, in `remaining_uncertainty`.
- Be direct. If the evidence does not support the claim, say so plainly.
"""
        response = await self._invoke("verdict_agent", instruction, VerdictResponse, payload)
        if response is None:
            return verdict

        verdict.generated_by = self.name
        verdict.headline = response.headline.strip() or verdict.headline
        verdict.narrative = response.narrative.strip() or verdict.narrative
        if response.evidence_summary:
            verdict.evidence_summary = [s.strip() for s in response.evidence_summary if s.strip()]
        if response.remaining_uncertainty:
            verdict.remaining_uncertainty = [s.strip() for s in response.remaining_uncertainty if s.strip()]

        # The rule engine has the last word on the status itself.
        try:
            proposed = VerdictStatus(response.status)
        except ValueError:
            proposed = verdict.rule_based_status
        if proposed != verdict.rule_based_status:
            self.overrides.append(
                {
                    "agent": "verdict_agent",
                    "reason": (
                        f"model proposed '{proposed.value}' but the measured evidence gives "
                        f"'{verdict.rule_based_status.value}'; the measured status stands"
                    ),
                }
            )
        verdict.status = verdict.rule_based_status
        return verdict
