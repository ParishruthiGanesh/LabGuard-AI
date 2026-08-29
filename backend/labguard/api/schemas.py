"""Request and response bodies for the HTTP API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

from ..models.domain import (
    AlternativeExplanation,
    BudgetPolicy,
    Claim,
    ClaimContext,
    Evidence,
    ExperimentPlan,
    Job,
    LedgerEntry,
    Loophole,
    ReliabilityScore,
    Subclaim,
    Verdict,
)
from ..models.enums import AutonomyMode


class CreateClaimRequest(BaseModel):
    text: str = Field(default="", max_length=2000)
    context: ClaimContext = Field(default_factory=ClaimContext)
    autonomy_mode: AutonomyMode = AutonomyMode.MANAGED_AUTONOMY
    #: Omit to keep the scenario's own budget policy.
    budget: BudgetPolicy | None = None
    #: Start from the bundled synthetic scenario instead of the supplied context.
    use_demo_scenario: bool = False

    @model_validator(mode="after")
    def _require_text_unless_demo(self) -> CreateClaimRequest:
        """The demo scenario brings its own claim; anything else must state one."""
        if not self.use_demo_scenario and len(self.text.strip()) < 8:
            raise ValueError("text must be at least 8 characters when use_demo_scenario is false")
        if not self.use_demo_scenario and len(self.context.models) < 2:
            raise ValueError("supply at least two model configurations to compare")
        return self


class PlanDecisionRequest(BaseModel):
    approved: bool
    decided_by: str = Field(default="researcher", max_length=120)


class JobApprovalRequest(BaseModel):
    approved: bool
    decided_by: str = Field(default="researcher", max_length=120)


class ClaimSnapshot(BaseModel):
    """Everything the dashboard needs for one claim, in a single poll."""

    claim: Claim
    subclaims: list[Subclaim]
    loopholes: list[Loophole]
    alternatives: list[AlternativeExplanation]
    plans: list[ExperimentPlan]
    jobs: list[Job]
    evidence: list[Evidence]
    ledger: list[LedgerEntry]
    verdict: Verdict | None
    score: ReliabilityScore | None
    #: Bumped whenever anything changes, so the client can skip re-rendering.
    revision: int
    infrastructure: dict[str, str]
    report_available: bool


class ActionSpecView(BaseModel):
    name: str
    category: str
    summary: str
    base_cost_units: float
    max_retries: int
    min_autonomy: str
    invoked_by: str
    addresses: list[str]
    parameters: dict[str, Any]


class ConfigResponse(BaseModel):
    infrastructure: dict[str, str]
    autonomy_modes: list[str]
    actions: list[ActionSpecView]
    demo_scenario: dict[str, Any]
