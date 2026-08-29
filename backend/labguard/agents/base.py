"""The agent interface shared by both reasoning backends."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

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
from ..models.enums import LoopholeStatus, SubclaimStatus


@dataclass
class AuditOutcome:
    """What the Evidence Auditor concluded from the completed jobs."""

    subclaim_status: dict[str, SubclaimStatus] = field(default_factory=dict)
    subclaim_confidence: dict[str, float] = field(default_factory=dict)
    subclaim_evidence: dict[str, list[str]] = field(default_factory=dict)
    loophole_status: dict[str, LoopholeStatus] = field(default_factory=dict)
    loophole_resolution: dict[str, str] = field(default_factory=dict)
    needs_more_testing: bool = False
    rationale: str = ""
    open_questions: list[str] = field(default_factory=list)


class AgentRuntime(Protocol):
    """The seven logical agents, as one interface over shared state.

    Both implementations produce the same object graph, so the orchestrator,
    the API and the dashboard cannot tell them apart.
    """

    name: str

    async def analyse_claim(self, claim: Claim) -> list[Subclaim]:
        """Claim Analyst: decompose into measurable subclaims."""

    async def find_loopholes(
        self, claim: Claim, subclaims: Sequence[Subclaim]
    ) -> tuple[list[Loophole], list[AlternativeExplanation]]:
        """Scientific Skeptic: enumerate weaknesses and alternative explanations."""

    async def plan_round(
        self,
        claim: Claim,
        subclaims: Sequence[Subclaim],
        loopholes: Sequence[Loophole],
        jobs: Sequence[Job],
        evidence: Sequence[Evidence],
        round_index: int,
    ) -> ExperimentPlan:
        """Experiment Planner: choose the next highest-information actions."""

    async def audit(
        self,
        claim: Claim,
        subclaims: Sequence[Subclaim],
        loopholes: Sequence[Loophole],
        jobs: Sequence[Job],
        evidence: Sequence[Evidence],
    ) -> AuditOutcome:
        """Evidence Auditor: turn measurements into subclaim conclusions."""

    async def write_verdict(
        self,
        claim: Claim,
        subclaims: Sequence[Subclaim],
        loopholes: Sequence[Loophole],
        jobs: Sequence[Job],
        evidence: Sequence[Evidence],
        score: ReliabilityScore,
    ) -> Verdict:
        """Verdict Agent: final status plus a traceable narrative."""
