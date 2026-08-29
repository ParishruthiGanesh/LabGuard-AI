"""Deterministic agent runtime.

Used in demo mode, and as the fallback whenever Gemini is unreachable.  It
produces exactly the same object graph as the Gemini runtime, so the dashboard
and the orchestrator behave identically; only the wording differs.
"""

from __future__ import annotations

from collections.abc import Sequence

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
from . import deterministic as rules
from .base import AuditOutcome


class DeterministicRuntime:
    """Rule-engine implementation of the seven agents."""

    name = "deterministic"

    async def analyse_claim(self, claim: Claim) -> list[Subclaim]:
        return rules.build_subclaims(claim)

    async def find_loopholes(
        self, claim: Claim, subclaims: Sequence[Subclaim]
    ) -> tuple[list[Loophole], list[AlternativeExplanation]]:
        return rules.detect_loopholes(claim, subclaims)

    async def plan_round(
        self,
        claim: Claim,
        subclaims: Sequence[Subclaim],
        loopholes: Sequence[Loophole],
        jobs: Sequence[Job],
        evidence: Sequence[Evidence],
        round_index: int,
    ) -> ExperimentPlan:
        return rules.plan_next_round(claim, subclaims, loopholes, jobs, round_index)

    async def audit(
        self,
        claim: Claim,
        subclaims: Sequence[Subclaim],
        loopholes: Sequence[Loophole],
        jobs: Sequence[Job],
        evidence: Sequence[Evidence],
    ) -> AuditOutcome:
        return rules.audit_evidence(claim, subclaims, loopholes, jobs, evidence)

    async def write_verdict(
        self,
        claim: Claim,
        subclaims: Sequence[Subclaim],
        loopholes: Sequence[Loophole],
        jobs: Sequence[Job],
        evidence: Sequence[Evidence],
        score: ReliabilityScore,
    ) -> Verdict:
        return rules.build_verdict(claim, subclaims, loopholes, jobs, evidence, score)
