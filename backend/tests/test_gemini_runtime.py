"""The Gemini/ADK runtime's guard rails.

The model itself is not called here — `_invoke` is replaced with canned
responses. What is under test is the layer that decides how much of a model
response is allowed to survive, which is the part that must hold even when the
model is wrong, stale or adversarial.
"""

from __future__ import annotations

from typing import Any

import pytest

from labguard.agents import deterministic as rules
from labguard.agents.gemini_runtime import (
    AdkAgentRuntime,
    LoopholeDraft,
    LoopholesResponse,
    PlanChoice,
    PlanResponse,
    SubclaimDraft,
    SubclaimsResponse,
    VerdictResponse,
    resolve_model,
)
from labguard.config import Settings
from labguard.experiments.scenario import demo_claim
from labguard.models.domain import ReliabilityScore
from labguard.models.enums import LoopholeKind, VerdictStatus


def build_runtime(responses: dict[str, Any]) -> AdkAgentRuntime:
    """A runtime whose ADK calls return canned structured responses."""
    runtime = AdkAgentRuntime.__new__(AdkAgentRuntime)
    runtime.settings = Settings(LABGUARD_MODE="cloud", GOOGLE_API_KEY="test-key")
    runtime.model = "gemini-3.5-flash"
    runtime.name = "gemini-adk:test"
    runtime._agents = {}
    runtime._runners = {}
    runtime.overrides = []

    async def fake_invoke(name, instruction, schema, payload):
        return responses.get(name)

    runtime._invoke = fake_invoke  # type: ignore[method-assign]
    return runtime


class TestModelResolution:
    def test_falls_back_to_the_first_candidate_when_listing_fails(self, monkeypatch):
        settings = Settings(GEMINI_MODEL="gemini-3.5-flash", GEMINI_MODEL_FALLBACKS="gemini-flash-latest")
        # No credentials and no network: resolution must still return something
        # usable rather than raising.
        assert resolve_model(settings) == "gemini-3.5-flash"

    def test_candidate_list_is_ordered_and_deduplicated(self):
        settings = Settings(GEMINI_MODEL="a", GEMINI_MODEL_FALLBACKS="b, a ,c")
        assert settings.model_candidates == ["a", "b", "c"]


class TestClaimAnalyst:
    async def test_model_wording_is_adopted_for_known_keys(self):
        claim = demo_claim()
        runtime = build_runtime(
            {
                "claim_analyst": SubclaimsResponse(
                    subclaims=[
                        SubclaimDraft(
                            key="seed_stability",
                            statement="The gap survives five independent seeds.",
                            measurable_quantity="mean per-seed accuracy delta",
                            rationale="One seed cannot separate signal from variance.",
                        )
                    ]
                )
            }
        )
        subclaims = await runtime.analyse_claim(claim)
        seed = next(s for s in subclaims if rules.subclaim_key(s) == "seed_stability")
        assert seed.statement == "The gap survives five independent seeds."
        # Every other subclaim is still present with its rule-engine wording.
        assert len(subclaims) == len(rules.build_subclaims(claim))

    async def test_an_invented_key_is_dropped_and_recorded(self):
        claim = demo_claim()
        runtime = build_runtime(
            {
                "claim_analyst": SubclaimsResponse(
                    subclaims=[
                        SubclaimDraft(key="vibes", statement="It feels better.", measurable_quantity="?", rationale="")
                    ]
                )
            }
        )
        subclaims = await runtime.analyse_claim(claim)
        assert all(rules.subclaim_key(s) != "vibes" for s in subclaims)
        assert any("vibes" in o["reason"] for o in runtime.overrides)

    async def test_an_unreachable_model_falls_back_to_the_rule_engine(self):
        claim = demo_claim()
        runtime = build_runtime({})  # every _invoke returns None
        subclaims = await runtime.analyse_claim(claim)
        assert [s.id for s in subclaims] == [s.id for s in rules.build_subclaims(claim)]


class TestScientificSkeptic:
    async def test_measured_severity_survives_a_model_that_disagrees(self):
        claim = demo_claim()
        runtime = build_runtime(
            {
                "scientific_skeptic": LoopholesResponse(
                    loopholes=[
                        LoopholeDraft(
                            kind="seed_sensitivity",
                            title="Single seed",
                            rationale="Rephrased by the model.",
                            severity=0.01,
                        )
                    ],
                    alternative_explanations=["The gap is noise."],
                )
            }
        )
        loopholes, alternatives = await runtime.find_loopholes(claim, rules.build_subclaims(claim))
        seed = next(h for h in loopholes if h.kind == LoopholeKind.SEED_SENSITIVITY)

        assert seed.rationale == "Rephrased by the model."  # wording is the model's
        assert seed.severity == 0.85  # severity is not
        assert seed.detected_by == "heuristic+gemini"
        assert any(a.statement == "The gap is noise." for a in alternatives)

    async def test_a_model_proposed_loophole_cannot_block_a_verdict_on_its_own(self):
        claim = demo_claim()
        runtime = build_runtime(
            {
                "scientific_skeptic": LoopholesResponse(
                    loopholes=[LoopholeDraft(kind="domain_shift", title="Shift", rationale="Maybe.", severity=1.0)],
                    alternative_explanations=[],
                )
            }
        )
        loopholes, _ = await runtime.find_loopholes(claim, rules.build_subclaims(claim))
        added = next(h for h in loopholes if h.kind == LoopholeKind.DOMAIN_SHIFT)
        # 0.6 is the threshold at which an open loophole blocks the verdict.
        assert added.severity <= 0.6
        assert added.detected_by == "gemini"

    async def test_an_invented_loophole_kind_is_dropped(self):
        claim = demo_claim()
        runtime = build_runtime(
            {
                "scientific_skeptic": LoopholesResponse(
                    loopholes=[LoopholeDraft(kind="bad_vibes", title="x", rationale="y", severity=0.9)],
                    alternative_explanations=[],
                )
            }
        )
        loopholes, _ = await runtime.find_loopholes(claim, rules.build_subclaims(claim))
        assert all(h.kind.value != "bad_vibes" for h in loopholes)
        assert any("bad_vibes" in o["reason"] for o in runtime.overrides)


class TestExperimentPlanner:
    async def test_the_model_may_reorder_but_not_invent_actions(self):
        claim = demo_claim()
        subclaims = rules.build_subclaims(claim)
        loopholes, _ = rules.detect_loopholes(claim, subclaims)
        runtime = build_runtime(
            {
                "experiment_planner": PlanResponse(
                    chosen=[
                        PlanChoice(
                            action_type="check_data_overlap", reason="cheapest first", expected_information_gain=0.5
                        ),
                        PlanChoice(action_type="rm -rf /", reason="malicious", expected_information_gain=1.0),
                        PlanChoice(
                            action_type="compare_configurations", reason="then this", expected_information_gain=0.6
                        ),
                    ],
                    summary="Two checks.",
                )
            }
        )
        plan = await runtime.plan_round(claim, subclaims, loopholes, [], [], 0)

        assert [i.action_type for i in plan.items] == ["check_data_overlap", "compare_configurations"]
        assert any("rm -rf /" in o["reason"] for o in runtime.overrides)
        # Costs stay the registry's, not the model's.
        assert plan.total_cost_units == pytest.approx(0.6)

    async def test_costs_and_approval_are_never_taken_from_the_model(self):
        claim = demo_claim()
        subclaims = rules.build_subclaims(claim)
        loopholes, _ = rules.detect_loopholes(claim, subclaims)
        runtime = build_runtime(
            {
                "experiment_planner": PlanResponse(
                    chosen=[
                        PlanChoice(
                            action_type="run_seed_comparison",
                            reason="cheap, honest, definitely fine",
                            expected_information_gain=0.99,
                        )
                    ],
                    summary="Just the big one.",
                )
            }
        )
        plan = await runtime.plan_round(claim, subclaims, loopholes, [], [], 0)
        item = plan.items[0]
        assert item.estimated_cost_units == 8.0
        assert item.requires_approval is True


class TestVerdictAgent:
    async def test_a_disagreeing_model_status_is_overruled_and_logged(self):
        claim = demo_claim()
        subclaims = rules.build_subclaims(claim)
        loopholes, _ = rules.detect_loopholes(claim, subclaims)
        score = ReliabilityScore(claim_id=claim.id)
        runtime = build_runtime(
            {
                "verdict_agent": VerdictResponse(
                    status="supported",
                    headline="Model B is clearly better.",
                    narrative="It just is.",
                    evidence_summary=["Trust me."],
                    remaining_uncertainty=[],
                )
            }
        )
        verdict = await runtime.write_verdict(claim, subclaims, loopholes, [], [], score)

        # With no seed comparison run, the measured status cannot be "supported".
        assert verdict.rule_based_status == VerdictStatus.INCONCLUSIVE
        assert verdict.status == VerdictStatus.INCONCLUSIVE
        assert verdict.headline == "Model B is clearly better."  # prose is the model's
        assert any("supported" in o["reason"] for o in runtime.overrides)

    async def test_an_invalid_status_does_not_crash_the_verdict(self):
        claim = demo_claim()
        subclaims = rules.build_subclaims(claim)
        score = ReliabilityScore(claim_id=claim.id)
        runtime = build_runtime(
            {
                "verdict_agent": VerdictResponse(
                    status="extremely_true",
                    headline="h",
                    narrative="n",
                    evidence_summary=[],
                    remaining_uncertainty=[],
                )
            }
        )
        verdict = await runtime.write_verdict(claim, subclaims, [], [], [], score)
        assert verdict.status == verdict.rule_based_status


class TestAgentConstruction:
    def test_adk_agents_are_built_with_a_schema_and_no_tools(self):
        """The model must have no way to act, only to propose."""
        runtime = build_runtime({})
        agent = runtime._agent("claim_analyst", "You are the Claim Analyst.\nDo the thing.", SubclaimsResponse)

        assert agent.model == "gemini-3.5-flash"
        assert agent.output_schema is SubclaimsResponse
        assert not agent.tools, "an agent with tools could act on its own output"
        assert agent.disallow_transfer_to_parent and agent.disallow_transfer_to_peers
        # Repeated calls reuse the same agent rather than rebuilding it.
        assert runtime._agent("claim_analyst", "x", SubclaimsResponse) is agent
