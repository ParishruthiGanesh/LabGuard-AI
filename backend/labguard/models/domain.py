"""Persisted domain objects.

Every model here maps 1:1 onto a Firestore document (or an entry in the
in-memory store used by demo mode), so the two backends are interchangeable.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .enums import (
    ActionCategory,
    AgentName,
    AnomalyKind,
    AutonomyMode,
    ClaimState,
    EvidenceStance,
    HealthStatus,
    JobState,
    LoopholeKind,
    LoopholeStatus,
    ScoreDimension,
    SubclaimStatus,
    VerdictStatus,
)


def utcnow() -> datetime:
    return datetime.now(UTC)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class Base(BaseModel):
    model_config = ConfigDict(use_enum_values=False, populate_by_name=True)


# --------------------------------------------------------------------------
# Claim submission
# --------------------------------------------------------------------------


class ModelConfig(Base):
    """One system under comparison, as described by the researcher."""

    name: str
    family: str = "linear"
    epochs: int = 30
    learning_rate: float = 0.05
    hidden_units: int = 0
    batch_size: int = 64
    #: "none", "balanced" or "sqrt_balanced" — inverse-frequency loss weighting.
    class_weight: str = "none"
    #: "bce" (bounded gradients) or "mse_logit" (squared error on the raw
    #: logit). `mse_logit` genuinely diverges above a critical learning rate,
    #: which is how the demo produces a real NaN failure rather than a fake one.
    objective: str = "bce"
    is_baseline: bool = False
    #: "primary" configs take part in the headline comparison; "variant"
    #: configs are extra setups the researcher also ran.
    role: str = "primary"
    notes: str = ""


class DatasetInfo(Base):
    name: str = "synthetic_violence_clips"
    n_samples: int = 4000
    n_features: int = 24
    positive_rate: float = 0.08
    test_fraction: float = 0.25
    #: Deliberate flaw switch used by the bundled demo scenario.
    inject_train_test_overlap: int = 0
    domain_shift_strength: float = 0.0


class ExistingResult(Base):
    """The result the researcher already has, and wants validated."""

    model_name: str
    metric: str = "accuracy"
    value: float = 0.0
    seed: int = 0
    checkpoint_selected_on: str = "test"
    epochs_trained: int = 0
    checkpoint_uri: str = ""


class BudgetPolicy(Base):
    """Compute budget expressed in abstract units (1 unit ~= one short run)."""

    total_units: float = 40.0
    consumed_units: float = 0.0
    #: Any experiment estimated above this needs a human decision first.
    approval_threshold_units: float = 6.0

    @property
    def remaining_units(self) -> float:
        return max(0.0, self.total_units - self.consumed_units)


class ClaimContext(Base):
    dataset: DatasetInfo = Field(default_factory=DatasetInfo)
    models: list[ModelConfig] = Field(default_factory=list)
    existing_results: list[ExistingResult] = Field(default_factory=list)
    #: Deliberate-fault switch for the bundled demo: when true the reported
    #: checkpoint fails its integrity check every time it is opened, which is
    #: what the recovery-loop detector is meant to catch.
    reported_checkpoint_corrupt: bool = False
    notes: str = ""


class Claim(Base):
    id: str = Field(default_factory=lambda: new_id("claim"))
    text: str
    context: ClaimContext = Field(default_factory=ClaimContext)
    autonomy_mode: AutonomyMode = AutonomyMode.SAFE_REPAIR
    budget: BudgetPolicy = Field(default_factory=BudgetPolicy)
    state: ClaimState = ClaimState.CREATED
    active_agent: AgentName | None = None
    latest_action: str = ""
    demo_mode: bool = True
    reasoning_backend: str = "deterministic"
    planning_round: int = 0
    halt_reason: str = ""
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


# --------------------------------------------------------------------------
# Scientific decomposition
# --------------------------------------------------------------------------


class Subclaim(Base):
    id: str = Field(default_factory=lambda: new_id("sub"))
    claim_id: str
    statement: str
    measurable_quantity: str
    rationale: str = ""
    status: SubclaimStatus = SubclaimStatus.UNTESTED
    #: 0..1, derived from evidence counts — never invented by the model.
    confidence: float = 0.0
    evidence_ids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)


class Loophole(Base):
    id: str = Field(default_factory=lambda: new_id("hole"))
    claim_id: str
    kind: LoopholeKind
    title: str
    rationale: str
    #: 0..1 — how damaging this would be to the claim if real.
    severity: float = 0.5
    status: LoopholeStatus = LoopholeStatus.OPEN
    detected_by: str = "heuristic"
    subclaim_ids: list[str] = Field(default_factory=list)
    resolution: str = ""
    created_at: datetime = Field(default_factory=utcnow)


class AlternativeExplanation(Base):
    id: str = Field(default_factory=lambda: new_id("alt"))
    claim_id: str
    statement: str
    tested_by_action: str = ""
    status: str = "open"


# --------------------------------------------------------------------------
# Planning and execution
# --------------------------------------------------------------------------


class PlanItem(Base):
    id: str = Field(default_factory=lambda: new_id("item"))
    action_type: str
    params: dict[str, Any] = Field(default_factory=dict)
    reason: str
    targets_loophole_ids: list[str] = Field(default_factory=list)
    targets_subclaim_ids: list[str] = Field(default_factory=list)
    estimated_cost_units: float = 1.0
    #: 0..1 expected reduction in uncertainty, computed by the planner scorer.
    expected_information_gain: float = 0.5
    requires_approval: bool = False
    category: ActionCategory = ActionCategory.DIAGNOSTIC
    job_id: str | None = None


class ExperimentPlan(Base):
    id: str = Field(default_factory=lambda: new_id("plan"))
    claim_id: str
    round_index: int = 0
    items: list[PlanItem] = Field(default_factory=list)
    summary: str = ""
    #: draft | awaiting_approval | approved | rejected | executed
    status: str = "draft"
    total_cost_units: float = 0.0
    requires_approval: bool = False
    approved_by: str = ""
    decided_at: datetime | None = None
    created_at: datetime = Field(default_factory=utcnow)


class EpochRecord(Base):
    epoch: int
    train_loss: float
    val_loss: float
    train_metric: float
    val_metric: float
    seconds: float = 0.0
    gpu_util_pct: float = 0.0
    memory_mb: float = 0.0


class HealthEvent(Base):
    id: str = Field(default_factory=lambda: new_id("hev"))
    job_id: str
    anomaly: AnomalyKind
    status: HealthStatus
    detail: str
    epoch: int | None = None
    action_taken: str = ""
    #: True only when a repair actually changed the run, so the UI never shows
    #: "recovered" for something that was merely observed.
    repaired: bool = False
    requires_approval: bool = False
    at: datetime = Field(default_factory=utcnow)


class RunHealth(Base):
    status: HealthStatus = HealthStatus.UNKNOWN
    summary: str = ""
    events: list[HealthEvent] = Field(default_factory=list)
    peak_memory_mb: float = 0.0
    mean_gpu_util_pct: float = 0.0


class Job(Base):
    id: str = Field(default_factory=lambda: new_id("job"))
    claim_id: str
    plan_id: str = ""
    plan_item_id: str = ""
    action_type: str
    params: dict[str, Any] = Field(default_factory=dict)
    state: JobState = JobState.PLANNED
    category: ActionCategory = ActionCategory.DIAGNOSTIC
    reason: str = ""
    attempts: int = 0
    max_retries: int = 2
    recovery_actions: list[str] = Field(default_factory=list)
    estimated_cost_units: float = 1.0
    actual_cost_units: float = 0.0
    error: str = ""
    #: A stable hash of (action_type, params) used for duplicate detection.
    fingerprint: str = ""
    #: Signature of the last failure, used to spot non-progressing recovery.
    failure_signature: str = ""
    curves: list[EpochRecord] = Field(default_factory=list)
    health: RunHealth = Field(default_factory=RunHealth)
    result: dict[str, Any] = Field(default_factory=dict)
    artifact_uris: list[str] = Field(default_factory=list)
    queued_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


# --------------------------------------------------------------------------
# Evidence, ledger, scoring, verdict
# --------------------------------------------------------------------------


class Evidence(Base):
    id: str = Field(default_factory=lambda: new_id("ev"))
    claim_id: str
    job_id: str = ""
    subclaim_ids: list[str] = Field(default_factory=list)
    loophole_ids: list[str] = Field(default_factory=list)
    stance: EvidenceStance = EvidenceStance.NEUTRAL
    statement: str = ""
    #: Machine-checkable numbers behind `statement`; never model-generated.
    measurements: dict[str, Any] = Field(default_factory=dict)
    #: 0..1 weight of this evidence, from sample size / seed count.
    strength: float = 0.5
    artifact_uris: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)


class LedgerEntry(Base):
    """Append-only audit trail row. Nothing in the system rewrites these."""

    id: str = Field(default_factory=lambda: new_id("led"))
    claim_id: str
    sequence: int = 0
    agent: AgentName
    action: str
    reason: str = ""
    input_summary: dict[str, Any] = Field(default_factory=dict)
    result_summary: dict[str, Any] = Field(default_factory=dict)
    decision: str = ""
    job_id: str | None = None
    artifact_uris: list[str] = Field(default_factory=list)
    at: datetime = Field(default_factory=utcnow)


class ScoreCheck(Base):
    """One named, deterministic check that feeds a reliability dimension."""

    id: str
    label: str
    passed: bool | None
    weight: float = 1.0
    detail: str = ""
    observed: dict[str, Any] = Field(default_factory=dict)


class DimensionScore(Base):
    dimension: ScoreDimension
    score: int
    checks: list[ScoreCheck] = Field(default_factory=list)
    #: Human-readable arithmetic, e.g. "2.0 of 3.0 weighted checks passed".
    calculation: str = ""


class ReliabilityScore(Base):
    claim_id: str
    dimensions: list[DimensionScore] = Field(default_factory=list)
    overall: int = 0
    calculation: str = ""
    computed_at: datetime = Field(default_factory=utcnow)


class Verdict(Base):
    id: str = Field(default_factory=lambda: new_id("verdict"))
    claim_id: str
    status: VerdictStatus = VerdictStatus.INCONCLUSIVE
    headline: str = ""
    narrative: str = ""
    evidence_summary: list[str] = Field(default_factory=list)
    remaining_uncertainty: list[str] = Field(default_factory=list)
    run_health_incidents: list[str] = Field(default_factory=list)
    reproducibility: dict[str, Any] = Field(default_factory=dict)
    score: ReliabilityScore | None = None
    #: Deterministic status before narrative generation, kept for audit.
    rule_based_status: VerdictStatus = VerdictStatus.INCONCLUSIVE
    generated_by: str = "deterministic"
    created_at: datetime = Field(default_factory=utcnow)
