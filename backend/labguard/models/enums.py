"""Enumerations shared by the agents, the action registry and the API."""

from __future__ import annotations

from enum import Enum


class StrEnum(str, Enum):
    """`str`-valued enum so members serialise directly to JSON."""

    def __str__(self) -> str:  # pragma: no cover - trivial
        return str(self.value)


class AutonomyMode(StrEnum):
    """How much the platform may do without a human in the loop."""

    OBSERVE_ONLY = "observe_only"
    SAFE_REPAIR = "safe_repair"
    MANAGED_AUTONOMY = "managed_autonomy"


class ClaimState(StrEnum):
    """Orchestrator state machine for a single claim."""

    CREATED = "created"
    ANALYZING = "analyzing"
    SKEPTIC_REVIEW = "skeptic_review"
    PLANNING = "planning"
    AWAITING_APPROVAL = "awaiting_approval"
    EXECUTING = "executing"
    AUDITING = "auditing"
    VERDICT = "verdict"
    HALTED_BUDGET = "halted_budget"
    HALTED_LOOP = "halted_loop"
    HALTED_APPROVAL = "halted_approval"

    @property
    def is_terminal(self) -> bool:
        return self in {
            ClaimState.VERDICT,
            ClaimState.HALTED_BUDGET,
            ClaimState.HALTED_LOOP,
            ClaimState.HALTED_APPROVAL,
        }


class JobState(StrEnum):
    """Lifecycle of one queued experiment action."""

    PLANNED = "planned"
    AWAITING_APPROVAL = "awaiting_approval"
    QUEUED = "queued"
    RUNNING = "running"
    RECOVERING = "recovering"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED_LOOP = "blocked_loop"
    REJECTED = "rejected"

    @property
    def is_terminal(self) -> bool:
        return self in {
            JobState.COMPLETED,
            JobState.FAILED,
            JobState.BLOCKED_LOOP,
            JobState.REJECTED,
        }


class LoopholeKind(StrEnum):
    """The closed vocabulary of scientific weaknesses the Skeptic may raise.

    Gemini selects from this list; it cannot invent new kinds, which keeps the
    downstream planner able to map every loophole onto a registry action.
    """

    SEED_SENSITIVITY = "seed_sensitivity"
    DATA_LEAKAGE = "data_leakage"
    TRAIN_TEST_OVERLAP = "train_test_overlap"
    CLASS_IMBALANCE = "class_imbalance"
    UNFAIR_BASELINE = "unfair_baseline"
    CHERRY_PICKED_CHECKPOINT = "cherry_picked_checkpoint"
    IMPROPER_THRESHOLD = "improper_threshold"
    MISLEADING_METRIC = "misleading_metric"
    MISSING_ABLATION = "missing_ablation"
    DOMAIN_SHIFT = "domain_shift"
    STATISTICAL_UNCERTAINTY = "statistical_uncertainty"
    INSUFFICIENT_SAMPLE_SIZE = "insufficient_sample_size"
    CONFOUNDING_VARIABLE = "confounding_variable"
    IRREPRODUCIBLE = "irreproducible"


class LoopholeStatus(StrEnum):
    OPEN = "open"
    INVESTIGATING = "investigating"
    CONFIRMED = "confirmed"
    REFUTED = "refuted"
    UNRESOLVED = "unresolved"


class SubclaimStatus(StrEnum):
    UNTESTED = "untested"
    TESTING = "testing"
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    INCONCLUSIVE = "inconclusive"


class EvidenceStance(StrEnum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    NEUTRAL = "neutral"


class VerdictStatus(StrEnum):
    SUPPORTED = "supported"
    PROVISIONALLY_SUPPORTED = "provisionally_supported"
    FRAGILE = "fragile"
    INCONCLUSIVE = "inconclusive"
    NOT_SUFFICIENTLY_SUPPORTED = "not_sufficiently_supported"
    REFUTED = "refuted"


class AgentName(StrEnum):
    CLAIM_ANALYST = "claim_analyst"
    SCIENTIFIC_SKEPTIC = "scientific_skeptic"
    EXPERIMENT_PLANNER = "experiment_planner"
    RUN_MANAGER = "run_manager"
    RUN_MEDIC = "run_medic"
    EVIDENCE_AUDITOR = "evidence_auditor"
    VERDICT_AGENT = "verdict_agent"
    ORCHESTRATOR = "orchestrator"


class ActionCategory(StrEnum):
    DIAGNOSTIC = "diagnostic"
    EXPERIMENT = "experiment"
    RECOVERY = "recovery"
    REPORT = "report"


class HealthStatus(StrEnum):
    """RunMedic's assessment of a live run."""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    RECOVERED = "recovered"
    UNKNOWN = "unknown"


class AnomalyKind(StrEnum):
    """Operational failure modes RunMedic watches for."""

    OVERFITTING = "overfitting"
    UNDERFITTING = "underfitting"
    NAN_LOSS = "nan_loss"
    EXPLODING_LOSS = "exploding_loss"
    STALLED_TRAINING = "stalled_training"
    CORRUPTED_CHECKPOINT = "corrupted_checkpoint"
    MISSING_ARTIFACT = "missing_artifact"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    SLOW_EPOCH = "slow_epoch"
    EXCESSIVE_RETRIES = "excessive_retries"
    RECOVERY_LOOP = "recovery_loop"
    DUPLICATE_EXPERIMENT = "duplicate_experiment"


class ScoreDimension(StrEnum):
    REPRODUCIBILITY = "reproducibility"
    DATA_INTEGRITY = "data_integrity"
    BASELINE_FAIRNESS = "baseline_fairness"
    STATISTICAL_STABILITY = "statistical_stability"
    TRAINING_HEALTH = "training_health"
    EVIDENCE_COMPLETENESS = "evidence_completeness"
    OVERALL_CLAIM_CONFIDENCE = "overall_claim_confidence"
