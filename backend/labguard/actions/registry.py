"""The safe action registry.

LabGuard never executes model-generated code or shell commands.  The planner
may only *name* an action from this registry and supply parameters, which are
validated against a pydantic model before anything runs.  Each entry declares
its cost, its retry limit, and the minimum autonomy level at which it may run
without a human decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from ..models.enums import ActionCategory, AutonomyMode

#: Ordering of autonomy levels, least to most permissive.
AUTONOMY_RANK: dict[AutonomyMode, int] = {
    AutonomyMode.OBSERVE_ONLY: 0,
    AutonomyMode.SAFE_REPAIR: 1,
    AutonomyMode.MANAGED_AUTONOMY: 2,
}


# --------------------------------------------------------------------------
# Parameter models
# --------------------------------------------------------------------------


class SeedComparisonParams(BaseModel):
    seeds: list[int] = Field(default_factory=lambda: [11, 1, 2, 3, 4], min_length=2, max_length=10)
    metrics: list[str] = Field(default_factory=lambda: ["accuracy", "macro_f1", "balanced_accuracy"])
    equalise_training_budget: bool = True
    #: Epoch budget applied to both arms when `equalise_training_budget`.
    equal_epochs: int = Field(default=45, ge=5, le=200)
    checkpoint_selection: str = Field(default="validation", pattern="^(validation|test|last)$")
    early_stopping_patience: int = Field(default=8, ge=1, le=50)


class RecalculateMetricsParams(BaseModel):
    seed: int = 11
    metrics: list[str] = Field(default_factory=lambda: ["accuracy", "balanced_accuracy", "macro_f1", "roc_auc"])
    bootstrap_samples: int = Field(default=200, ge=50, le=2000)


class ClasswiseParams(BaseModel):
    seed: int = 11
    checkpoint_selection: str = Field(default="validation", pattern="^(validation|test|last)$")


class DataOverlapParams(BaseModel):
    seed: int = 11
    #: Also re-run with the leakage switch on, to prove the check has teeth.
    verify_detector_with_positive_control: bool = True


class CompareConfigurationsParams(BaseModel):
    fields: list[str] = Field(
        default_factory=lambda: ["epochs", "learning_rate", "batch_size", "class_weight", "hidden_units"]
    )


class InspectCurveParams(BaseModel):
    #: Name of the configuration to replay; empty means the candidate arm.
    config_name: str = ""
    seed: int = 11
    epochs: int = Field(default=90, ge=5, le=300)


class ThresholdSweepParams(BaseModel):
    seed: int = 11
    #: Compare both arms at their own best threshold instead of a fixed 0.5.
    metric: str = Field(default="macro_f1", pattern="^(macro_f1|balanced_accuracy)$")


class DomainShiftParams(BaseModel):
    seed: int = 11
    shift_strength: float = Field(default=0.35, ge=0.0, le=2.0)


class EarlyStoppingParams(BaseModel):
    patience: int = Field(default=8, ge=1, le=50)
    monitor: str = Field(default="val_macro_f1")


class RetryTransientParams(BaseModel):
    reason: str = "transient_failure"


class ReduceBatchSizeParams(BaseModel):
    factor: float = Field(default=0.5, ge=0.1, le=0.9)
    min_batch_size: int = Field(default=8, ge=1)


class AdjustLearningRateParams(BaseModel):
    factor: float = Field(default=0.25, ge=0.05, le=0.95)
    #: Hard bounds the agent may never leave, whatever it proposes.
    min_lr: float = Field(default=0.001, ge=1e-6)
    max_lr: float = Field(default=0.5, le=10.0)


class ResumeCheckpointParams(BaseModel):
    from_epoch: int = Field(default=0, ge=0)


class ReportParams(BaseModel):
    include_curves: bool = True


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ActionSpec:
    name: str
    category: ActionCategory
    summary: str
    params_model: type[BaseModel]
    #: Abstract compute units. 1 unit is roughly one short training run.
    base_cost_units: float
    max_retries: int
    #: Lowest autonomy mode at which this may run without explicit approval.
    min_autonomy: AutonomyMode
    #: True when the action yields evidence for or against a subclaim.
    produces_evidence: bool = True
    #: Loophole kinds this action is capable of settling.
    addresses: tuple[str, ...] = ()

    def cost(self, params: dict[str, Any]) -> float:
        """Cost scales with the work actually requested."""
        if self.name == "run_seed_comparison":
            seeds = len(params.get("seeds") or []) or 3
            return round(self.base_cost_units * seeds, 2)
        return self.base_cost_units

    def validate_params(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Coerce and bound-check parameters. Raises `ActionValidationError`."""
        try:
            return self.params_model.model_validate(raw or {}).model_dump()
        except ValidationError as exc:  # surfaced to the ledger, never executed
            raise ActionValidationError(self.name, exc.errors()) from exc


class ActionValidationError(ValueError):
    def __init__(self, action: str, errors: Any) -> None:
        super().__init__(f"invalid parameters for action '{action}': {errors}")
        self.action = action
        self.errors = errors


class UnknownActionError(KeyError):
    def __init__(self, action: str) -> None:
        super().__init__(action)
        self.action = action

    def __str__(self) -> str:
        return f"'{self.action}' is not in the LabGuard action registry"


REGISTRY: dict[str, ActionSpec] = {
    spec.name: spec
    for spec in [
        ActionSpec(
            name="run_seed_comparison",
            category=ActionCategory.EXPERIMENT,
            summary="Retrain both arms across several seeds under an equal budget and compare paired deltas.",
            params_model=SeedComparisonParams,
            base_cost_units=1.6,
            max_retries=2,
            min_autonomy=AutonomyMode.MANAGED_AUTONOMY,
            addresses=("seed_sensitivity", "statistical_uncertainty", "unfair_baseline", "cherry_picked_checkpoint"),
        ),
        ActionSpec(
            name="recalculate_metrics",
            category=ActionCategory.DIAGNOSTIC,
            summary="Recompute accuracy, balanced accuracy, macro F1 and AUC with bootstrap intervals.",
            params_model=RecalculateMetricsParams,
            base_cost_units=1.0,
            max_retries=2,
            min_autonomy=AutonomyMode.SAFE_REPAIR,
            addresses=("misleading_metric", "class_imbalance", "statistical_uncertainty"),
        ),
        ActionSpec(
            name="evaluate_classwise",
            category=ActionCategory.DIAGNOSTIC,
            summary="Break the comparison down per class to locate where any advantage comes from.",
            params_model=ClasswiseParams,
            base_cost_units=0.8,
            max_retries=2,
            min_autonomy=AutonomyMode.SAFE_REPAIR,
            addresses=("class_imbalance", "misleading_metric"),
        ),
        ActionSpec(
            name="check_data_overlap",
            category=ActionCategory.DIAGNOSTIC,
            summary="Hash every row and test for train/test duplication and label leakage.",
            params_model=DataOverlapParams,
            base_cost_units=0.4,
            max_retries=2,
            min_autonomy=AutonomyMode.SAFE_REPAIR,
            addresses=("data_leakage", "train_test_overlap"),
        ),
        ActionSpec(
            name="compare_configurations",
            category=ActionCategory.DIAGNOSTIC,
            summary="Diff the two training configurations and flag unequal budgets or unfair settings.",
            params_model=CompareConfigurationsParams,
            base_cost_units=0.2,
            max_retries=1,
            min_autonomy=AutonomyMode.SAFE_REPAIR,
            addresses=("unfair_baseline", "confounding_variable"),
        ),
        ActionSpec(
            name="inspect_training_curve",
            category=ActionCategory.DIAGNOSTIC,
            summary="Replay a run's loss curves and classify its training health.",
            params_model=InspectCurveParams,
            base_cost_units=1.0,
            max_retries=2,
            min_autonomy=AutonomyMode.SAFE_REPAIR,
            addresses=("cherry_picked_checkpoint", "irreproducible"),
        ),
        ActionSpec(
            name="sweep_decision_threshold",
            category=ActionCategory.DIAGNOSTIC,
            summary="Give both arms their own best decision threshold before comparing them.",
            params_model=ThresholdSweepParams,
            base_cost_units=0.9,
            max_retries=2,
            min_autonomy=AutonomyMode.MANAGED_AUTONOMY,
            addresses=("improper_threshold", "misleading_metric"),
        ),
        ActionSpec(
            name="test_domain_shift",
            category=ActionCategory.EXPERIMENT,
            summary="Re-evaluate both arms under a covariate shift on the test split.",
            params_model=DomainShiftParams,
            base_cost_units=1.2,
            max_retries=2,
            min_autonomy=AutonomyMode.MANAGED_AUTONOMY,
            addresses=("domain_shift",),
        ),
        ActionSpec(
            name="apply_early_stopping",
            category=ActionCategory.RECOVERY,
            summary="Stop a diverging run and keep the last healthy checkpoint.",
            params_model=EarlyStoppingParams,
            base_cost_units=0.1,
            max_retries=1,
            min_autonomy=AutonomyMode.SAFE_REPAIR,
            produces_evidence=False,
        ),
        ActionSpec(
            name="retry_transient_failure",
            category=ActionCategory.RECOVERY,
            summary="Retry a run that failed for an infrastructure reason, not a scientific one.",
            params_model=RetryTransientParams,
            base_cost_units=0.1,
            max_retries=2,
            min_autonomy=AutonomyMode.SAFE_REPAIR,
            produces_evidence=False,
        ),
        ActionSpec(
            name="reduce_batch_size",
            category=ActionCategory.RECOVERY,
            summary="Halve the batch size after an out-of-memory failure and resume.",
            params_model=ReduceBatchSizeParams,
            base_cost_units=0.2,
            max_retries=2,
            min_autonomy=AutonomyMode.MANAGED_AUTONOMY,
            produces_evidence=False,
        ),
        ActionSpec(
            name="adjust_learning_rate_within_bounds",
            category=ActionCategory.RECOVERY,
            summary="Scale the learning rate down inside hard bounds after numerical divergence.",
            params_model=AdjustLearningRateParams,
            base_cost_units=0.2,
            max_retries=2,
            min_autonomy=AutonomyMode.MANAGED_AUTONOMY,
            produces_evidence=False,
        ),
        ActionSpec(
            name="resume_from_checkpoint",
            category=ActionCategory.RECOVERY,
            summary="Resume an interrupted run from its last verified checkpoint.",
            params_model=ResumeCheckpointParams,
            base_cost_units=0.2,
            max_retries=2,
            min_autonomy=AutonomyMode.SAFE_REPAIR,
            produces_evidence=False,
        ),
        ActionSpec(
            name="generate_reliability_report",
            category=ActionCategory.REPORT,
            summary="Render the evidence ledger and reliability scores into a downloadable report.",
            params_model=ReportParams,
            base_cost_units=0.1,
            max_retries=1,
            min_autonomy=AutonomyMode.SAFE_REPAIR,
            produces_evidence=False,
        ),
    ]
}

#: Actions the RunMedic may invoke as repairs, keyed by anomaly.
RECOVERY_FOR_ANOMALY: dict[str, str] = {
    "overfitting": "apply_early_stopping",
    "nan_loss": "adjust_learning_rate_within_bounds",
    "exploding_loss": "adjust_learning_rate_within_bounds",
    "resource_exhausted": "reduce_batch_size",
    "missing_artifact": "retry_transient_failure",
    "stalled_training": "retry_transient_failure",
    "corrupted_checkpoint": "resume_from_checkpoint",
}


def get_action(name: str) -> ActionSpec:
    spec = REGISTRY.get(name)
    if spec is None:
        raise UnknownActionError(name)
    return spec


def action_names() -> list[str]:
    return sorted(REGISTRY)


def requires_approval(
    spec: ActionSpec, params: dict[str, Any], mode: AutonomyMode, remaining_units: float, threshold_units: float
) -> tuple[bool, str]:
    """Decide whether a human must sign off before this action runs.

    Returns `(required, reason)`; the reason is recorded in the ledger so the
    approval screen can explain itself.
    """
    cost = spec.cost(params)
    if mode == AutonomyMode.OBSERVE_ONLY:
        return True, "autonomy mode is observe-only: every action is recommended, never executed automatically"
    if AUTONOMY_RANK[spec.min_autonomy] > AUTONOMY_RANK[mode]:
        return True, (f"'{spec.name}' needs {spec.min_autonomy.value} autonomy, the claim is running at {mode.value}")
    if cost > remaining_units:
        return True, f"estimated cost {cost} units exceeds the remaining budget of {remaining_units} units"
    if spec.category == ActionCategory.EXPERIMENT and cost > threshold_units:
        return True, f"experiment cost {cost} units is above the {threshold_units}-unit approval threshold"
    return False, "within autonomy policy and budget"
