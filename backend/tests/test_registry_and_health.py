"""The action registry is the safety boundary; RunMedic is the detector."""

from __future__ import annotations

import pytest

from labguard.actions.registry import (
    ActionValidationError,
    UnknownActionError,
    get_action,
    requires_approval,
)
from labguard.agents.health import (
    analyse_curve,
    classify_failure,
    should_stop_now,
    summarise,
)
from labguard.models.domain import EpochRecord, Job
from labguard.models.enums import AnomalyKind, AutonomyMode, HealthStatus


def _curve(values, train=None):
    train = train or [0.5 - 0.004 * i for i in range(len(values))]
    return [
        EpochRecord(epoch=i + 1, train_loss=train[i], val_loss=v, train_metric=0.7, val_metric=0.6, seconds=0.02)
        for i, v in enumerate(values)
    ]


class TestRegistry:
    def test_unknown_action_is_rejected(self):
        with pytest.raises(UnknownActionError):
            get_action("rm_minus_rf")

    def test_parameters_are_validated_and_bounded(self):
        spec = get_action("adjust_learning_rate_within_bounds")
        with pytest.raises(ActionValidationError):
            spec.validate_params({"factor": 9.0})  # outside the declared bounds
        assert spec.validate_params({"factor": 0.25})["factor"] == 0.25

    def test_seed_comparison_cost_scales_with_seed_count(self):
        spec = get_action("run_seed_comparison")
        assert spec.cost({"seeds": [1, 2, 3]}) < spec.cost({"seeds": [1, 2, 3, 4, 5]})

    def test_observe_only_never_executes_anything(self):
        spec = get_action("check_data_overlap")
        needed, reason = requires_approval(spec, {}, AutonomyMode.OBSERVE_ONLY, 100, 6)
        assert needed and "observe-only" in reason

    def test_action_above_the_modes_autonomy_needs_approval(self):
        spec = get_action("reduce_batch_size")  # requires managed autonomy
        needed, _ = requires_approval(spec, {}, AutonomyMode.SAFE_REPAIR, 100, 6)
        assert needed
        allowed, _ = requires_approval(spec, {}, AutonomyMode.MANAGED_AUTONOMY, 100, 6)
        assert not allowed

    def test_expensive_experiment_needs_approval(self):
        spec = get_action("run_seed_comparison")
        params = spec.validate_params({"seeds": [1, 2, 3, 4, 5]})
        needed, reason = requires_approval(spec, params, AutonomyMode.MANAGED_AUTONOMY, 100, 6)
        assert needed and "approval threshold" in reason

    def test_action_over_budget_needs_approval(self):
        spec = get_action("recalculate_metrics")
        needed, reason = requires_approval(spec, {}, AutonomyMode.MANAGED_AUTONOMY, 0.1, 6)
        assert needed and "remaining budget" in reason


class TestRunMedic:
    def test_overfitting_is_detected(self):
        curve = _curve([0.5, 0.4, 0.3, 0.25, 0.24, 0.26, 0.28, 0.30, 0.33, 0.36, 0.39, 0.42])
        findings = analyse_curve(curve)
        assert any(f.anomaly == AnomalyKind.OVERFITTING for f in findings)

    def test_a_healthy_curve_reports_nothing(self):
        curve = _curve([0.5, 0.44, 0.39, 0.35, 0.32, 0.30, 0.29, 0.285, 0.283, 0.282])
        assert analyse_curve(curve) == []
        assert summarise([])[0] == HealthStatus.HEALTHY

    def test_nan_loss_short_circuits_every_other_finding(self):
        curve = _curve([0.5, 0.4, float("nan")])
        findings = analyse_curve(curve)
        assert len(findings) == 1 and findings[0].anomaly == AnomalyKind.NAN_LOSS

    def test_a_flat_run_that_never_learned_is_stalled(self):
        flat = [0.69] * 10
        findings = analyse_curve(_curve(flat, train=flat))
        assert any(f.anomaly == AnomalyKind.STALLED_TRAINING for f in findings)

    def test_a_converged_plateau_is_not_a_stall(self):
        values = [0.6, 0.4, 0.3, 0.25, 0.22, 0.21, 0.208, 0.2079, 0.2079, 0.2079]
        findings = analyse_curve(_curve(values, train=values))
        assert not any(f.anomaly == AnomalyKind.STALLED_TRAINING for f in findings)

    def test_a_run_is_only_stopped_once_the_metric_also_turns(self):
        rising_loss = [0.3, 0.28, 0.27, 0.29, 0.31, 0.33, 0.35, 0.37, 0.39, 0.41, 0.43, 0.45]
        curve = _curve(rising_loss)
        for i, record in enumerate(curve):
            record.val_metric = 0.5 + 0.01 * i  # metric still improving
        findings = analyse_curve(curve)
        assert not should_stop_now(curve, findings)

        for i, record in enumerate(curve):  # metric has now turned over
            record.val_metric = 0.9 if i == 1 else 0.5
        assert should_stop_now(curve, analyse_curve(curve))

    def test_an_unchanging_failure_becomes_a_recovery_loop(self):
        job = Job(claim_id="c", action_type="resume_from_checkpoint")
        signature = "corrupted_checkpoint:crc failure"
        assert classify_failure(job, signature).anomaly == AnomalyKind.CORRUPTED_CHECKPOINT

        job.recovery_actions = [f"failure:{signature}", f"failure:{signature}"]
        job.attempts = 3
        assert classify_failure(job, signature).anomaly == AnomalyKind.RECOVERY_LOOP

    def test_a_changing_failure_is_not_a_loop(self):
        job = Job(claim_id="c", action_type="inspect_training_curve", attempts=2)
        job.recovery_actions = ["failure:nan_loss:diverged at epoch 7"]
        finding = classify_failure(job, "resource_exhausted:out of memory")
        assert finding.anomaly == AnomalyKind.RESOURCE_EXHAUSTED


class TestRegistryCoverage:
    """Nothing in the registry is decorative: every action has a real invoker."""

    def test_every_planner_action_has_an_executor(self):
        from labguard.actions.executors import EXECUTORS
        from labguard.actions.registry import REGISTRY

        missing = [
            spec.name for spec in REGISTRY.values() if spec.invoked_by == "planner" and spec.name not in EXECUTORS
        ]
        assert not missing, f"planner-scheduled actions with no executor: {missing}"

    def test_every_recovery_action_is_reachable_from_an_anomaly(self):
        from labguard.actions.registry import RECOVERY_FOR_ANOMALY, REGISTRY

        reachable = set(RECOVERY_FOR_ANOMALY.values())
        unreachable = [
            spec.name for spec in REGISTRY.values() if spec.invoked_by == "runmedic" and spec.name not in reachable
        ]
        assert not unreachable, f"repairs no anomaly can trigger: {unreachable}"

    def test_every_action_declares_who_invokes_it(self):
        from labguard.actions.registry import REGISTRY

        for spec in REGISTRY.values():
            assert spec.invoked_by in {"planner", "runmedic", "orchestrator"}

    def test_every_anomaly_with_a_repair_maps_to_a_real_action(self):
        from labguard.actions.registry import RECOVERY_FOR_ANOMALY, REGISTRY
        from labguard.models.enums import AnomalyKind

        for anomaly, action in RECOVERY_FOR_ANOMALY.items():
            assert AnomalyKind(anomaly)
            assert action in REGISTRY
