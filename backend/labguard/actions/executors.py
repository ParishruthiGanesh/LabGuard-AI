"""Implementations behind the action registry.

Each executor runs real computation on the synthetic benchmark and returns
structured measurements plus the evidence those measurements justify.  No
executor takes free-form code or commands: every input has already been
validated against the registry's pydantic model.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from ..experiments import metrics as M
from ..experiments.dataset import make_dataset
from ..experiments.scenario import arms as scenario_arms
from ..experiments.scenario import find_config
from ..experiments.trainer import (
    FaultProfile,
    Trainer,
    TrainingFailure,
    TrainResult,
)
from ..models.domain import (
    Claim,
    DatasetInfo,
    EpochRecord,
    Evidence,
    Job,
    ModelConfig,
)
from ..models.enums import AnomalyKind, EvidenceStance

#: The corrected evaluation protocol every diagnostic uses: checkpoints chosen
#: on validation, with early stopping, rather than the reported run's protocol.
HONEST_PROTOCOL: dict[str, Any] = {"checkpoint_selection": "validation", "early_stopping_patience": 8}

EpochHook = Callable[[EpochRecord], Awaitable[None]]
RunStartHook = Callable[[str], Awaitable[None]]


@dataclass
class ExecutionContext:
    """Everything an executor is allowed to touch."""

    claim: Claim
    job: Job
    on_epoch: EpochHook
    on_run_start: RunStartHook
    #: Writes an artifact and returns its URI.
    write_artifact: Callable[[str, Any], Awaitable[str]]
    #: RunMedic raises this to stop a diverging run. The trainer still
    #: finalises, so the best validated checkpoint is kept.
    should_stop: Callable[[], bool] = lambda: False

    @property
    def dataset_info(self) -> DatasetInfo:
        return self.claim.context.dataset


@dataclass
class ActionOutcome:
    result: dict[str, Any] = field(default_factory=dict)
    evidence: list[Evidence] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    #: Curve of the last sub-run, surfaced on the Live Run Health page.
    curves: list[EpochRecord] = field(default_factory=list)


def _evidence(
    ctx: ExecutionContext,
    statement: str,
    stance: EvidenceStance,
    measurements: dict[str, Any],
    strength: float,
) -> Evidence:
    return Evidence(
        claim_id=ctx.claim.id,
        job_id=ctx.job.id,
        stance=stance,
        statement=statement,
        measurements=measurements,
        strength=round(min(1.0, max(0.05, strength)), 3),
    )


async def _train_streaming(
    ctx: ExecutionContext,
    label: str,
    config: ModelConfig,
    seed: int,
    **kwargs: Any,
) -> tuple[TrainResult, list[EpochRecord]]:
    """Train one configuration, streaming each epoch to the dashboard."""
    data = make_dataset(ctx.dataset_info, seed)
    trainer = Trainer(config, data, seed, **kwargs)
    await ctx.on_run_start(label)
    for record in trainer.epochs_iter():
        await ctx.on_epoch(record)
        if ctx.should_stop():
            trainer.early_stopped = True
            trainer.notes.append(f"stopped by RunMedic at epoch {record.epoch}; the best validated checkpoint is kept")
            break
    return trainer.finalize(), trainer.curves


def _apply_job_overrides(job: Job, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Fold RunMedic's recovery adjustments into the next training attempt."""
    overrides = job.params.get("_recovery_overrides") or {}
    if "learning_rate" in overrides:
        kwargs["learning_rate_override"] = float(overrides["learning_rate"])
    if "batch_size" in overrides:
        kwargs["batch_size_override"] = int(overrides["batch_size"])
    if "early_stopping_patience" in overrides:
        kwargs["early_stopping_patience"] = int(overrides["early_stopping_patience"])
    return kwargs


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------


async def compare_configurations(ctx: ExecutionContext, params: dict[str, Any]) -> ActionOutcome:
    """Diff the two training configurations and flag unfair comparisons."""
    baseline, candidate = scenario_arms(ctx.claim.context)
    fields = list(params.get("fields") or [])
    diff: dict[str, dict[str, Any]] = {}
    for name in fields:
        a, b = getattr(baseline, name, None), getattr(candidate, name, None)
        diff[name] = {"baseline": a, "candidate": b, "equal": a == b}

    unequal = [k for k, v in diff.items() if not v["equal"]]
    epoch_ratio = 0.0
    if baseline.epochs:
        epoch_ratio = round(candidate.epochs / baseline.epochs, 2)

    selections = {r.model_name: r.checkpoint_selected_on for r in ctx.claim.context.existing_results}
    test_selected = [n for n, s in selections.items() if s == "test"]

    result = {
        "baseline": baseline.name,
        "candidate": candidate.name,
        "diff": diff,
        "unequal_fields": unequal,
        "training_budget_ratio": epoch_ratio,
        "checkpoint_selection_by_model": selections,
        "models_selected_on_test": test_selected,
    }
    uri = await ctx.write_artifact("configuration_diff.json", result)

    evidence: list[Evidence] = []
    if epoch_ratio > 1.2:
        evidence.append(
            _evidence(
                ctx,
                f"{candidate.name} was trained {epoch_ratio}x longer than {baseline.name} "
                f"({candidate.epochs} vs {baseline.epochs} epochs), so the reported gap "
                f"confounds architecture with training budget.",
                EvidenceStance.CONTRADICTS,
                {"candidate_epochs": candidate.epochs, "baseline_epochs": baseline.epochs, "ratio": epoch_ratio},
                strength=0.75,
            )
        )
    if test_selected:
        evidence.append(
            _evidence(
                ctx,
                "Checkpoints for " + ", ".join(sorted(test_selected)) + " were selected on the test split, "
                "which optimistically biases the reported numbers.",
                EvidenceStance.CONTRADICTS,
                {"models_selected_on_test": test_selected},
                strength=0.7,
            )
        )
    if baseline.class_weight != candidate.class_weight:
        evidence.append(
            _evidence(
                ctx,
                f"The arms use different class weighting ({baseline.name}: {baseline.class_weight}, "
                f"{candidate.name}: {candidate.class_weight}), so accuracy is not comparable between them "
                f"without a class-balanced metric.",
                EvidenceStance.NEUTRAL,
                {"baseline_class_weight": baseline.class_weight, "candidate_class_weight": candidate.class_weight},
                strength=0.6,
            )
        )
    if not evidence:
        evidence.append(
            _evidence(
                ctx,
                "The two configurations are matched on every compared field.",
                EvidenceStance.SUPPORTS,
                {"compared_fields": fields},
                strength=0.5,
            )
        )
    return ActionOutcome(result=result, evidence=evidence, artifacts=[uri])


async def check_data_overlap(ctx: ExecutionContext, params: dict[str, Any]) -> ActionOutcome:
    """Hash every row and look for train/test duplication."""
    seed = int(params.get("seed", 11))
    data = make_dataset(ctx.dataset_info, seed)
    train_set = set(data.train_hashes)
    overlap = sorted(train_set.intersection(data.test_hashes))
    result: dict[str, Any] = {
        "seed": seed,
        "n_train": len(data.train_hashes),
        "n_test": len(data.test_hashes),
        "overlapping_rows": len(overlap),
        "overlap_fraction": round(len(overlap) / max(1, len(data.test_hashes)), 5),
        "sample_hashes": overlap[:5],
    }

    # Positive control: prove the detector fires when leakage is present.
    if params.get("verify_detector_with_positive_control", True):
        leaky_info = ctx.dataset_info.model_copy(update={"inject_train_test_overlap": 25})
        leaky = make_dataset(leaky_info, seed)
        detected = len(set(leaky.train_hashes).intersection(leaky.test_hashes))
        result["positive_control"] = {
            "injected_rows": leaky.injected_overlap,
            "detected_rows": detected,
            "detector_working": detected >= leaky.injected_overlap,
        }

    uri = await ctx.write_artifact("data_overlap.json", result)
    control = result.get("positive_control", {})
    if overlap:
        ev = _evidence(
            ctx,
            f"{len(overlap)} of {len(data.test_hashes)} test rows are byte-identical to training rows; "
            f"the reported numbers are inflated by leakage.",
            EvidenceStance.CONTRADICTS,
            result,
            strength=0.9,
        )
    else:
        ev = _evidence(
            ctx,
            f"No train/test overlap found across {len(data.test_hashes)} test rows. The detector was "
            f"verified on a positive control where {control.get('injected_rows', 0)} injected duplicates "
            f"were all recovered.",
            EvidenceStance.SUPPORTS,
            result,
            strength=0.8,
        )
    return ActionOutcome(result=result, evidence=[ev], artifacts=[uri])


async def recalculate_metrics(ctx: ExecutionContext, params: dict[str, Any]) -> ActionOutcome:
    """Recompute the full metric bundle with bootstrap intervals."""
    seed = int(params.get("seed", 11))
    n_boot = int(params.get("bootstrap_samples", 200))
    baseline, candidate = scenario_arms(ctx.claim.context)

    kwargs = _apply_job_overrides(ctx.job, dict(HONEST_PROTOCOL))
    res_a, _ = await _train_streaming(ctx, f"{baseline.name} · seed {seed}", baseline, seed, **kwargs)
    res_b, curves_b = await _train_streaming(ctx, f"{candidate.name} · seed {seed}", candidate, seed, **kwargs)

    wanted = [m for m in (params.get("metrics") or []) if m in res_a.metrics]
    comparison: dict[str, Any] = {}
    for metric in wanted:
        a_val, b_val = float(res_a.metrics[metric]), float(res_b.metrics[metric])
        comparison[metric] = {
            "baseline": a_val,
            "candidate": b_val,
            "delta": round(b_val - a_val, 4),
            "candidate_better": b_val > a_val,
            "baseline_ci": M.bootstrap_ci(res_a.y_test, res_a.test_scores, metric, seed, n_boot)
            if metric != "roc_auc"
            else M.bootstrap_ci(res_a.y_test, res_a.test_scores, "roc_auc", seed, n_boot),
            "candidate_ci": M.bootstrap_ci(res_b.y_test, res_b.test_scores, metric, seed, n_boot),
        }

    flipped = [
        m
        for m, v in comparison.items()
        if m != "accuracy" and comparison.get("accuracy", {}).get("candidate_better") and not v["candidate_better"]
    ]
    result = {
        "seed": seed,
        "baseline": baseline.name,
        "candidate": candidate.name,
        "metrics": comparison,
        "metrics_that_reverse_the_conclusion": flipped,
        "positive_rate": res_a.metrics["positive_rate"],
    }
    uri = await ctx.write_artifact(f"recalculated_metrics_seed{seed}.json", result)

    evidence: list[Evidence] = []
    if flipped:
        detail = ", ".join(f"{m} {comparison[m]['delta']:+.4f}" for m in flipped)
        evidence.append(
            _evidence(
                ctx,
                f"On seed {seed} the candidate's accuracy advantage reverses under class-balanced "
                f"metrics ({detail}). Accuracy is a misleading summary at a "
                f"{res_a.metrics['positive_rate'] * 100:.1f}% positive rate.",
                EvidenceStance.CONTRADICTS,
                {"flipped": flipped, "comparison": comparison},
                strength=0.8,
            )
        )
    else:
        evidence.append(
            _evidence(
                ctx,
                f"On seed {seed} the candidate leads on every recomputed metric.",
                EvidenceStance.SUPPORTS,
                {"comparison": comparison},
                strength=0.6,
            )
        )
    overlapping = [m for m, v in comparison.items() if v["baseline_ci"]["high"] >= v["candidate_ci"]["low"]]
    if overlapping:
        evidence.append(
            _evidence(
                ctx,
                "Bootstrap confidence intervals overlap for "
                + ", ".join(overlapping)
                + f", so a single-seed difference at seed {seed} is not statistically separated.",
                EvidenceStance.CONTRADICTS,
                {"overlapping_metrics": overlapping},
                strength=0.65,
            )
        )
    return ActionOutcome(result=result, evidence=evidence, artifacts=[uri], curves=curves_b)


async def evaluate_classwise(ctx: ExecutionContext, params: dict[str, Any]) -> ActionOutcome:
    """Locate which class any advantage actually comes from."""
    seed = int(params.get("seed", 11))
    selection = str(params.get("checkpoint_selection", "validation"))
    baseline, candidate = scenario_arms(ctx.claim.context)
    kwargs = _apply_job_overrides(ctx.job, dict(HONEST_PROTOCOL) | {"checkpoint_selection": selection})

    res_a, _ = await _train_streaming(ctx, f"{baseline.name} · seed {seed}", baseline, seed, **kwargs)
    res_b, curves_b = await _train_streaming(ctx, f"{candidate.name} · seed {seed}", candidate, seed, **kwargs)

    per_class: dict[str, Any] = {}
    for label in ("class_0", "class_1"):
        a, b = res_a.metrics["classwise"][label], res_b.metrics["classwise"][label]
        per_class[label] = {
            "baseline": a,
            "candidate": b,
            "f1_delta": round(b["f1"] - a["f1"], 4),
            "recall_delta": round(b["recall"] - a["recall"], 4),
            "support": a["support"],
        }

    majority_gain = per_class["class_0"]["f1_delta"]
    minority_gain = per_class["class_1"]["f1_delta"]
    result = {
        "seed": seed,
        "checkpoint_selection": selection,
        "per_class": per_class,
        "accuracy_delta": round(res_b.metrics["accuracy"] - res_a.metrics["accuracy"], 4),
        "macro_f1_delta": round(res_b.metrics["macro_f1"] - res_a.metrics["macro_f1"], 4),
        "advantage_source": "majority_class" if majority_gain > minority_gain else "minority_class",
    }
    uri = await ctx.write_artifact(f"classwise_seed{seed}.json", result)

    if majority_gain > 0 and minority_gain < 0:
        ev = _evidence(
            ctx,
            f"The candidate's gain is confined to the majority class (class 0 F1 {majority_gain:+.4f}) "
            f"while the minority class gets worse (class 1 F1 {minority_gain:+.4f}, recall "
            f"{per_class['class_1']['recall_delta']:+.4f} on {per_class['class_1']['support']} positives). "
            f"That is the opposite of what a violence-detection improvement should look like.",
            EvidenceStance.CONTRADICTS,
            result,
            strength=0.85,
        )
    elif minority_gain > 0:
        ev = _evidence(
            ctx,
            f"The candidate improves the minority class (class 1 F1 {minority_gain:+.4f}), which is the "
            f"class the claim cares about.",
            EvidenceStance.SUPPORTS,
            result,
            strength=0.75,
        )
    else:
        ev = _evidence(
            ctx,
            "Neither class shows a clear per-class improvement for the candidate.",
            EvidenceStance.NEUTRAL,
            result,
            strength=0.5,
        )
    return ActionOutcome(result=result, evidence=[ev], artifacts=[uri], curves=curves_b)


async def run_seed_comparison(ctx: ExecutionContext, params: dict[str, Any]) -> ActionOutcome:
    """Retrain both arms across seeds under an equal budget and compare."""
    seeds = [int(s) for s in params["seeds"]]
    equalise = bool(params.get("equalise_training_budget", True))
    equal_epochs = int(params.get("equal_epochs", 45))
    selection = str(params.get("checkpoint_selection", "validation"))
    patience = int(params.get("early_stopping_patience", 8))
    wanted = list(params.get("metrics") or ["accuracy", "macro_f1", "balanced_accuracy"])

    baseline, candidate = scenario_arms(ctx.claim.context)
    kwargs: dict[str, Any] = {"checkpoint_selection": selection, "early_stopping_patience": patience}
    if equalise:
        kwargs["epochs_override"] = equal_epochs
    kwargs = _apply_job_overrides(ctx.job, kwargs)

    runs: list[dict[str, Any]] = []
    last_curves: list[EpochRecord] = []
    for seed in seeds:
        res_a, _ = await _train_streaming(ctx, f"{baseline.name} · seed {seed}", baseline, seed, **kwargs)
        res_b, curves_b = await _train_streaming(ctx, f"{candidate.name} · seed {seed}", candidate, seed, **kwargs)
        last_curves = curves_b
        runs.append(
            {
                "seed": seed,
                "baseline": {k: res_a.metrics[k] for k in wanted if k in res_a.metrics}
                | {"epochs_run": res_a.epochs_run, "selected_epoch": res_a.selected_epoch},
                "candidate": {k: res_b.metrics[k] for k in wanted if k in res_b.metrics}
                | {"epochs_run": res_b.epochs_run, "selected_epoch": res_b.selected_epoch},
                "minority_recall": {
                    "baseline": res_a.metrics["minority_recall"],
                    "candidate": res_b.metrics["minority_recall"],
                },
            }
        )

    summaries: dict[str, Any] = {}
    for metric in wanted:
        deltas = [r["candidate"][metric] - r["baseline"][metric] for r in runs if metric in r["candidate"]]
        if deltas:
            summaries[metric] = M.paired_seed_summary(deltas)

    result = {
        "seeds": seeds,
        "equalised_training_budget": equalise,
        "epochs_per_arm": equal_epochs if equalise else None,
        "checkpoint_selection": selection,
        "early_stopping_patience": patience,
        "runs": runs,
        "paired_summary": summaries,
    }
    uri = await ctx.write_artifact("seed_comparison.json", result)

    evidence: list[Evidence] = []
    for metric, summary in summaries.items():
        n = summary["n_seeds"]
        wins, losses = summary["wins_for_b"], summary["losses_for_b"]
        strength = min(0.95, 0.45 + 0.1 * n)
        if summary["ci_includes_zero"]:
            stance = EvidenceStance.CONTRADICTS if wins <= losses else EvidenceStance.NEUTRAL
            statement = (
                f"Across {n} seeds the {metric} difference averages {summary['mean_delta']:+.4f} with a 95% "
                f"interval of [{summary['ci_low']:+.4f}, {summary['ci_high']:+.4f}], which contains zero. "
                f"The candidate wins on {wins} of {n} seeds."
            )
        elif summary["mean_delta"] > 0:
            stance = EvidenceStance.SUPPORTS
            statement = (
                f"Across {n} seeds the candidate leads on {metric} by {summary['mean_delta']:+.4f} "
                f"[{summary['ci_low']:+.4f}, {summary['ci_high']:+.4f}], winning {wins} of {n} seeds."
            )
        else:
            stance = EvidenceStance.CONTRADICTS
            statement = (
                f"Across {n} seeds the candidate is *worse* on {metric} by {summary['mean_delta']:+.4f} "
                f"[{summary['ci_low']:+.4f}, {summary['ci_high']:+.4f}], losing {losses} of {n} seeds."
            )
        evidence.append(_evidence(ctx, statement, stance, {"metric": metric, **summary}, strength))

    if equalise:
        evidence.append(
            _evidence(
                ctx,
                f"Both arms were retrained under an identical {equal_epochs}-epoch budget with checkpoints "
                f"selected on validation, removing the training-budget and checkpoint confounds from the "
                f"original comparison.",
                EvidenceStance.NEUTRAL,
                {"equal_epochs": equal_epochs, "checkpoint_selection": selection},
                strength=0.6,
            )
        )
    return ActionOutcome(result=result, evidence=evidence, artifacts=[uri], curves=last_curves)


async def inspect_training_curve(ctx: ExecutionContext, params: dict[str, Any]) -> ActionOutcome:
    """Replay one run and classify its training health."""
    from ..agents.health import analyse_curve, summarise

    seed = int(params.get("seed", 11))
    name = str(params.get("config_name") or "")
    epochs = int(params.get("epochs", 90))
    config = find_config(ctx.claim.context, name) if name else None
    if config is None:
        _, config = scenario_arms(ctx.claim.context)

    kwargs = _apply_job_overrides(ctx.job, {"checkpoint_selection": "validation", "epochs_override": epochs})
    res, curves = await _train_streaming(ctx, f"{config.name} · seed {seed}", config, seed, **kwargs)

    findings = analyse_curve(curves)
    status, summary = summarise(findings)
    result = {
        "config": config.name,
        "seed": seed,
        "epochs_run": res.epochs_run,
        "best_validation_epoch": res.best_val_epoch,
        "best_test_epoch": res.best_test_epoch,
        "selected_epoch": res.selected_epoch,
        "early_stopped": res.early_stopped,
        "health_status": status.value,
        "health_summary": summary,
        "findings": [
            {
                "anomaly": f.anomaly.value,
                "status": f.status.value,
                "detail": f.detail,
                "epoch": f.epoch,
                "evidence": f.evidence,
            }
            for f in findings
        ],
        "metrics_at_selected_checkpoint": {
            k: res.metrics[k] for k in ("accuracy", "macro_f1", "balanced_accuracy", "minority_recall")
        },
        "metrics_at_final_epoch": {
            k: res.final_epoch_metrics[k] for k in ("accuracy", "macro_f1", "balanced_accuracy", "minority_recall")
        },
        "curve": [c.model_dump(mode="json") for c in curves],
    }
    uri = await ctx.write_artifact(f"curve_{config.name.replace(' ', '_')}_seed{seed}.json", result)

    evidence: list[Evidence] = []
    overfit = next((f for f in findings if f.anomaly == AnomalyKind.OVERFITTING), None)
    if overfit:
        gap = round(
            res.final_epoch_metrics["macro_f1"] - res.metrics["macro_f1"],
            4,
        )
        evidence.append(
            _evidence(
                ctx,
                f"{config.name} overfits: {overfit.detail}. The epoch-{res.epochs_run} checkpoint the original "
                f"result used scores {gap:+.4f} macro F1 against the validation-selected epoch-"
                f"{res.best_val_epoch} checkpoint.",
                EvidenceStance.CONTRADICTS,
                {"finding": overfit.evidence or {}, "macro_f1_gap_final_vs_selected": gap},
                strength=0.75,
            )
        )
    if res.best_test_epoch != res.best_val_epoch:
        evidence.append(
            _evidence(
                ctx,
                f"The epoch that looks best on the test split (epoch {res.best_test_epoch}) is not the epoch "
                f"validation would have chosen (epoch {res.best_val_epoch}), so selecting on test buys a gain "
                f"that honest checkpointing does not.",
                EvidenceStance.CONTRADICTS,
                {"best_test_epoch": res.best_test_epoch, "best_val_epoch": res.best_val_epoch},
                strength=0.7,
            )
        )
    if not evidence:
        evidence.append(
            _evidence(
                ctx,
                f"{config.name} trained cleanly over {res.epochs_run} epochs with no anomaly detected.",
                EvidenceStance.NEUTRAL,
                {"health": status.value},
                strength=0.5,
            )
        )
    return ActionOutcome(result=result, evidence=evidence, artifacts=[uri], curves=curves)


async def sweep_decision_threshold(ctx: ExecutionContext, params: dict[str, Any]) -> ActionOutcome:
    """Give each arm its own best threshold before comparing them."""
    seed = int(params.get("seed", 11))
    metric = str(params.get("metric", "macro_f1"))
    baseline, candidate = scenario_arms(ctx.claim.context)
    kwargs = _apply_job_overrides(ctx.job, dict(HONEST_PROTOCOL))

    res_a, _ = await _train_streaming(ctx, f"{baseline.name} · seed {seed}", baseline, seed, **kwargs)
    res_b, curves_b = await _train_streaming(ctx, f"{candidate.name} · seed {seed}", candidate, seed, **kwargs)

    t_a, best_a = M.best_threshold_by_macro_f1(res_a.y_test, res_a.test_scores)
    t_b, best_b = M.best_threshold_by_macro_f1(res_b.y_test, res_b.test_scores)
    default_a = float(res_a.metrics[metric])
    default_b = float(res_b.metrics[metric])

    result = {
        "seed": seed,
        "metric": metric,
        "at_default_threshold": {
            "baseline": default_a,
            "candidate": default_b,
            "delta": round(default_b - default_a, 4),
        },
        "at_tuned_threshold": {
            "baseline": {"threshold": t_a, metric: best_a},
            "candidate": {"threshold": t_b, metric: best_b},
            "delta": round(best_b - best_a, 4),
        },
        "conclusion_changes": (default_b > default_a) != (best_b > best_a),
    }
    uri = await ctx.write_artifact(f"threshold_sweep_seed{seed}.json", result)

    if result["conclusion_changes"]:
        ev = _evidence(
            ctx,
            f"Which model wins depends on the decision threshold: at 0.5 the delta is "
            f"{result['at_default_threshold']['delta']:+.4f}, but with each arm at its own best threshold "
            f"({t_a} vs {t_b}) it is {result['at_tuned_threshold']['delta']:+.4f}.",
            EvidenceStance.CONTRADICTS,
            result,
            strength=0.7,
        )
    else:
        ev = _evidence(
            ctx,
            f"The ordering survives threshold tuning: delta {result['at_default_threshold']['delta']:+.4f} at 0.5 "
            f"and {result['at_tuned_threshold']['delta']:+.4f} at per-arm best thresholds.",
            EvidenceStance.SUPPORTS if best_b > best_a else EvidenceStance.CONTRADICTS,
            result,
            strength=0.7,
        )
    return ActionOutcome(result=result, evidence=[ev], artifacts=[uri], curves=curves_b)


async def test_domain_shift(ctx: ExecutionContext, params: dict[str, Any]) -> ActionOutcome:
    """Re-evaluate both arms under a covariate shift on the test split."""
    seed = int(params.get("seed", 11))
    strength = float(params.get("shift_strength", 0.35))
    baseline, candidate = scenario_arms(ctx.claim.context)
    kwargs = _apply_job_overrides(ctx.job, dict(HONEST_PROTOCOL))

    clean_a, _ = await _train_streaming(ctx, f"{baseline.name} · clean", baseline, seed, **kwargs)
    clean_b, curves_b = await _train_streaming(ctx, f"{candidate.name} · clean", candidate, seed, **kwargs)

    shifted_info = ctx.dataset_info.model_copy(update={"domain_shift_strength": strength})
    shifted = make_dataset(shifted_info, seed)

    def _score(train_result_config: ModelConfig) -> dict[str, Any]:
        trainer = Trainer(train_result_config, shifted, seed, **kwargs)
        for _ in trainer.epochs_iter():
            pass
        res = trainer.finalize()
        return {k: res.metrics[k] for k in ("accuracy", "macro_f1", "balanced_accuracy")}

    shift_a, shift_b = _score(baseline), _score(candidate)
    drop_a = round(shift_a["macro_f1"] - clean_a.metrics["macro_f1"], 4)
    drop_b = round(shift_b["macro_f1"] - clean_b.metrics["macro_f1"], 4)

    result = {
        "seed": seed,
        "shift_strength": strength,
        "clean": {"baseline": clean_a.metrics["macro_f1"], "candidate": clean_b.metrics["macro_f1"]},
        "shifted": {"baseline": shift_a["macro_f1"], "candidate": shift_b["macro_f1"]},
        "macro_f1_drop": {"baseline": drop_a, "candidate": drop_b},
        "candidate_degrades_more": drop_b < drop_a,
    }
    uri = await ctx.write_artifact(f"domain_shift_seed{seed}.json", result)
    ev = _evidence(
        ctx,
        f"Under a covariate shift of {strength}, the candidate's macro F1 moves {drop_b:+.4f} against the "
        f"baseline's {drop_a:+.4f}.",
        EvidenceStance.CONTRADICTS if drop_b < drop_a else EvidenceStance.SUPPORTS,
        result,
        strength=0.6,
    )
    return ActionOutcome(result=result, evidence=[ev], artifacts=[uri], curves=curves_b)


async def resume_from_checkpoint(ctx: ExecutionContext, params: dict[str, Any]) -> ActionOutcome:
    """Verify that a reported checkpoint reproduces its reported number.

    Raises `TrainingFailure` when the checkpoint fails its integrity check —
    which is what the corrupted-checkpoint and recovery-loop paths exercise.
    """
    reported = next(
        (r for r in ctx.claim.context.existing_results if r.checkpoint_uri),
        None,
    )
    if reported is None:
        raise TrainingFailure(AnomalyKind.MISSING_ARTIFACT, "no reported checkpoint URI was supplied")

    fault = (
        FaultProfile(kind="corrupted_checkpoint") if ctx.claim.context.reported_checkpoint_corrupt else FaultProfile()
    )
    config = find_config(ctx.claim.context, reported.model_name)
    if config is None:
        _, config = scenario_arms(ctx.claim.context)

    data = make_dataset(ctx.dataset_info, reported.seed)
    trainer = Trainer(config, data, reported.seed, attempt=ctx.job.attempts, fault=fault, checkpoint_selection="last")
    await ctx.on_run_start(f"verify checkpoint · {reported.model_name}")
    for record in trainer.epochs_iter():  # raises on a corrupt checkpoint
        await ctx.on_epoch(record)
    res = trainer.finalize()

    reproduced = float(res.metrics.get(reported.metric, 0.0))
    delta = round(reproduced - reported.value, 4)
    result = {
        "checkpoint_uri": reported.checkpoint_uri,
        "reported_metric": reported.metric,
        "reported_value": reported.value,
        "reproduced_value": reproduced,
        "delta": delta,
        "reproduces_within_tolerance": abs(delta) <= 0.01,
    }
    uri = await ctx.write_artifact("checkpoint_verification.json", result)
    ev = _evidence(
        ctx,
        f"The reported checkpoint reproduces {reported.metric} at {reproduced:.4f} against the reported "
        f"{reported.value:.4f} ({delta:+.4f}).",
        EvidenceStance.SUPPORTS if result["reproduces_within_tolerance"] else EvidenceStance.CONTRADICTS,
        result,
        strength=0.7,
    )
    return ActionOutcome(result=result, evidence=[ev], artifacts=[uri], curves=trainer.curves)


#: Executors keyed by registry action name. Recovery actions are applied by
#: RunMedic to the parent job rather than run standalone, and the report action
#: is fulfilled by the orchestrator, which owns the scoring inputs.
EXECUTORS: dict[str, Any] = {
    "compare_configurations": compare_configurations,
    "check_data_overlap": check_data_overlap,
    "recalculate_metrics": recalculate_metrics,
    "evaluate_classwise": evaluate_classwise,
    "run_seed_comparison": run_seed_comparison,
    "inspect_training_curve": inspect_training_curve,
    "sweep_decision_threshold": sweep_decision_threshold,
    "test_domain_shift": test_domain_shift,
    "resume_from_checkpoint": resume_from_checkpoint,
}
