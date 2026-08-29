"""The synthetic benchmark must actually contain the weaknesses we claim."""

from __future__ import annotations

import numpy as np
import pytest

from labguard.experiments.dataset import make_dataset
from labguard.experiments.metrics import bootstrap_ci, compute_all, paired_seed_summary, roc_auc
from labguard.experiments.scenario import (
    DEMO_DATASET,
    MODEL_A,
    MODEL_B,
    MODEL_B_VARIANT,
    VERIFICATION_SEEDS,
)
from labguard.experiments.trainer import Trainer, TrainingFailure, run_to_completion
from labguard.models.enums import AnomalyKind


def test_dataset_is_imbalanced_and_deterministic():
    a = make_dataset(DEMO_DATASET, 11)
    b = make_dataset(DEMO_DATASET, 11)
    assert np.array_equal(a.x_train, b.x_train)
    assert np.array_equal(a.y_train, b.y_train)
    assert 0.05 < a.positive_rate_train < 0.25


def test_clean_split_has_no_overlap_but_injection_is_detected():
    clean = make_dataset(DEMO_DATASET, 11)
    assert not set(clean.train_hashes) & set(clean.test_hashes)

    leaky = make_dataset(DEMO_DATASET.model_copy(update={"inject_train_test_overlap": 25}), 11)
    assert len(set(leaky.train_hashes) & set(leaky.test_hashes)) >= 25


def test_candidate_wins_accuracy_but_loses_balanced_metrics():
    """The core of the demo: accuracy and macro F1 disagree."""
    acc_deltas, f1_deltas, ba_deltas = [], [], []
    for seed in VERIFICATION_SEEDS:
        data = make_dataset(DEMO_DATASET, seed)
        a = run_to_completion(
            Trainer(
                MODEL_A, data, seed, checkpoint_selection="validation", early_stopping_patience=8, epochs_override=45
            )
        )
        b = run_to_completion(
            Trainer(
                MODEL_B, data, seed, checkpoint_selection="validation", early_stopping_patience=8, epochs_override=45
            )
        )
        acc_deltas.append(b.metrics["accuracy"] - a.metrics["accuracy"])
        f1_deltas.append(b.metrics["macro_f1"] - a.metrics["macro_f1"])
        ba_deltas.append(b.metrics["balanced_accuracy"] - a.metrics["balanced_accuracy"])

    accuracy = paired_seed_summary(acc_deltas)
    macro_f1 = paired_seed_summary(f1_deltas)
    balanced = paired_seed_summary(ba_deltas)

    # Accuracy: not separated from zero, so the headline claim is unstable.
    assert accuracy["ci_includes_zero"]
    # Class-balanced metrics: the candidate is genuinely worse.
    assert macro_f1["mean_delta"] < 0 and not macro_f1["ci_includes_zero"]
    assert balanced["mean_delta"] < 0 and not balanced["ci_includes_zero"]


def test_candidate_gains_on_the_majority_class_only():
    data = make_dataset(DEMO_DATASET, 11)
    a = run_to_completion(Trainer(MODEL_A, data, 11, checkpoint_selection="validation", early_stopping_patience=8))
    b = run_to_completion(Trainer(MODEL_B, data, 11, checkpoint_selection="validation", early_stopping_patience=8))
    assert b.metrics["classwise"]["class_0"]["f1"] > a.metrics["classwise"]["class_0"]["f1"]
    assert b.metrics["minority_recall"] < a.metrics["minority_recall"]


def test_reported_run_overfits():
    """Model B's 90-epoch run diverges on validation while training improves."""
    data = make_dataset(DEMO_DATASET, 4)
    result = run_to_completion(Trainer(MODEL_B, data, 4, checkpoint_selection="validation"))
    losses = [c.val_loss for c in result.curves]
    best = min(range(len(losses)), key=lambda i: losses[i])
    assert losses[-1] > losses[best] * 1.2, "validation loss should climb well above its best"
    assert result.curves[-1].train_loss < result.curves[best].train_loss


def test_selecting_on_test_picks_a_different_checkpoint_than_validation():
    data = make_dataset(DEMO_DATASET, 11)
    result = run_to_completion(Trainer(MODEL_B, data, 11, checkpoint_selection="test"))
    assert result.best_test_epoch != result.best_val_epoch


def test_unstable_variant_genuinely_diverges_and_a_lower_rate_recovers():
    data = make_dataset(DEMO_DATASET, 3)
    with pytest.raises(TrainingFailure) as exc:
        run_to_completion(Trainer(MODEL_B_VARIANT, data, 3))
    assert exc.value.anomaly == AnomalyKind.NAN_LOSS

    # The same configuration at the recovered learning rate completes.
    recovered = run_to_completion(Trainer(MODEL_B_VARIANT, data, 3, learning_rate_override=0.5))
    assert recovered.epochs_run > 0
    assert np.isfinite(recovered.curves[-1].val_loss)


def test_out_of_memory_is_raised_and_a_smaller_batch_fits():
    data = make_dataset(DEMO_DATASET, 3)
    with pytest.raises(TrainingFailure) as exc:
        run_to_completion(Trainer(MODEL_B, data, 3, batch_size_override=4096))
    assert exc.value.anomaly == AnomalyKind.RESOURCE_EXHAUSTED
    assert run_to_completion(Trainer(MODEL_B, data, 3, batch_size_override=64)).epochs_run > 0


def test_corrupted_checkpoint_fails_identically_every_attempt():
    from labguard.experiments.trainer import FaultProfile

    data = make_dataset(DEMO_DATASET, 11)
    signatures = set()
    for attempt in (1, 2, 3):
        with pytest.raises(TrainingFailure) as exc:
            run_to_completion(
                Trainer(MODEL_B, data, 11, attempt=attempt, fault=FaultProfile(kind="corrupted_checkpoint"))
            )
        signatures.add(exc.value.signature)
    assert len(signatures) == 1, "an unchanging failure must have an unchanging signature"


def test_metric_helpers():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = compute_all(y, scores)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert roc_auc(y, scores) == 1.0

    ci = bootstrap_ci(y, scores, "accuracy", seed=0, n_boot=50)
    assert 0.0 <= ci["low"] <= ci["high"] <= 1.0

    summary = paired_seed_summary([0.1, -0.1, 0.05])
    assert summary["n_seeds"] == 3 and summary["wins_for_b"] == 2
