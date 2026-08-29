"""A small, fast, fully reproducible training loop.

Deliberately not a deep-learning framework: a numpy logistic / one-hidden-layer
model trained with mini-batch SGD.  Runs finish in well under a second, which
is what makes recursive verification affordable.

The loop emits one `EpochRecord` per epoch as a generator, so the worker can
persist partial curves and the dashboard can watch a run progress.

Resource figures (`gpu_util_pct`, `memory_mb`) come from an explicit analytic
model of the configuration, not from a real accelerator.  They are surfaced in
the UI labelled as simulated.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..models.domain import EpochRecord, ModelConfig
from ..models.enums import AnomalyKind
from .dataset import Dataset, split_validation
from .metrics import compute_all

#: Simulated accelerator memory ceiling in MB, used by the OOM scenario.
DEVICE_MEMORY_MB = 8192.0


class TrainingFailure(RuntimeError):
    """A run-level failure RunMedic is expected to classify and possibly repair."""

    def __init__(self, anomaly: AnomalyKind, message: str, epoch: int | None = None) -> None:
        super().__init__(message)
        self.anomaly = anomaly
        self.epoch = epoch

    @property
    def signature(self) -> str:
        """Stable identity of this failure, used to detect recovery loops."""
        return f"{self.anomaly.value}:{str(self)[:80]}"


@dataclass
class FaultProfile:
    """Deliberate weaknesses injected into a run by the demo scenario."""

    #: One of "", "transient", "resource_exhausted", "corrupted_checkpoint".
    #: Numerical divergence is *not* injected here — it is produced by a
    #: genuinely unstable configuration (see `ModelConfig.objective`).
    kind: str = ""
    #: Attempt numbers on which the fault fires. Empty means every attempt.
    fires_on_attempts: tuple[int, ...] = ()

    def active(self, attempt: int) -> bool:
        if not self.kind:
            return False
        if not self.fires_on_attempts:
            return True
        return attempt in self.fires_on_attempts


@dataclass
class TrainResult:
    config: dict[str, Any]
    seed: int
    curves: list[EpochRecord]
    #: Metrics using the checkpoint chosen by `checkpoint_selection`.
    metrics: dict[str, Any]
    #: Metrics of the final epoch's weights, regardless of selection.
    final_epoch_metrics: dict[str, Any]
    test_scores: np.ndarray
    y_test: np.ndarray
    selected_epoch: int
    best_val_epoch: int
    best_test_epoch: int
    epochs_run: int
    early_stopped: bool = False
    peak_memory_mb: float = 0.0
    mean_gpu_util_pct: float = 0.0
    notes: list[str] = field(default_factory=list)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


def _bce(p: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None) -> float:
    eps = 1e-9
    terms = -(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
    if weights is None:
        return float(np.mean(terms))
    return float(np.sum(terms * weights) / np.sum(weights))


class Trainer:
    """Mini-batch SGD over a linear or one-hidden-layer model."""

    def __init__(
        self,
        config: ModelConfig,
        data: Dataset,
        seed: int,
        *,
        attempt: int = 1,
        fault: FaultProfile | None = None,
        checkpoint_selection: str = "validation",
        early_stopping_patience: int | None = None,
        learning_rate_override: float | None = None,
        batch_size_override: int | None = None,
        epochs_override: int | None = None,
    ) -> None:
        self.config = config
        self.data = data
        self.seed = seed
        self.attempt = attempt
        self.fault = fault or FaultProfile()
        self.checkpoint_selection = checkpoint_selection
        self.early_stopping_patience = early_stopping_patience
        self.learning_rate = float(
            learning_rate_override if learning_rate_override is not None else config.learning_rate
        )
        self.batch_size = int(batch_size_override if batch_size_override is not None else config.batch_size)
        self.epochs = int(epochs_override if epochs_override is not None else config.epochs)

        self.x_tr, self.y_tr, self.x_val, self.y_val = split_validation(data, seed)
        self.class_weights = self._class_weights()
        self._rng = np.random.default_rng(seed * 7919 + 13)
        self._init_params()

        self.curves: list[EpochRecord] = []
        self._checkpoints: list[dict[str, np.ndarray]] = []
        self._val_metric_history: list[float] = []
        self._test_metric_history: list[float] = []
        self.early_stopped = False
        self._peak_memory = 0.0
        self._gpu_utils: list[float] = []
        self.notes: list[str] = []

    def _class_weights(self) -> np.ndarray:
        """Per-class loss weights.

        "balanced" is full inverse class frequency; "sqrt_balanced" damps it
        with an exponent of 0.5, which is the common practical compromise on
        heavily imbalanced data.
        """
        power = {"balanced": 1.0, "sqrt_balanced": 0.5}.get(str(self.config.class_weight).lower(), 0.0)
        if power == 0.0:
            return np.ones(2)
        counts = np.array([np.sum(self.y_tr == 0), np.sum(self.y_tr == 1)], dtype=float)
        counts = np.maximum(counts, 1.0)
        return (float(len(self.y_tr)) / (2.0 * counts)) ** power

    def _sample_weights(self, y: np.ndarray) -> np.ndarray:
        return self.class_weights[y.astype(int)]

    # -- model ------------------------------------------------------------

    def _init_params(self) -> None:
        d = self.data.n_features
        h = int(self.config.hidden_units)
        scale = 0.35
        if h > 0:
            self.w1 = self._rng.normal(0, scale, size=(d, h))
            self.b1 = np.zeros(h)
            self.w2 = self._rng.normal(0, scale, size=h)
            self.b2 = 0.0
        else:
            self.w1 = self._rng.normal(0, scale, size=d)
            self.b1 = 0.0

    def _logits(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        if int(self.config.hidden_units) > 0:
            hidden = np.tanh(x @ self.w1 + self.b1)
            return hidden @ self.w2 + self.b2, hidden
        return x @ self.w1 + self.b1, None

    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        z, hidden = self._logits(x)
        return _sigmoid(z), hidden

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self._forward(x)[0]

    def _snapshot(self) -> dict[str, np.ndarray]:
        if int(self.config.hidden_units) > 0:
            return {"w1": self.w1.copy(), "b1": self.b1.copy(), "w2": self.w2.copy(), "b2": np.array(self.b2)}
        return {"w1": self.w1.copy(), "b1": np.array(self.b1)}

    def _restore(self, snap: dict[str, np.ndarray]) -> None:
        self.w1 = snap["w1"].copy()
        if int(self.config.hidden_units) > 0:
            self.b1 = snap["b1"].copy()
            self.w2 = snap["w2"].copy()
            self.b2 = float(snap["b2"])
        else:
            self.b1 = float(snap["b1"])

    # -- resource model ---------------------------------------------------

    def _resource_estimate(self) -> tuple[float, float]:
        """Analytic (simulated) memory footprint and utilisation."""
        h = max(1, int(self.config.hidden_units))
        memory = 640.0 + self.batch_size * h * self.data.n_features * 0.0125
        util = min(99.0, 22.0 + self.batch_size * 0.09 + h * 0.55)
        return memory, util

    # -- training ---------------------------------------------------------

    def epochs_iter(self) -> Iterator[EpochRecord]:
        """Yield one record per epoch; raises `TrainingFailure` on a fault."""
        if self.fault.kind == "corrupted_checkpoint" and self.fault.active(self.attempt):
            raise TrainingFailure(
                AnomalyKind.CORRUPTED_CHECKPOINT,
                "checkpoint header failed CRC validation; weights unreadable",
                epoch=0,
            )
        if self.fault.kind == "transient" and self.fault.active(self.attempt):
            raise TrainingFailure(
                AnomalyKind.MISSING_ARTIFACT,
                "transient storage error while opening the run manifest",
                epoch=0,
            )

        memory, util = self._resource_estimate()
        if self.fault.kind == "resource_exhausted" and self.fault.active(self.attempt):
            memory = max(memory, DEVICE_MEMORY_MB * 1.18)
        self._peak_memory = memory
        if memory > DEVICE_MEMORY_MB:
            raise TrainingFailure(
                AnomalyKind.RESOURCE_EXHAUSTED,
                f"out of memory: requested {memory:.0f}MB of {DEVICE_MEMORY_MB:.0f}MB at batch_size={self.batch_size}",
                epoch=0,
            )

        lr = self.learning_rate

        n = len(self.y_tr)
        best_val = -np.inf
        patience_left = self.early_stopping_patience

        for epoch in range(1, self.epochs + 1):
            order = self._rng.permutation(n)
            with np.errstate(over="ignore", invalid="ignore"):
                for start in range(0, n, self.batch_size):
                    idx = order[start : start + self.batch_size]
                    xb, yb = self.x_tr[idx], self.y_tr[idx]
                    self._sgd_step(xb, yb, lr)

            train_p = self.predict_proba(self.x_tr)
            val_p = self.predict_proba(self.x_val)
            train_loss = self._loss(self.x_tr, self.y_tr, train_p)
            val_loss = self._loss(self.x_val, self.y_val, val_p)

            if not np.isfinite(train_loss) or not np.isfinite(val_loss) or not np.all(np.isfinite(self.w1)):
                self.curves.append(
                    EpochRecord(
                        epoch=epoch,
                        train_loss=float("nan"),
                        val_loss=float("nan"),
                        train_metric=0.0,
                        val_metric=0.0,
                        seconds=0.02,
                        gpu_util_pct=util,
                        memory_mb=memory,
                    )
                )
                raise TrainingFailure(
                    AnomalyKind.NAN_LOSS,
                    f"loss became non-finite at epoch {epoch} (learning_rate={lr})",
                    epoch=epoch,
                )

            train_metric = float(compute_all(self.y_tr, train_p)["macro_f1"])
            val_metric = float(compute_all(self.y_val, val_p)["macro_f1"])
            test_metric = float(compute_all(self.data.y_test, self.predict_proba(self.data.x_test))["macro_f1"])

            self._checkpoints.append(self._snapshot())
            self._val_metric_history.append(val_metric)
            self._test_metric_history.append(test_metric)
            self._gpu_utils.append(util)

            record = EpochRecord(
                epoch=epoch,
                train_loss=round(train_loss, 5),
                val_loss=round(val_loss, 5),
                train_metric=round(train_metric, 5),
                val_metric=round(val_metric, 5),
                seconds=round(0.018 + self.batch_size / 40000.0, 4),
                gpu_util_pct=round(util, 1),
                memory_mb=round(memory, 1),
            )
            self.curves.append(record)
            yield record

            if self.early_stopping_patience is not None:
                if val_metric > best_val + 1e-6:
                    best_val = val_metric
                    patience_left = self.early_stopping_patience
                else:
                    patience_left = (patience_left or 0) - 1
                    if patience_left <= 0:
                        self.early_stopped = True
                        self.notes.append(
                            f"early stopping fired at epoch {epoch}: no validation "
                            f"improvement for {self.early_stopping_patience} epochs"
                        )
                        return

    def _loss(self, x: np.ndarray, y: np.ndarray, p: np.ndarray) -> float:
        weights = self._sample_weights(y)
        if str(self.config.objective).lower() == "mse_logit":
            z, _ = self._logits(x)
            with np.errstate(over="ignore", invalid="ignore"):
                terms = (z - (2.0 * y - 1.0)) ** 2
                return float(np.sum(terms * weights) / np.sum(weights))
        return _bce(p, y, weights)

    def _sgd_step(self, xb: np.ndarray, yb: np.ndarray, lr: float) -> None:
        m = max(1, len(yb))
        sw = self._sample_weights(yb)
        mse = str(self.config.objective).lower() == "mse_logit"
        if int(self.config.hidden_units) > 0:
            hidden = np.tanh(xb @ self.w1 + self.b1)
            z = hidden @ self.w2 + self.b2
            # Squared error on the logit has an unbounded gradient, so it can
            # and does diverge above the critical learning rate.
            residual = (z - (2.0 * yb - 1.0)) if mse else (_sigmoid(z) - yb)
            dz = sw * residual / m
            grad_w2 = hidden.T @ dz
            grad_b2 = float(np.sum(dz))
            dh = np.outer(dz, self.w2) * (1 - hidden**2)
            grad_w1 = xb.T @ dh
            grad_b1 = np.sum(dh, axis=0)
            self.w2 -= lr * grad_w2
            self.b2 -= lr * grad_b2
            self.w1 -= lr * grad_w1
            self.b1 -= lr * grad_b1
        else:
            z = xb @ self.w1 + self.b1
            residual = (z - (2.0 * yb - 1.0)) if mse else (_sigmoid(z) - yb)
            dz = sw * residual / m
            self.w1 -= lr * (xb.T @ dz)
            self.b1 -= lr * float(np.sum(dz))

    # -- results ----------------------------------------------------------

    def finalize(self, threshold: float = 0.5) -> TrainResult:
        """Select a checkpoint and score it on the held-out test split."""
        if not self._checkpoints:
            raise TrainingFailure(AnomalyKind.MISSING_ARTIFACT, "no checkpoint was written")

        best_val_epoch = int(np.argmax(self._val_metric_history)) + 1
        best_test_epoch = int(np.argmax(self._test_metric_history)) + 1
        if self.checkpoint_selection == "test":
            # The loophole under test: picking the checkpoint that happens to
            # look best on the *test* split.
            selected = best_test_epoch
        elif self.checkpoint_selection == "last":
            selected = len(self._checkpoints)
        else:
            selected = best_val_epoch

        final_snapshot = self._snapshot()
        self._restore(self._checkpoints[selected - 1])
        test_scores = self.predict_proba(self.data.x_test)
        metrics = compute_all(self.data.y_test, test_scores, threshold)

        self._restore(final_snapshot)
        final_metrics = compute_all(self.data.y_test, self.predict_proba(self.data.x_test), threshold)

        return TrainResult(
            config={
                "name": self.config.name,
                "family": self.config.family,
                "hidden_units": int(self.config.hidden_units),
                "epochs_configured": self.epochs,
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "class_weight": str(self.config.class_weight),
                "objective": str(self.config.objective),
                "checkpoint_selection": self.checkpoint_selection,
            },
            seed=self.seed,
            curves=list(self.curves),
            metrics=metrics,
            final_epoch_metrics=final_metrics,
            test_scores=test_scores,
            y_test=self.data.y_test,
            selected_epoch=selected,
            best_val_epoch=best_val_epoch,
            best_test_epoch=best_test_epoch,
            epochs_run=len(self._checkpoints),
            early_stopped=self.early_stopped,
            peak_memory_mb=self._peak_memory,
            mean_gpu_util_pct=float(np.mean(self._gpu_utils)) if self._gpu_utils else 0.0,
            notes=list(self.notes),
        )


def run_to_completion(trainer: Trainer, threshold: float = 0.5) -> TrainResult:
    """Convenience helper for synchronous callers (tests, seed sweeps)."""
    for _ in trainer.epochs_iter():
        pass
    return trainer.finalize(threshold)
