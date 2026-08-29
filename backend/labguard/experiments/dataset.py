"""Synthetic dataset generator for the bundled demo scenario.

The generator deliberately builds a dataset with the weaknesses LabGuard is
meant to catch:

* heavy class imbalance (default 8% positive),
* a minority class that is only partly separable, with label noise, so that a
  higher-capacity model trained for longer memorises it and *loses* minority
  recall while gaining raw accuracy,
* optional train/test row duplication, to exercise the overlap detector,
* optional covariate shift on the test split, for the domain-shift check.

Everything is seeded, so two runs with the same seed are bit-identical.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..models.domain import DatasetInfo


@dataclass(frozen=True)
class Dataset:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    #: Row hashes, used by the train/test overlap check.
    train_hashes: list[str]
    test_hashes: list[str]
    #: Number of rows deliberately duplicated across the split.
    injected_overlap: int

    @property
    def n_features(self) -> int:
        return int(self.x_train.shape[1])

    @property
    def positive_rate_train(self) -> float:
        return float(self.y_train.mean())

    @property
    def positive_rate_test(self) -> float:
        return float(self.y_test.mean())


def _row_hashes(x: np.ndarray) -> list[str]:
    """Stable content hash per row, rounded so float noise does not matter."""
    rounded = np.round(x, 6)
    return [hash(row.tobytes()).__and__(0xFFFFFFFFFFFF).__format__("012x") for row in rounded]


def make_dataset(info: DatasetInfo, seed: int) -> Dataset:
    rng = np.random.default_rng(seed)
    n = int(info.n_samples)
    d = int(info.n_features)
    pos_rate = float(info.positive_rate)

    n_pos = max(8, round(n * pos_rate))
    n_neg = n - n_pos

    # Negative class: a single broad Gaussian blob.
    x_neg = rng.normal(0.0, 1.0, size=(n_neg, d))

    # Positive class: shifted along a handful of informative features only,
    # and split into an "easy" and a "hard" sub-population. The hard
    # sub-population sits inside the negative cloud, so extra model capacity
    # buys training-set memorisation rather than test-set recall.
    n_easy = round(n_pos * 0.32)
    n_hard = n_pos - n_easy
    informative = np.zeros(d)
    informative[:4] = np.array([0.95, 0.8, -0.7, 0.6])

    x_easy = rng.normal(0.0, 1.0, size=(n_easy, d)) + informative
    x_hard = rng.normal(0.0, 1.3, size=(n_hard, d)) + informative * 0.12

    x = np.vstack([x_neg, x_easy, x_hard])
    y = np.concatenate([np.zeros(n_neg), np.ones(n_easy + n_hard)]).astype(np.int64)

    # Label noise concentrated near the boundary: a slice of negatives that
    # look positive are labelled positive. Long training memorises these.
    n_noise = round(n_pos * 0.38)
    if n_noise > 0:
        scores = x[:n_neg] @ informative
        flip_idx = np.argsort(scores)[-n_noise:]
        y[flip_idx] = 1

    order = rng.permutation(len(y))
    x, y = x[order], y[order]

    n_test = round(len(y) * float(info.test_fraction))
    x_test, y_test = x[:n_test], y[:n_test]
    x_train, y_train = x[n_test:], y[n_test:]

    # Optional deliberate leakage: copy rows from train into test verbatim.
    injected = int(info.inject_train_test_overlap)
    if injected > 0:
        injected = min(injected, len(y_train), len(y_test))
        x_test = x_test.copy()
        y_test = y_test.copy()
        x_test[:injected] = x_train[:injected]
        y_test[:injected] = y_train[:injected]

    # Optional covariate shift applied to the test split only.
    shift = float(info.domain_shift_strength)
    if shift:
        x_test = x_test + rng.normal(shift, 0.05, size=x_test.shape)

    return Dataset(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        train_hashes=_row_hashes(x_train),
        test_hashes=_row_hashes(x_test),
        injected_overlap=injected,
    )


def split_validation(
    data: Dataset, seed: int, fraction: float = 0.2
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Carve a validation split out of train, for honest checkpoint selection."""
    rng = np.random.default_rng(seed + 9973)
    idx = rng.permutation(len(data.y_train))
    n_val = max(1, round(len(idx) * fraction))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    return (
        data.x_train[tr_idx],
        data.y_train[tr_idx],
        data.x_train[val_idx],
        data.y_train[val_idx],
    )
