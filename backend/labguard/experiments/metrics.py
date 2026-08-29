"""Metric computation.

Every number LabGuard reports comes from this module. The language model is
never asked to produce a metric, a confidence interval or a score.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def classwise(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for label in (0, 1):
        tp = float(np.sum((y_true == label) & (y_pred == label)))
        fp = float(np.sum((y_true != label) & (y_pred == label)))
        fn = float(np.sum((y_true == label) & (y_pred != label)))
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        out[f"class_{label}"] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": int(np.sum(y_true == label)),
        }
    return out


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney U), tie-aware."""
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)
    # Average ranks within tie groups.
    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = float(np.mean(ranks[order[i : j + 1]]))
        i = j + 1
    rank_sum_pos = float(np.sum(ranks[y_true == 1]))
    n_pos, n_neg = float(len(pos)), float(len(neg))
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def compute_all(y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    """The full metric bundle for one prediction set."""
    y_pred = (scores >= threshold).astype(np.int64)
    per_class = classwise(y_true, y_pred)
    recalls = [per_class["class_0"]["recall"], per_class["class_1"]["recall"]]
    f1s = [per_class["class_0"]["f1"], per_class["class_1"]["f1"]]
    return {
        "accuracy": round(float(np.mean(y_true == y_pred)), 4),
        "balanced_accuracy": round(float(np.mean(recalls)), 4),
        "macro_f1": round(float(np.mean(f1s)), 4),
        "minority_recall": per_class["class_1"]["recall"],
        "minority_precision": per_class["class_1"]["precision"],
        "roc_auc": round(roc_auc(y_true, scores), 4),
        "threshold": threshold,
        "confusion": confusion(y_true, y_pred),
        "classwise": per_class,
        "n_samples": len(y_true),
        "positive_rate": round(float(np.mean(y_true)), 4),
    }


def best_threshold_by_macro_f1(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Sweep thresholds and return `(threshold, macro_f1)` at the best point."""
    grid = np.linspace(0.05, 0.95, 19)
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        metrics = compute_all(y_true, scores, float(t))
        if metrics["macro_f1"] > best_f1:
            best_t, best_f1 = float(t), float(metrics["macro_f1"])
    return round(best_t, 3), round(best_f1, 4)


def bootstrap_ci(
    y_true: np.ndarray,
    scores: np.ndarray,
    metric: str,
    seed: int = 0,
    n_boot: int = 200,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Percentile bootstrap CI over the test set for one metric."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    values: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample_y, sample_s = y_true[idx], scores[idx]
        if len(np.unique(sample_y)) < 2:
            continue
        values.append(float(compute_all(sample_y, sample_s, threshold)[metric]))
    if not values:
        return {"low": 0.0, "high": 0.0, "mean": 0.0, "n_boot": 0}
    arr = np.array(values)
    return {
        "low": round(float(np.percentile(arr, 2.5)), 4),
        "high": round(float(np.percentile(arr, 97.5)), 4),
        "mean": round(float(arr.mean()), 4),
        "n_boot": len(values),
    }


def paired_seed_summary(deltas: list[float]) -> dict[str, Any]:
    """Summarise a per-seed paired difference (model B minus model A).

    Reports a normal-approximation CI on the mean difference plus a sign test,
    which is what actually decides whether a per-seed win is stable.
    """
    arr = np.array(deltas, dtype=float)
    n = len(arr)
    mean = float(arr.mean()) if n else 0.0
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    stderr = std / np.sqrt(n) if n > 1 else 0.0
    half_width = 1.96 * stderr
    wins = int(np.sum(arr > 0))
    return {
        "n_seeds": n,
        "mean_delta": round(mean, 4),
        "std_delta": round(std, 4),
        "ci_low": round(mean - half_width, 4),
        "ci_high": round(mean + half_width, 4),
        "ci_includes_zero": bool(mean - half_width <= 0.0 <= mean + half_width) if n > 1 else True,
        "wins_for_b": wins,
        "losses_for_b": int(np.sum(arr < 0)),
        "ties": int(np.sum(arr == 0)),
        "per_seed_deltas": [round(float(v), 4) for v in arr],
    }
