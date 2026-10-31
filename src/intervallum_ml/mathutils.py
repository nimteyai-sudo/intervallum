"""Small, dependency-light numerical helpers.

The online path in production is intended to be ported to TypeScript, so the
Python implementations below favour clarity and numerical stability over
micro-optimisation.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def clamp(value: float, low: float, high: float) -> float:
    if math.isnan(value):
        return low
    return max(low, min(high, value))


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def logit(p: float, eps: float = 1e-9) -> float:
    p = clamp(p, eps, 1.0 - eps)
    return math.log(p / (1.0 - p))


def softmin(values: Sequence[float], tau: float = 2.0, weights: Sequence[float] | None = None) -> float:
    """Smooth minimum used for conjunctive task requirements.

    ``tau -> 0`` approaches the weighted arithmetic mean; ``tau -> inf``
    approaches the hard minimum.  The implementation is the log-sum-exp form
    of ``-log(sum(w * exp(-tau * value))) / tau``.
    """

    if not values:
        return 0.0
    tau = max(float(tau), 1e-6)
    arr = np.asarray(values, dtype=float)
    if weights is None:
        w = np.full(arr.shape, 1.0 / arr.size, dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        total = float(np.sum(w))
        if total <= 0 or w.shape != arr.shape:
            w = np.full(arr.shape, 1.0 / arr.size, dtype=float)
        else:
            w = w / total
    # Stable log-sum-exp with weights.
    m = float(np.max(-tau * arr))
    log_mean = m + math.log(max(float(np.sum(w * np.exp(-tau * arr - m))), 1e-300))
    return -log_mean / tau


def softmax(values: Sequence[float], temperature: float = 1.0) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    temperature = max(float(temperature), 1e-6)
    z = arr / temperature
    z = z - float(np.max(z))
    exp = np.exp(z)
    total = float(np.sum(exp))
    if total <= 0 or not math.isfinite(total):
        return np.full(arr.shape, 1.0 / arr.size)
    return exp / total


def gumbel_noise(rng: np.random.Generator, size: int = 1) -> np.ndarray:
    u = rng.random(size)
    u = np.clip(u, 1e-12, 1.0 - 1e-12)
    return -np.log(-np.log(u))


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    return numerator / denominator if denominator else default


def brier_score(y_true: Sequence[float], p: Sequence[float]) -> float:
    y = np.asarray(y_true, dtype=float)
    pred = np.asarray(p, dtype=float)
    if y.size == 0:
        return 0.0
    return float(np.mean((pred - y) ** 2))


def log_loss(y_true: Sequence[float], p: Sequence[float], eps: float = 1e-9) -> float:
    y = np.asarray(y_true, dtype=float)
    pred = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    if y.size == 0:
        return 0.0
    return float(-np.mean(y * np.log(pred) + (1.0 - y) * np.log(1.0 - pred)))


def expected_calibration_error(
    y_true: Sequence[float], p: Sequence[float], n_bins: int = 10
) -> float:
    y = np.asarray(y_true, dtype=float)
    pred = np.asarray(p, dtype=float)
    if y.size == 0:
        return 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    # Rightmost bin is inclusive.
    idx = np.clip(np.digitize(pred, bins[1:-1], right=True), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not np.any(mask):
            continue
        total += float(np.mean(mask)) * abs(float(np.mean(y[mask])) - float(np.mean(pred[mask])))
    return float(total)


def roc_auc(y_true: Sequence[float], scores: Sequence[float]) -> float:
    """Rank-based AUC with tie handling (Mann-Whitney U)."""

    y = np.asarray(y_true, dtype=float)
    s = np.asarray(scores, dtype=float)
    if y.size == 0:
        return 0.5
    positive = y == 1
    negative = y == 0
    n_pos = int(np.sum(positive))
    n_neg = int(np.sum(negative))
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and sorted_s[j] == sorted_s[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    sum_pos_ranks = float(np.sum(ranks[positive]))
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def point_biserial(item_scores: Sequence[float], total_scores: Sequence[float]) -> float:
    x = np.asarray(item_scores, dtype=float)
    y = np.asarray(total_scores, dtype=float)
    if x.size < 3 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def percentile_interval(values: Sequence[float], low: float = 0.10, high: float = 0.90) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (0.0, 0.0)
    return (float(np.quantile(arr, low)), float(np.quantile(arr, high)))
