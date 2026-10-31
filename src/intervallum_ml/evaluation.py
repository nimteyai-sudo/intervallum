"""Evaluation utilities with leakage-safe splits and off-policy estimates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .mathutils import (
    brier_score,
    expected_calibration_error,
    log_loss,
    roc_auc,
)
from .types import Attempt


@dataclass(frozen=True)
class PredictionMetrics:
    n: int
    brier: float
    log_loss: float
    ece: float
    auc: float


def evaluate_predictions(y_true: Sequence[float], p: Sequence[float]) -> PredictionMetrics:
    y = list(y_true)
    p = list(p)
    return PredictionMetrics(
        n=len(y),
        brier=brier_score(y, p),
        log_loss=log_loss(y, p),
        ece=expected_calibration_error(y, p),
        auc=roc_auc(y, p),
    )


def temporal_user_split(
    attempts: Sequence[Attempt],
    holdout_fraction: float = 0.2,
    history_fraction: float = 0.7,
    seed: int = 42,
) -> tuple[list[Attempt], list[Attempt]]:
    """Split by user and time.

    A random split over attempts leaks the future of the same user into the
    training set.  Here 20% of users are held out entirely, and for each
    holdout user the first 70% of their history trains the online state while
    the remaining 30% is predicted forward in time.
    """

    by_user: dict[str, list[Attempt]] = {}
    for a in attempts:
        by_user.setdefault(a.user_id, []).append(a)
    users = sorted(by_user)
    rng = np.random.default_rng(seed)
    rng.shuffle(users)
    n_holdout = max(1, int(round(len(users) * holdout_fraction))) if users else 0
    holdout_users = set(users[:n_holdout])
    train: list[Attempt] = []
    test: list[Attempt] = []
    for user, rows in by_user.items():
        rows = sorted(rows, key=lambda a: a.timestamp)
        if user not in holdout_users:
            train.extend(rows)
            continue
        cut = max(1, int(len(rows) * history_fraction))
        train.extend(rows[:cut])
        test.extend(rows[cut:])
    return train, test


def inverse_propensity_score(
    rewards: Sequence[float],
    behavior_propensity: Sequence[float],
    target_propensity: Sequence[float] | None = None,
    clip: float = 10.0,
) -> float:
    """IPS estimate of a new policy's mean reward.

    ``behavior_propensity`` is the probability with which the logged policy
    picked the observed action.  ``target_propensity`` is the new policy's
    probability for that same action; if omitted, the target is treated as
    deterministic (1.0).
    """

    r = np.asarray(rewards, dtype=float)
    b = np.clip(np.asarray(behavior_propensity, dtype=float), 1e-6, 1.0)
    t = (
        np.ones_like(b)
        if target_propensity is None
        else np.asarray(target_propensity, dtype=float)
    )
    weights = np.clip(t / b, 0.0, clip)
    if weights.size == 0:
        return 0.0
    return float(np.mean(r * weights))


def effective_sample_size(weights: Sequence[float]) -> float:
    w = np.asarray(weights, dtype=float)
    if w.size == 0:
        return 0.0
    denom = float(np.sum(w ** 2))
    return float((np.sum(w) ** 2) / denom) if denom > 0 else 0.0


def doubly_robust(
    rewards: Sequence[float],
    behavior_propensity: Sequence[float],
    reward_model: Sequence[float],
    target_propensity: Sequence[float] | None = None,
    clip: float = 10.0,
) -> float:
    """Doubly robust off-policy estimate (unbiased if either model is right)."""

    r = np.asarray(rewards, dtype=float)
    b = np.clip(np.asarray(behavior_propensity, dtype=float), 1e-6, 1.0)
    m = np.asarray(reward_model, dtype=float)
    t = (
        np.ones_like(b)
        if target_propensity is None
        else np.asarray(target_propensity, dtype=float)
    )
    weights = np.clip(t / b, 0.0, clip)
    if weights.size == 0:
        return 0.0
    return float(np.mean(m + weights * (r - m)))


def bootstrap_mean_ci(
    values: Sequence[float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (0.0, 0.0, 0.0)
    rng = np.random.default_rng(seed)
    samples = rng.choice(arr, size=(n_boot, arr.size), replace=True)
    means = np.mean(samples, axis=1)
    return (
        float(np.mean(arr)),
        float(np.quantile(means, alpha / 2)),
        float(np.quantile(means, 1 - alpha / 2)),
    )
