from __future__ import annotations

from datetime import datetime, timedelta, timezone

from intervallum_ml.evaluation import (
    doubly_robust,
    effective_sample_size,
    evaluate_predictions,
    inverse_propensity_score,
    temporal_user_split,
)
from intervallum_ml.types import Attempt, Rating


def attempt(user: str, day: int) -> Attempt:
    return Attempt(
        user_id=user,
        task_id=1,
        session_id="s",
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=day),
        raw_score=1.0,
        duration_ms=30000,
        self_grade=Rating.GOOD,
    )


def test_temporal_split_never_puts_future_of_holdout_user_in_train():
    rows = [attempt("a", i) for i in range(10)] + [attempt("b", i) for i in range(10)]
    train, test = temporal_user_split(rows, holdout_fraction=0.5, history_fraction=0.7, seed=1)
    test_users = {a.user_id for a in test}
    assert test_users
    for user in test_users:
        train_times = [a.timestamp for a in train if a.user_id == user]
        test_times = [a.timestamp for a in test if a.user_id == user]
        if train_times and test_times:
            assert max(train_times) <= min(test_times)


def test_metrics_are_perfect_for_perfect_predictions():
    m = evaluate_predictions([0, 1, 1, 0], [0, 1, 1, 0])
    assert m.brier == 0
    assert m.log_loss < 1e-6
    assert m.auc == 1.0


def test_ips_and_dr():
    rewards = [1.0, 0.0, 1.0]
    behavior = [0.5, 0.5, 0.5]
    target = [1.0, 1.0, 1.0]
    ips = inverse_propensity_score(rewards, behavior, target)
    assert abs(ips - 4.0 / 3.0) < 1e-9
    dr = doubly_robust(rewards, behavior, [0.5, 0.5, 0.5], target)
    assert isinstance(dr, float)
    assert effective_sample_size([1, 1, 1]) == 3.0
