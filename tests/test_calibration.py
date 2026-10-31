from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from intervallum_ml.calibration import ItemCalibrator, detect_bad_items
from intervallum_ml.coldstart import MetaDifficultyModel
from intervallum_ml.config import CalibrationConfig
from intervallum_ml.simulation import build_world, simulate
from intervallum_ml.types import Attempt, Rating, StateSnapshot, Task, TaskMechanic


def test_calibration_recovers_synthetic_parameters():
    world = build_world(n_users=12, n_kus=8, n_tasks=20, seed=3)
    sim = simulate(world, attempts_per_user=15, seed=4)
    meta = MetaDifficultyModel().fit(list(world.tasks.values()))
    report = ItemCalibrator(meta=meta).fit(
        world.tasks,
        sim.attempts,
        tau0=2.0,
        gamma0=0.8,
    )
    assert report.n_attempts > 100
    assert report.n_items > 5
    assert report.brier < 0.35
    assert 1.0 <= report.tau <= 4.0
    assert 0.0 <= report.gamma <= 2.0
    assert report.converged


def test_bad_item_detection_flags_negative_discrimination():
    task = Task(1, TaskMechanic.MCQ_SINGLE, (1,), t_expected_sec=30)
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    attempts = []
    for i in range(20):
        weak = i < 10
        theta = -1.0 if weak else 1.0
        # Weak students succeed, strong students fail: a broken key.
        y = 1.0 if weak else 0.0
        snap = StateSnapshot(theta=theta, sigma=0.5, stability=10, difficulty=5, retrievability=0.9)
        attempts.append(
            Attempt(
                user_id=f"u{i}",
                task_id=1,
                session_id="s",
                timestamp=now,
                raw_score=y,
                duration_ms=30000,
                self_grade=Rating.GOOD if y else Rating.AGAIN,
                state_before={1: snap},
            )
        )
    health = detect_bad_items({1: task}, attempts, min_attempts=10)
    assert len(health) == 1
    assert health[0].point_biserial < 0
    assert health[0].quarantine


def test_calibration_recovers_controlled_gamma_and_tau():
    rng = np.random.default_rng(21)
    true_tau = 2.0
    true_gamma = 1.0
    tasks = {}
    true_b = {}
    true_a = {}
    for i in range(1, 13):
        true_b[i] = float(rng.normal(0, 0.8))
        true_a[i] = float(rng.uniform(0.8, 1.8))
        tasks[i] = Task(
            i,
            TaskMechanic.NUMERIC,
            (1, 2),
            b_difficulty=true_b[i],
            a_discrimination=true_a[i],
            t_expected_sec=60,
        )
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    attempts = []
    for j in range(3000):
        tid = int(rng.integers(1, 13))
        theta1 = float(rng.normal(0, 1))
        theta2 = float(rng.normal(0, 1))
        r1 = float(rng.choice([1.0, 0.85, 0.65, 0.45]))
        r2 = float(rng.choice([1.0, 0.85, 0.65, 0.45]))
        x = [
            theta1 + true_gamma * np.log(r1),
            theta2 + true_gamma * np.log(r2),
        ]
        theta_task = float(-np.log(np.mean(np.exp(-true_tau * np.asarray(x)))) / true_tau)
        p = 1.0 / (1.0 + np.exp(-true_a[tid] * (theta_task - true_b[tid])))
        y = float(rng.random() < p)
        attempts.append(
            Attempt(
                user_id=f"u{j}",
                task_id=tid,
                session_id="s",
                timestamp=now,
                raw_score=y,
                duration_ms=30000,
                self_grade=Rating.GOOD if y else Rating.AGAIN,
                state_before={
                    1: StateSnapshot(theta1, 0.5, 10, 5, r1),
                    2: StateSnapshot(theta2, 0.5, 10, 5, r2),
                },
            )
        )
    report = ItemCalibrator(
        CalibrationConfig(min_attempts_for_tau=1000, min_attempts_for_gamma=1000)
    ).fit(tasks, attempts, tau0=1.5, gamma0=0.5)
    assert abs(report.gamma - true_gamma) < 0.25, report.gamma
    assert abs(report.tau - true_tau) < 0.35, report.tau
    assert report.brier < 0.30


def test_calibration_gradient_matches_finite_differences(monkeypatch):
    from types import SimpleNamespace

    from scipy.optimize import check_grad

    import intervallum_ml.calibration as calibration_module
    from intervallum_ml.config import CalibrationConfig

    world = build_world(n_users=6, n_kus=6, n_tasks=8, seed=31)
    sim = simulate(world, attempts_per_user=8, seed=32)
    captured = {}

    def fake_minimize(fun, x0, **kwargs):
        captured["fun"] = fun
        captured["x0"] = x0
        return SimpleNamespace(x=x0, fun=fun(x0)[0], success=True)

    monkeypatch.setattr(calibration_module, "minimize", fake_minimize)
    cal = ItemCalibrator(
        CalibrationConfig(max_iter=1),
        meta=MetaDifficultyModel().fit(list(world.tasks.values())),
    )
    cal.fit(world.tasks, sim.attempts, tau0=2.0, gamma0=0.8)
    fun = captured["fun"]
    x0 = captured["x0"]
    err = check_grad(
        lambda p: fun(p)[0],
        lambda p: fun(p)[1],
        x0,
    )
    assert err < 1e-3, err
