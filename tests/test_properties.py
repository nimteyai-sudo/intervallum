from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from intervallum_ml.calibration import _softmin_derivative
from intervallum_ml.config import EngineConfig
from intervallum_ml.engine import IntervallumEngine
from intervallum_ml.irt import MasteryEngine
from intervallum_ml.mathutils import softmin
from intervallum_ml.telemetry import scrub_attempt
from intervallum_ml.types import (
    Attempt,
    KnowledgeUnit,
    KUType,
    Rating,
    SelectionContext,
    Task,
    TaskMechanic,
    UserKUState,
)

finite_float = st.floats(
    min_value=-4.0, max_value=4.0, allow_nan=False, allow_infinity=False
)
prob = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


@st.composite
def task_strategy(draw):
    return Task(
        id=draw(st.integers(1, 10_000)),
        mechanic=draw(st.sampled_from(list(TaskMechanic))),
        ku_ids=(1,),
        c_guess=draw(st.floats(0.0, 0.5)),
        t_expected_sec=draw(st.floats(20.0, 300.0)),
        b_difficulty=draw(st.floats(-3.0, 3.0)),
        a_discrimination=draw(st.floats(0.5, 2.5)),
    )


@given(
    raw=prob,
    duration=st.integers(0, 600_000),
    self_grade=st.sampled_from(list(Rating)),
    task=task_strategy(),
    hints=st.floats(0.0, 1.5),
)
@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_telemetry_invariants(raw, duration, self_grade, task, hints):
    result = scrub_attempt(
        raw_score=raw,
        duration_ms=duration,
        self_grade=self_grade,
        task=task,
        hint_weight_sum=hints,
    )
    assert 0.0 <= result.y_eff <= 1.0
    assert 1 <= int(result.g_sync) <= 4
    assert float(result.g_sync) <= float(result.objective_grade)
    assert 0.0 <= result.hint_multiplier <= 1.0
    assert 0.0 <= result.time_penalty <= 1.0


@given(
    values=st.lists(finite_float, min_size=1, max_size=5),
    weights=st.lists(st.floats(0.01, 5.0), min_size=1, max_size=5),
    tau=st.floats(0.1, 5.0),
)
@settings(max_examples=200, deadline=None)
def test_softmin_bounds(values, weights, tau):
    n = min(len(values), len(weights))
    values = values[:n]
    weights = weights[:n]
    sm = softmin(values, tau, weights)
    assert min(values) - 1e-9 <= sm <= max(values) + 1e-9


@given(
    theta1=finite_float,
    theta2=finite_float,
    sigma1=st.floats(0.25, 1.5),
    sigma2=st.floats(0.25, 1.5),
    b=st.floats(-3.0, 3.0),
    a=st.floats(0.5, 2.5),
    c=st.floats(0.0, 0.5),
    y=prob,
)
@settings(max_examples=200, deadline=None)
def test_irt_invariants(theta1, theta2, sigma1, sigma2, b, a, c, y):
    engine = MasteryEngine()
    states = {
        1: UserKUState(1, theta=theta1, sigma=sigma1),
        2: UserKUState(2, theta=theta2, sigma=sigma2),
    }
    task = Task(1, TaskMechanic.NUMERIC, (1, 2), b_difficulty=b, a_discrimination=a, c_guess=c)
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    pred = engine.predict(states, task, now)
    assert 0.0 <= pred.p_star <= 1.0
    assert 0.0 <= pred.p_correct <= 1.0
    assert math.isfinite(pred.theta_task)
    assert abs(sum(pred.responsibilities.values()) - 1.0) < 1e-6
    engine.update(states, task, pred, y, now)
    for state in states.values():
        assert math.isfinite(state.theta)
        assert 0.25 - 1e-9 <= state.sigma <= 1.5 + 1e-9
        assert state.attempts_count == 1


def make_engine() -> IntervallumEngine:
    kus = {
        i: KnowledgeUnit(i, f"ku{i}", f"KU {i}", KUType.HYBRID, 1, 1)
        for i in range(1, 6)
    }
    tasks = {
        i: Task(
            i,
            TaskMechanic.MCQ_SINGLE,
            (1 + (i % 5),),
            c_guess=0.25,
            b_difficulty=((i % 5) - 2) * 0.5,
            a_discrimination=1.0 + (i % 3) * 0.2,
        )
        for i in range(1, 21)
    }
    return IntervallumEngine(kus, tasks, EngineConfig())


@given(
    thetas=st.lists(finite_float, min_size=5, max_size=5),
    sigmas=st.lists(st.floats(0.25, 1.5), min_size=5, max_size=5),
    session_position=st.integers(0, 10),
    review_count=st.integers(0, 20),
    days_to_exam=st.integers(0, 365),
    seed=st.integers(0, 10_000),
)
@settings(max_examples=100, deadline=None)
def test_selector_invariants(thetas, sigmas, session_position, review_count, days_to_exam, seed):
    engine = make_engine()
    states = engine.new_user_state()
    for i, (theta, sigma) in enumerate(zip(thetas, sigmas), start=1):
        states[i].theta = theta
        states[i].sigma = sigma
    ctx = SelectionContext(
        session_position=session_position,
        session_length=10,
        review_count=review_count,
        days_to_exam=days_to_exam,
        seed=seed,
    )
    result = engine.select(states, ctx)
    assert result.task.id in engine.tasks
    assert 0.0 < result.propensity <= 1.0
    assert not result.task.is_quarantined
    assert 0.0 <= result.score.p_correct <= 1.0
    assert math.isfinite(result.score.utility)


@given(
    theta=finite_float,
    sigma=st.floats(0.25, 1.5),
    b=st.floats(-3.0, 3.0),
    a=st.floats(0.5, 2.5),
    y=prob,
)
@settings(max_examples=100, deadline=None)
def test_memory_update_is_finite(theta, sigma, b, a, y):
    engine = make_engine()
    states = engine.new_user_state()
    states[1].theta = theta
    states[1].sigma = sigma
    task = Task(1, TaskMechanic.NUMERIC, (1,), b_difficulty=b, a_discrimination=a)
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    attempt = Attempt(
        user_id="u",
        task_id=1,
        session_id="s",
        timestamp=now,
        raw_score=y,
        duration_ms=30_000,
        self_grade=Rating.GOOD,
        state_before=engine.snapshot(states, now),
    )
    result = engine.review_attempt(states, task, attempt)
    assert math.isfinite(result.telemetry.y_eff)
    assert states[1].memory is not None
    assert math.isfinite(states[1].memory.stability)


@given(
    x=st.lists(finite_float, min_size=2, max_size=4),
    w=st.lists(st.floats(0.1, 5.0), min_size=2, max_size=4),
    tau=st.floats(0.5, 4.0),
)
@settings(max_examples=100, deadline=None)
def test_softmin_analytic_derivative(x, w, tau):
    n = min(len(x), len(w))
    x_arr = list(x[:n])
    w_arr = list(w[:n])
    theta, q, d_tau = _softmin_derivative(
        np.asarray(x_arr),
        tau,
        np.asarray(w_arr),
    )
    eps = 1e-6
    theta_plus = softmin(x_arr, tau + eps, w_arr)
    theta_minus = softmin(x_arr, tau - eps, w_arr)
    numeric = (theta_plus - theta_minus) / (2 * eps)
    assert abs(d_tau - numeric) < 1e-4
    assert abs(sum(q) - 1.0) < 1e-6
