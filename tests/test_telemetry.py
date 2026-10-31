from __future__ import annotations

from intervallum_ml.telemetry import scrub_attempt
from intervallum_ml.types import Rating, Task, TaskMechanic


def task(c_guess: float = 0.0, expected: float = 60.0) -> Task:
    return Task(
        id=1,
        mechanic=TaskMechanic.NUMERIC,
        ku_ids=(1,),
        c_guess=c_guess,
        t_expected_sec=expected,
    )


def test_correct_without_hints_is_good_or_easy():
    t = task()
    r = scrub_attempt(
        raw_score=1.0,
        duration_ms=30_000,
        self_grade=Rating.GOOD,
        task=t,
    )
    assert r.y_eff == 1.0
    assert r.g_sync in (Rating.GOOD, Rating.EASY)
    assert r.time_penalty == 1.0


def test_wrong_answer_cannot_be_easy():
    t = task()
    r = scrub_attempt(
        raw_score=0.0,
        duration_ms=10_000,
        self_grade=Rating.EASY,
        task=t,
    )
    assert r.g_sync == Rating.AGAIN
    assert r.y_eff == 0.0


def test_hints_suppress_memory_grade():
    t = task()
    r = scrub_attempt(
        raw_score=1.0,
        duration_ms=30_000,
        self_grade=Rating.EASY,
        task=t,
        hint_weight_sum=0.6,
    )
    assert r.g_sync == Rating.AGAIN
    assert r.hint_multiplier == 0.4


def test_slow_correct_answer_is_hard():
    t = task()
    r = scrub_attempt(
        raw_score=1.0,
        duration_ms=150_000,
        self_grade=Rating.EASY,
        task=t,
    )
    assert r.objective_grade == Rating.HARD
    assert r.g_sync == Rating.HARD
    assert r.time_penalty < 1.0


def test_fast_correct_mcq_is_not_full_credit():
    t = task(c_guess=0.25)
    r = scrub_attempt(
        raw_score=1.0,
        duration_ms=5_000,
        self_grade=Rating.EASY,
        task=t,
    )
    assert r.y_eff < 0.8
    assert r.g_sync != Rating.EASY


def test_telemetry_is_bounded():
    t = task()
    for score in (0.0, 0.5, 1.0):
        for duration in (1_000, 60_000, 900_000):
            for hints in (0.0, 0.5, 1.5):
                r = scrub_attempt(
                    raw_score=score,
                    duration_ms=duration,
                    self_grade=Rating.GOOD,
                    task=t,
                    hint_weight_sum=hints,
                )
                assert 0.0 <= r.y_eff <= 1.0
