"""Objective telemetry scrubbing.

The student's self-rating is useful but gameable, especially in a school
context.  This module turns one raw attempt into:

* ``y_eff`` — a continuous 0..1 effectiveness score used by the mastery model;
* ``g_sync`` — a synthetic FSRS grade that is mostly objective.

The rules are intentionally conservative: a correct answer obtained through
hints, or an answer that contradicts the student's own self-rating, cannot
silently inflate the memory state.
"""

from __future__ import annotations

from .config import TelemetryConfig
from .mathutils import clamp
from .types import Rating, Task, TelemetryResult


def _time_penalty(duration_sec: float, expected_sec: float, cfg: TelemetryConfig) -> float:
    ratio = duration_sec / max(expected_sec, 1e-9)
    if ratio <= cfg.time_penalty_start:
        return 1.0
    if ratio >= cfg.slow_answer_ratio:
        return cfg.time_penalty_floor
    span = cfg.slow_answer_ratio - cfg.time_penalty_start
    progress = (ratio - cfg.time_penalty_start) / span
    return 1.0 - (1.0 - cfg.time_penalty_floor) * progress


def objective_grade(
    *,
    raw_score: float,
    duration_sec: float,
    expected_sec: float,
    hint_weight_sum: float,
    c_guess: float,
    cfg: TelemetryConfig,
) -> Rating:
    ratio = duration_sec / max(expected_sec, 1e-9)
    hints = max(0.0, hint_weight_sum)

    if raw_score < 0.5 or hints > cfg.hard_hint_sum:
        return Rating.AGAIN

    # A fast correct answer with a non-trivial guessing probability is not
    # evidence of durable knowledge.
    if raw_score >= 0.99 and ratio < cfg.fast_answer_ratio and c_guess >= 0.15:
        return Rating.HARD

    if raw_score >= 0.8 and hints <= 0.0 and ratio > cfg.slow_but_correct_ratio:
        return Rating.HARD

    if raw_score >= 0.99 and hints <= 0.0 and ratio <= cfg.slow_but_correct_ratio:
        if ratio < cfg.confident_fast_ratio:
            return Rating.EASY
        return Rating.GOOD

    if raw_score >= 0.8:
        return Rating.GOOD if hints <= 0.0 else Rating.HARD
    if raw_score >= 0.5:
        return Rating.HARD
    return Rating.AGAIN


def blend_grade(objective: Rating, self_grade: Rating, cfg: TelemetryConfig) -> Rating:
    # Contradictions are resolved in the conservative direction.
    if objective == Rating.AGAIN or self_grade == Rating.AGAIN:
        return Rating.AGAIN
    blended = (
        cfg.objective_weight * float(objective)
        + cfg.self_weight * float(self_grade)
    )
    # Round half up to keep the mapping monotone and easy to explain.
    rounded = int(blended + 0.5)
    # The objective grade is an upper bound: a slow, hinted or guessed answer
    # cannot become "Good" merely because the student tapped "Easy".
    return Rating(max(1, min(int(objective), rounded)))


def scrub_attempt(
    *,
    raw_score: float,
    duration_ms: int,
    self_grade: Rating,
    task: Task,
    hint_weight_sum: float = 0.0,
    cfg: TelemetryConfig | None = None,
) -> TelemetryResult:
    cfg = cfg or TelemetryConfig()
    duration_sec = max(0.0, duration_ms / 1000.0)
    expected = task.t_expected_sec
    hint_multiplier = clamp(1.0 - max(0.0, hint_weight_sum), cfg.hint_multiplier_floor, 1.0)
    penalty = _time_penalty(duration_sec, expected, cfg)
    y_eff = clamp(raw_score * hint_multiplier * penalty, 0.0, 1.0)

    ratio = duration_sec / max(expected, 1e-9)
    # A very fast "correct" answer on a multiple-choice item is mostly luck.
    if raw_score >= 0.99 and ratio < cfg.fast_answer_ratio and task.c_guess >= 0.15:
        guess_ceiling = 0.5 + 0.5 * task.c_guess
        y_eff = min(y_eff, guess_ceiling)

    obj = objective_grade(
        raw_score=raw_score,
        duration_sec=duration_sec,
        expected_sec=expected,
        hint_weight_sum=hint_weight_sum,
        c_guess=task.c_guess,
        cfg=cfg,
    )
    g_sync = blend_grade(obj, self_grade, cfg)
    return TelemetryResult(
        y_eff=y_eff,
        g_sync=g_sync,
        hint_multiplier=hint_multiplier,
        time_penalty=penalty,
        objective_grade=obj,
        self_grade=self_grade,
    )
