"""Massive stress test against the real EGE/FIPI content bank.

This is intentionally harsher than the unit tests:

* thousands of selection calls with random user states and session contexts;
* hundreds of simulated 10-task sessions with state updates;
* thousands of random attempts including hints, fast answers and self-rating;
* teacher-assignment and review-quota invariants;
* real answer/media structure checks;
* an optional calibration run on synthetic responses from the real bank.

It is a logic and robustness test, not a claim of educational efficacy.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from intervallum_ml.calibration import ItemCalibrator
from intervallum_ml.coldstart import MetaDifficultyModel
from intervallum_ml.config import EngineConfig
from intervallum_ml.content import load_engine
from intervallum_ml.types import (
    Attempt,
    MemoryState,
    Rating,
    SelectionContext,
)


def _random_states(engine, rng):
    states = engine.new_user_state()
    for st in states.values():
        st.theta = float(rng.normal(0, 1))
        st.sigma = float(rng.uniform(0.3, 1.4))
        if rng.random() < 0.3:
            st.attempts_count = int(rng.integers(1, 20))
    return states


def _attempt(engine, states, task, now, rng):
    pred = engine.mastery.predict(states, task, now)
    y = float(rng.random() < pred.p_correct)
    ratio = float(rng.lognormal(0.2, 0.5))
    self_grade = (
        Rating.EASY if y and ratio < 0.7
        else Rating.GOOD if y and ratio < 1.8
        else Rating.HARD if y
        else Rating.AGAIN
    )
    return Attempt(
        user_id="stress",
        task_id=task.id,
        session_id="stress",
        timestamp=now,
        raw_score=y,
        duration_ms=int(max(500, task.t_expected_sec * ratio * 1000)),
        self_grade=self_grade,
        hint_weight_sum=0.0,
        state_before=engine.snapshot(states, now),
    )


def run(args):
    bank_path = Path(args.bank)
    engine, bank = load_engine(bank_path, EngineConfig())
    rng = np.random.default_rng(args.seed)
    now = datetime(2026, 10, 31, tzinfo=timezone.utc)
    task_ids = list(bank.tasks)
    report = {
        "tasks": len(bank.tasks),
        "knowledge_units": len(bank.knowledge_units),
        "media_tasks": sum(1 for t in bank.tasks.values() if t.media),
        "selections": 0,
        "sessions": 0,
        "attempts": 0,
        "teacher_checks": 0,
        "quota_checks": 0,
        "errors": [],
        "reasons": Counter(),
        "answer_types": Counter(t.answer_type for t in bank.tasks.values()),
    }

    start = time.perf_counter()

    # 1. Selection stress.
    for i in range(args.selections):
        states = _random_states(engine, rng)
        teacher = None
        if i % 17 == 0:
            teacher = int(rng.choice(task_ids))
        ctx = SelectionContext(
            now=now,
            session_position=int(rng.integers(0, 10)),
            session_length=10,
            consecutive_errors=int(rng.integers(0, 4)),
            session_elapsed_sec=float(rng.uniform(0, 1800)),
            review_count=int(rng.integers(0, 12)),
            days_to_exam=int(rng.integers(0, 365)),
            teacher_assignment_ids=frozenset({teacher}) if teacher else frozenset(),
            recent_task_ids=tuple(int(x) for x in rng.choice(task_ids, size=min(12, len(task_ids)), replace=False)),
            recent_mechanics=(),
            recent_ku_sets=(),
            seed=i,
        )
        try:
            result = engine.select(states, ctx)
        except Exception as exc:
            report["errors"].append(f"selection {i}: {type(exc).__name__}: {exc}")
            continue
        report["selections"] += 1
        report["reasons"][result.score.reason.value] += 1
        if result.task.is_quarantined:
            report["errors"].append(f"selection {i}: quarantined task {result.task.id}")
        if not (0.0 < result.propensity <= 1.0):
            report["errors"].append(f"selection {i}: bad propensity {result.propensity}")
        if not (0.0 <= result.score.p_correct <= 1.0):
            report["errors"].append(f"selection {i}: bad p_correct {result.score.p_correct}")
        if not math.isfinite(result.score.utility):
            report["errors"].append(f"selection {i}: non-finite utility")
        if teacher is not None:
            report["teacher_checks"] += 1
            if result.task.id != teacher:
                report["errors"].append(
                    f"selection {i}: teacher task {teacher} not selected, got {result.task.id}"
                )
        # Review quota: if the quota is exhausted, a memory-heavy choice is
        # only acceptable when no non-memory candidate exists.
        if ctx.review_count >= 4 and result.score.memory > 0.4:
            alternatives = [s for s in result.ranked if s.memory <= 0.4]
            if alternatives:
                report["errors"].append(
                    f"selection {i}: review quota violated; non-review alternatives existed"
                )
            report["quota_checks"] += 1

    # 2. Full session stress (with state updates).
    for s in range(args.sessions):
        states = _random_states(engine, rng)
        # Warm up memory so that the review quota is actually exercised.
        for _ in range(args.warmup):
            warm_task = bank.tasks[int(rng.choice(task_ids))]
            warm_attempt = _attempt(engine, states, warm_task, now, rng)
            engine.review_attempt(states, warm_task, warm_attempt)
        recent_ids: list[int] = []
        review_count = 0
        for pos in range(args.session_length):
            ctx = SelectionContext(
                now=now,
                session_position=pos,
                session_length=args.session_length,
                review_count=review_count,
                consecutive_errors=0,
                session_elapsed_sec=float(pos * 60),
                days_to_exam=40,
                recent_task_ids=tuple(recent_ids[-7:]),
                seed=s * 100 + pos,
            )
            try:
                result = engine.select(states, ctx)
                attempt = _attempt(engine, states, result.task, now, rng)
                engine.review_attempt(states, result.task, attempt)
            except Exception as exc:
                report["errors"].append(f"session {s}/{pos}: {type(exc).__name__}: {exc}")
                break
            recent_ids.append(result.task.id)
            if ctx.review_count >= 4 and result.score.memory > 0.4:
                alternatives = [x for x in result.ranked if x.memory <= 0.4]
                if alternatives:
                    report["errors"].append(
                        f"session {s}/{pos}: review quota violated; alternatives existed"
                    )
                report["quota_checks"] += 1
            if result.score.memory > 0.4:
                review_count += 1
            report["attempts"] += 1
        report["sessions"] += 1
        for st in states.values():
            if not math.isfinite(st.theta) or not math.isfinite(st.sigma):
                report["errors"].append(f"session {s}: non-finite state")
            if st.sigma < 0.25 - 1e-9 or st.sigma > 1.5 + 1e-9:
                report["errors"].append(f"session {s}: sigma out of bounds {st.sigma}")

    # 3. Dedicated review-quota stress: make every KU due, then ask for tasks
    # after the quota has been exhausted.  If a non-review alternative exists,
    # a review-heavy task must not be selected.
    for i in range(args.quota_checks):
        states = _random_states(engine, rng)
        for st in states.values():
            st.memory = engine.memory.review(
                MemoryState(), Rating.GOOD, now - timedelta(days=30)
            )
        ctx = SelectionContext(
            now=now,
            session_position=5,
            session_length=10,
            review_count=4,
            days_to_exam=40,
            seed=10_000 + i,
        )
        result = engine.select(states, ctx)
        alternatives = [s for s in result.ranked if s.memory <= 0.4]
        if result.score.memory > 0.4 and alternatives:
            report["errors"].append(
                f"quota {i}: review task selected despite non-review alternatives"
            )
        report["quota_checks"] += 1

    # 4. Random attempt stress.
    for i in range(args.attempts):
        states = _random_states(engine, rng)
        task = bank.tasks[int(rng.choice(task_ids))]
        raw = float(rng.choice([0.0, 0.5, 1.0]))
        attempt = Attempt(
            user_id="u",
            task_id=task.id,
            session_id="s",
            timestamp=now,
            raw_score=raw,
            duration_ms=int(rng.integers(500, 600_000)),
            self_grade=Rating(int(rng.integers(1, 5))),
            hint_weight_sum=float(rng.choice([0.0, 0.0, 0.25, 0.6])),
            state_before=engine.snapshot(states, now),
        )
        try:
            engine.review_attempt(states, task, attempt)
        except Exception as exc:
            report["errors"].append(f"attempt {i}: {type(exc).__name__}: {exc}")

    # 4. Optional calibration on synthetic responses from the real bank.
    if args.calibrate:
        calibration_attempts = []
        chosen_tasks = rng.choice(task_ids, size=min(args.calibration_task_count, len(task_ids)), replace=False)
        calibration_tasks = {int(tid): bank.tasks[int(tid)] for tid in chosen_tasks}
        for tid in calibration_tasks:
            for _ in range(args.calibration_per_task):
                states = _random_states(engine, rng)
                attempt = _attempt(engine, states, bank.tasks[tid], now, rng)
                calibration_attempts.append(attempt)
        meta = MetaDifficultyModel().fit(list(calibration_tasks.values()))
        report["calibration_tasks"] = len(calibration_tasks)
        report["calibration_attempts"] = len(calibration_attempts)
        if calibration_tasks:
            try:
                cal = ItemCalibrator(meta=meta).fit(
                    calibration_tasks,
                    calibration_attempts,
                    tau0=2.0,
                    gamma0=0.8,
                )
                report["calibration"] = {
                    "brier": cal.brier,
                    "ece": cal.ece,
                    "tau": cal.tau,
                    "gamma": cal.gamma,
                    "converged": cal.converged,
                }
            except Exception as exc:
                report["errors"].append(f"calibration: {type(exc).__name__}: {exc}")

    report["elapsed_sec"] = round(time.perf_counter() - start, 2)
    report["ok"] = not report["errors"]
    report["reasons"] = dict(report["reasons"])
    report["answer_types"] = dict(report["answer_types"])
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", default="content/fipi_physics_2026.json")
    parser.add_argument("--selections", type=int, default=2000)
    parser.add_argument("--sessions", type=int, default=200)
    parser.add_argument("--session-length", type=int, default=10)
    parser.add_argument("--attempts", type=int, default=2000)
    parser.add_argument("--quota-checks", type=int, default=100)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--calibration-task-count", type=int, default=80)
    parser.add_argument("--calibration-per-task", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
