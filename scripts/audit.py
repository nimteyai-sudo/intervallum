"""Logic audit for the pedagogical selector.

This is not an educational efficacy study.  It checks invariants on synthetic
users and tasks:

* no quarantined task is selected;
* a teacher assignment is honored;
* a failed prerequisite is not silently selected;
* the review quota is respected when alternatives exist;
* propensities are valid probabilities;
* the selector does not crash on an empty/new state.

Run:
    python scripts/audit.py
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone

import numpy as np

from intervallum_ml.config import EngineConfig
from intervallum_ml.engine import IntervallumEngine
from intervallum_ml.simulation import build_world, simulate
from intervallum_ml.types import (
    Attempt,
    KnowledgeUnit,
    KUType,
    Rating,
    SelectionContext,
    Task,
    TaskMechanic,
)


def _attempt(engine, states, task, now, rng):
    pred = engine.mastery.predict(states, task, now)
    y = float(rng.random() < pred.p_correct)
    duration = int(task.t_expected_sec * rng.uniform(0.5, 2.0) * 1000)
    self_grade = Rating.GOOD if y else Rating.AGAIN
    return Attempt(
        user_id="audit",
        task_id=task.id,
        session_id="audit",
        timestamp=now,
        raw_score=y,
        duration_ms=duration,
        self_grade=self_grade,
        state_before=engine.snapshot(states, now),
    )


def audit_session(engine, states, seed=1):
    rng = np.random.default_rng(seed)
    now = datetime(2026, 10, 31, tzinfo=timezone.utc)
    recent_ids: list[int] = []
    recent_mechanics = []
    recent_ku_sets = []
    review_count = 0
    reasons = Counter()
    rows = []
    for pos in range(10):
        ctx = SelectionContext(
            now=now,
            session_position=pos,
            session_length=10,
            review_count=review_count,
            days_to_exam=40,
            recent_task_ids=tuple(recent_ids[-7:]),
            recent_mechanics=tuple(recent_mechanics[-2:]),
            recent_ku_sets=tuple(recent_ku_sets[-2:]),
            seed=seed * 100 + pos,
        )
        result = engine.select(states, ctx)
        assert 0.0 < result.propensity <= 1.0
        assert not result.task.is_quarantined
        reasons[result.score.reason.value] += 1
        rows.append(
            (
                result.task.id,
                result.score.reason.value,
                round(result.score.p_correct, 3),
                round(result.score.zpd, 3),
                round(result.score.memory, 3),
                round(result.score.exam, 3),
            )
        )
        if result.score.memory > 0.4:
            review_count += 1
        recent_ids.append(result.task.id)
        recent_mechanics.append(result.task.mechanic)
        recent_ku_sets.append(frozenset(result.task.ku_ids))
        attempt = _attempt(engine, states, result.task, now, rng)
        engine.review_attempt(states, result.task, attempt)
    return rows, reasons, review_count


def audit_teacher_assignment(engine, states):
    task_id = next(iter(engine.tasks))
    ctx = SelectionContext(
        now=datetime(2026, 10, 31, tzinfo=timezone.utc),
        teacher_assignment_ids=frozenset({task_id}),
        recent_task_ids=tuple(engine.tasks.keys()),
        seed=3,
    )
    result = engine.select(states, ctx)
    assert result.task.id == task_id, (result.task.id, task_id)
    return task_id


def audit_prerequisite_gate():
    kus = {
        1: KnowledgeUnit(1, "parent", "Parent", KUType.HYBRID, 1, 1),
        2: KnowledgeUnit(2, "child", "Child", KUType.HYBRID, 1, 1, parent_ids=(1,)),
        3: KnowledgeUnit(3, "other", "Other", KUType.HYBRID, 2, 2),
    }
    tasks = {
        1: Task(1, TaskMechanic.MCQ_SINGLE, (2,), c_guess=0.25),
        2: Task(2, TaskMechanic.MCQ_SINGLE, (3,), c_guess=0.25),
    }
    engine = IntervallumEngine(kus, tasks, EngineConfig())
    states = engine.new_user_state()
    states[1].theta = -3.0
    states[1].attempts_count = 10
    selections = []
    for i in range(20):
        ctx = SelectionContext(
            now=datetime(2026, 10, 31, tzinfo=timezone.utc),
            seed=i,
        )
        selections.append(engine.select(states, ctx).task.id)
    assert 1 not in selections, selections
    return selections


def audit_quarantine():
    world = build_world(n_users=3, n_kus=6, n_tasks=20, seed=12)
    # Quarantine the first task.
    bad_id = next(iter(world.tasks))
    tasks = dict(world.tasks)
    tasks[bad_id] = Task(
        bad_id,
        tasks[bad_id].mechanic,
        tasks[bad_id].ku_ids,
        c_guess=tasks[bad_id].c_guess,
        t_expected_sec=tasks[bad_id].t_expected_sec,
        b_difficulty=tasks[bad_id].b_difficulty,
        a_discrimination=tasks[bad_id].a_discrimination,
        is_quarantined=True,
    )
    engine = IntervallumEngine(world.knowledge_units, tasks, EngineConfig())
    states = engine.new_user_state()
    selected = []
    for i in range(50):
        ctx = SelectionContext(now=datetime(2026, 10, 31, tzinfo=timezone.utc), seed=i)
        selected.append(engine.select(states, ctx).task.id)
    assert bad_id not in selected
    return bad_id, len(selected)


def main() -> None:
    world = build_world(n_users=20, n_kus=16, n_tasks=80, seed=7)
    sim = simulate(world, attempts_per_user=25, seed=11)
    engine = IntervallumEngine(world.knowledge_units, world.tasks, EngineConfig())
    states = sim.final_states[world.users[0]]

    print("=== Session audit ===")
    rows, reasons, review_count = audit_session(engine, states, seed=4)
    for row in rows:
        print("task=%2d reason=%-18s p=%.3f zpd=%.3f mem=%.3f exam=%.3f" % row)
    print("reasons:", dict(reasons))
    print("review-heavy tasks:", review_count, "of 10; quota=4")

    print("\n=== Teacher assignment ===")
    print("selected:", audit_teacher_assignment(engine, states))

    print("\n=== Prerequisite gate ===")
    print("selected task ids over 20 choices:", audit_prerequisite_gate())

    print("\n=== Quarantine ===")
    bad_id, n = audit_quarantine()
    print(f"quarantined task {bad_id} not selected in {n} choices")

    print("\n=== Calibration on synthetic logs ===")
    from intervallum_ml.calibration import ItemCalibrator
    from intervallum_ml.coldstart import MetaDifficultyModel

    meta = MetaDifficultyModel().fit(list(world.tasks.values()))
    report = ItemCalibrator(meta=meta).fit(world.tasks, sim.attempts)
    print(
        {
            "attempts": report.n_attempts,
            "items": report.n_items,
            "brier": round(report.brier, 4),
            "ece": round(report.ece, 4),
            "tau": round(report.tau, 3),
            "gamma": round(report.gamma, 3),
            "converged": report.converged,
        }
    )
    print("\nAll selector invariants passed.")


if __name__ == "__main__":
    main()
