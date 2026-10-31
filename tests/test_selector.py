from __future__ import annotations

from datetime import datetime, timezone

from intervallum_ml.config import EngineConfig
from intervallum_ml.engine import IntervallumEngine
from intervallum_ml.mathutils import softmax
from intervallum_ml.types import (
    KnowledgeUnit,
    KUType,
    MemoryState,
    Rating,
    SelectionContext,
    Task,
    TaskMechanic,
)


def make_engine() -> IntervallumEngine:
    kus = {
        1: KnowledgeUnit(1, "parent", "Parent", KUType.HYBRID, 1, 1),
        2: KnowledgeUnit(2, "child", "Child", KUType.HYBRID, 1, 1, parent_ids=(1,)),
        3: KnowledgeUnit(3, "other", "Other", KUType.HYBRID, 2, 2),
        4: KnowledgeUnit(4, "fresh", "Fresh", KUType.HYBRID, 3, 3),
    }
    tasks = {
        1: Task(1, TaskMechanic.MCQ_SINGLE, (2,), b_difficulty=0.0, c_guess=0.25),
        2: Task(2, TaskMechanic.MCQ_SINGLE, (3,), b_difficulty=0.0, c_guess=0.25),
        3: Task(3, TaskMechanic.NUMERIC, (3,), b_difficulty=2.0, c_guess=0.0),
        4: Task(4, TaskMechanic.NUMERIC, (4,), b_difficulty=0.0, c_guess=0.0),
    }
    return IntervallumEngine(kus, tasks, EngineConfig())


def test_teacher_assignment_wins():
    engine = make_engine()
    states = engine.new_user_state()
    ctx = SelectionContext(
        now=datetime(2026, 1, 1, tzinfo=timezone.utc),
        teacher_assignment_ids=frozenset({3}),
        seed=1,
    )
    result = engine.select(states, ctx)
    assert result.task.id == 3
    assert result.score.reason.value == "teacher_assignment"


def test_prerequisite_gate_blocks_failed_parent():
    engine = make_engine()
    states = engine.new_user_state()
    states[1].theta = -3.0
    states[1].attempts_count = 10
    ctx = SelectionContext(now=datetime(2026, 1, 1, tzinfo=timezone.utc), seed=1)
    result = engine.select(states, ctx)
    assert result.task.id != 1


def test_zpd_prefers_task_near_target():
    engine = make_engine()
    states = engine.new_user_state()
    # Task 1 has p_star ~ 0.5, task 3 has p_star ~ 0.12; warmup target is 0.85,
    # so task 1 should be closer than task 3.
    ctx = SelectionContext(now=datetime(2026, 1, 1, tzinfo=timezone.utc), session_position=1, seed=2)
    ranked = engine.rank(states, ctx)
    by_id = {s.task_id: s for s in ranked}
    assert by_id[1].zpd > by_id[3].zpd


def test_propensity_is_probability():
    engine = make_engine()
    states = engine.new_user_state()
    ctx = SelectionContext(now=datetime(2026, 1, 1, tzinfo=timezone.utc), seed=3)
    result = engine.select(states, ctx)
    assert 0.0 < result.propensity <= 1.0
    assert len(result.ranked) == result.candidates


def test_review_quota_prefers_non_review_task():
    engine = make_engine()
    states = engine.new_user_state()
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    mem = engine.memory.review(MemoryState(), Rating.GOOD, datetime(2025, 12, 1, tzinfo=timezone.utc))
    states[3].memory = mem
    # Task 3 is review-heavy, task 4 uses a fresh KU and is not.
    ctx = SelectionContext(
        now=now,
        session_position=6,
        session_length=10,
        review_count=10,
        recent_task_ids=(1, 2),
        seed=4,
    )
    result = engine.select(states, ctx)
    assert result.task.id == 4


def test_review_quota_keeps_review_task_if_no_alternative():
    engine = make_engine()
    states = engine.new_user_state()
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    mem = engine.memory.review(MemoryState(), Rating.GOOD, datetime(2025, 12, 1, tzinfo=timezone.utc))
    states[3].memory = mem
    ctx = SelectionContext(
        now=now,
        session_position=6,
        session_length=10,
        review_count=10,
        recent_task_ids=(1, 2, 4),
        seed=4,
    )
    result = engine.select(states, ctx)
    assert result.task.id == 3
    assert result.score.memory > 0


def test_diversity_penalty_is_applied():
    engine = make_engine()
    states = engine.new_user_state()
    ctx = SelectionContext(
        now=datetime(2026, 1, 1, tzinfo=timezone.utc),
        recent_mechanics=(TaskMechanic.NUMERIC,),
        recent_ku_sets=(frozenset({3}),),
        seed=5,
    )
    ranked = engine.rank(states, ctx)
    by_id = {s.task_id: s for s in ranked}
    assert by_id[3].diversity_penalty > 0
    assert by_id[1].diversity_penalty == 0


def test_propensity_matches_gumbel_max_policy():
    engine = make_engine()
    states = engine.new_user_state()
    ctx = SelectionContext(
        now=datetime(2026, 1, 1, tzinfo=timezone.utc),
        exploration_scale=0.05,
        seed=9,
    )
    result = engine.select(states, ctx)
    base = [s.base_utility for s in result.ranked]
    probs = softmax(base, temperature=ctx.exploration_scale)
    assert abs(result.propensity - float(probs[0])) < 1e-9
