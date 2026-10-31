from __future__ import annotations

from datetime import datetime, timezone

from intervallum_ml.config import EngineConfig
from intervallum_ml.engine import IntervallumEngine
from intervallum_ml.types import (
    Attempt,
    KnowledgeUnit,
    KUType,
    Rating,
    SelectionContext,
    Task,
    TaskMechanic,
)


def make_engine() -> IntervallumEngine:
    kus = {
        1: KnowledgeUnit(1, "f", "Formula", KUType.DECLARATIVE, 1, 1),
        2: KnowledgeUnit(2, "s", "Skill", KUType.PROCEDURAL, 1, 1),
    }
    tasks = {
        1: Task(1, TaskMechanic.FORMULA_DERIVE, (1, 2), t_expected_sec=60),
    }
    return IntervallumEngine(kus, tasks, EngineConfig())


def test_full_attempt_updates_mastery_and_memory():
    engine = make_engine()
    states = engine.new_user_state()
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    snapshot = engine.snapshot(states, now)
    attempt = Attempt(
        user_id="u1",
        task_id=1,
        session_id="s1",
        timestamp=now,
        raw_score=1.0,
        duration_ms=30000,
        self_grade=Rating.GOOD,
        state_before=snapshot,
    )
    result = engine.review_attempt(states, engine.tasks[1], attempt)
    assert result.telemetry.y_eff == 1.0
    assert result.mastery_updates
    assert result.memory_updates
    assert states[1].memory is not None and states[1].memory.stability > 0
    assert states[2].memory is None  # procedural unit is not scheduled by FSRS


def test_engine_select_and_readiness_smoke():
    engine = make_engine()
    states = engine.new_user_state()
    selection = engine.select(
        states,
        SelectionContext(now=datetime(2026, 1, 1, tzinfo=timezone.utc), seed=1),
    )
    assert selection.task.id == 1
    assert selection.propensity > 0
