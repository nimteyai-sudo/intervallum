from __future__ import annotations

from datetime import datetime, timezone

from intervallum_ml.config import EngineConfig
from intervallum_ml.engine import IntervallumEngine
from intervallum_ml.types import (
    ExamBlueprintItem,
    KnowledgeUnit,
    KUType,
    Task,
    TaskMechanic,
)


def make_engine() -> IntervallumEngine:
    kus = {
        i: KnowledgeUnit(i, f"ku{i}", f"KU {i}", KUType.HYBRID, 1, 1)
        for i in range(1, 5)
    }
    tasks = {1: Task(1, TaskMechanic.NUMERIC, (1, 2), t_expected_sec=60)}
    return IntervallumEngine(kus, tasks, EngineConfig())


def blueprint() -> list[ExamBlueprintItem]:
    return [
        ExamBlueprintItem(
            position=i,
            max_points=2.0,
            ku_ids=(1, 2) if i % 2 else (3, 4),
            b_exam=0.0,
            a_exam=1.0,
        )
        for i in range(1, 27)
    ]


def test_readiness_is_bounded_and_ordered():
    engine = make_engine()
    states = engine.new_user_state()
    proj = engine.readiness_projection(
        states,
        blueprint(),
        datetime(2027, 6, 1, tzinfo=timezone.utc),
        now=datetime(2026, 10, 31, tzinfo=timezone.utc),
    )
    assert 0 <= proj.expected_raw <= 52
    assert proj.raw_low <= proj.expected_raw <= proj.raw_high
    assert proj.scaled_low <= proj.expected_scaled <= proj.scaled_high
    assert 0.0 <= proj.confidence <= 1.0


def test_stronger_state_has_higher_projection():
    engine = make_engine()
    weak = engine.new_user_state()
    strong = engine.new_user_state()
    for st in strong.values():
        st.theta = 2.5
    p_weak = engine.readiness_projection(
        weak, blueprint(), datetime(2027, 6, 1, tzinfo=timezone.utc)
    )
    p_strong = engine.readiness_projection(
        strong, blueprint(), datetime(2027, 6, 1, tzinfo=timezone.utc)
    )
    assert p_strong.expected_raw > p_weak.expected_raw
    assert p_strong.expected_scaled > p_weak.expected_scaled


def test_confidence_grows_with_attempts():
    engine = make_engine()
    states = engine.new_user_state()
    low = engine.readiness_projection(
        states, blueprint(), datetime(2027, 6, 1, tzinfo=timezone.utc)
    )
    for st in states.values():
        st.attempts_count = 200
        st.sigma = 0.3
    high = engine.readiness_projection(
        states, blueprint(), datetime(2027, 6, 1, tzinfo=timezone.utc)
    )
    assert high.confidence > low.confidence


def test_blueprint_item_without_kus_uses_c_exam():
    engine = make_engine()
    states = engine.new_user_state()
    item = ExamBlueprintItem(
        position=1,
        max_points=2.0,
        ku_ids=(),
        b_exam=0.0,
        a_exam=1.0,
        c_exam=0.25,
    )
    proj = engine.readiness_projection(
        states, [item], datetime(2027, 6, 1, tzinfo=timezone.utc)
    )
    assert proj.expected_raw > 0.5
    assert 0 <= proj.raw_low <= proj.expected_raw <= proj.raw_high <= 2
