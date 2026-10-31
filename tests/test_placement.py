from __future__ import annotations

from datetime import datetime, timezone

from intervallum_ml.coldstart import AdaptivePlacement, cold_start_task
from intervallum_ml.config import EngineConfig
from intervallum_ml.engine import IntervallumEngine
from intervallum_ml.simulation import build_world
from intervallum_ml.types import Attempt, Rating


def test_placement_covers_sections_and_sets_priors():
    world = build_world(n_users=2, n_kus=12, n_tasks=60, seed=9)
    engine = IntervallumEngine(world.knowledge_units, world.tasks, EngineConfig())
    states = engine.new_user_state()
    section_by_ku = {k: v.section_id for k, v in world.knowledge_units.items()}
    placement = AdaptivePlacement(engine.mastery, section_by_ku, n_items=8)
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    tasks = placement.plan(list(world.tasks.values()), states, now)
    assert len(tasks) == 8
    covered = {section_by_ku[k] for t in tasks for k in t.ku_ids}
    assert len(covered) >= 3

    attempts = []
    for task in tasks:
        snapshot = engine.snapshot(states, now)
        attempts.append(
            (
                task,
                Attempt(
                    user_id="u1",
                    task_id=task.id,
                    session_id="placement",
                    timestamp=now,
                    raw_score=1.0,
                    duration_ms=int(task.t_expected_sec * 1000),
                    self_grade=Rating.GOOD,
                    state_before=snapshot,
                ),
            )
        )
    result = placement.apply(engine, states, attempts)
    assert len(result.selected_task_ids) == 8
    assert result.theta_updates
    assert any(st.attempts_count > 0 for st in states.values())


def test_cold_start_task_uses_meta_model():
    from intervallum_ml.coldstart import MetaDifficultyModel

    world = build_world(n_users=2, n_kus=4, n_tasks=10, seed=10)
    tasks = list(world.tasks.values())
    meta = MetaDifficultyModel().fit(tasks)
    new_task = cold_start_task(tasks[0], meta)
    assert new_task.b_difficulty == meta.predict(tasks[0])
