from __future__ import annotations

from intervallum_ml.coldstart import MetaDifficultyModel, cold_start_ku, task_features
from intervallum_ml.types import KnowledgeUnit, KUType, Task, TaskMechanic, UserKUState


def test_meta_difficulty_is_learnable():
    tasks = [
        Task(i, TaskMechanic.MCQ_SINGLE, (1,), expert_difficulty=1, t_expected_sec=30)
        for i in range(1, 6)
    ] + [
        Task(i, TaskMechanic.FORMULA_DERIVE, (1, 2), expert_difficulty=5, t_expected_sec=180)
        for i in range(6, 11)
    ]
    model = MetaDifficultyModel(ridge=0.1).fit(tasks)
    easy = model.predict(tasks[0])
    hard = model.predict(tasks[-1])
    assert hard > easy


def test_task_features_are_stable_length():
    task = Task(1, TaskMechanic.NUMERIC, (1, 2), t_expected_sec=60)
    assert len(task_features(task)) == 9


def test_cold_start_ku_uses_parents():
    states = {1: UserKUState(ku_id=1, theta=1.5)}
    ku = KnowledgeUnit(
        2, "child", "Child", KUType.HYBRID, 1, 1, parent_ids=(1,)
    )
    st = cold_start_ku(states, ku)
    assert 2 in states
    assert st.theta == 1.5
    assert st.sigma == 0.85
