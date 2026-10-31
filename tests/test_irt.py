from __future__ import annotations

from datetime import datetime, timezone

import pytest

from intervallum_ml.irt import MasteryEngine
from intervallum_ml.mathutils import sigmoid, softmin
from intervallum_ml.types import KnowledgeUnit, KUType, Task, TaskMechanic, UserKUState


def ku(i: int) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=i,
        code=f"ku{i}",
        name=f"KU {i}",
        ku_type=KUType.HYBRID,
        topic_id=1,
        section_id=1,
    )


def make_task(ku_ids=(1, 2), b=0.0, a=1.0, c=0.0) -> Task:
    return Task(
        id=10,
        mechanic=TaskMechanic.FIND_ERROR,
        ku_ids=ku_ids,
        b_difficulty=b,
        a_discrimination=a,
        c_guess=c,
        t_expected_sec=60,
    )


def test_softmin_is_between_min_and_mean():
    vals = [2.0, 0.0, -1.0]
    sm = softmin(vals, tau=2.0)
    assert min(vals) <= sm <= sum(vals) / len(vals)
    assert sm < sum(vals) / len(vals)


def test_conjunctive_task_is_pulled_down_by_weakest_ku():
    engine = MasteryEngine()
    states = {
        1: UserKUState(ku_id=1, theta=2.0),
        2: UserKUState(ku_id=2, theta=-1.0),
    }
    pred = engine.predict(states, make_task(), datetime.now(timezone.utc))
    assert pred.theta_task < 0.5
    assert pred.p_star < sigmoid(0.5)


def test_credit_assignment_blames_weakest_ku():
    engine = MasteryEngine()
    states = {
        1: UserKUState(ku_id=1, theta=2.0),
        2: UserKUState(ku_id=2, theta=-0.5),
    }
    task = make_task()
    now = datetime.now(timezone.utc)
    pred = engine.predict(states, task, now)
    assert pred.responsibilities[2] > pred.responsibilities[1]
    before = {k: v.theta for k, v in states.items()}
    engine.update(states, task, pred, y_eff=0.0, now=now)
    delta_weak = states[2].theta - before[2]
    delta_strong = states[1].theta - before[1]
    assert delta_weak < delta_strong
    assert delta_weak < 0


def test_correct_answer_increases_mastery_and_reduces_sigma():
    engine = MasteryEngine()
    states = {1: UserKUState(ku_id=1, theta=0.0), 2: UserKUState(ku_id=2, theta=0.0)}
    task = make_task()
    now = datetime.now(timezone.utc)
    pred = engine.predict(states, task, now)
    before_theta = states[1].theta
    before_sigma = states[1].sigma
    engine.update(states, task, pred, y_eff=1.0, now=now)
    assert states[1].theta > before_theta
    assert states[1].sigma < before_sigma
    assert states[1].attempts_count == 1


def test_memory_decay_reduces_effective_theta():
    from intervallum_ml.memory import MemoryScheduler
    from intervallum_ml.types import MemoryState, Rating

    scheduler = MemoryScheduler()
    engine = MasteryEngine(memory=scheduler)
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    mem = scheduler.review(MemoryState(), Rating.GOOD, now)
    st = UserKUState(ku_id=1, theta=2.0, memory=mem)
    fresh = engine.effective_theta(st, now)
    later = engine.effective_theta(st, datetime(2026, 6, 1, tzinfo=timezone.utc))
    assert later < fresh
    assert fresh == pytest.approx(2.0, abs=1e-6)


def test_no_kus_does_not_crash():
    engine = MasteryEngine()
    task = Task(id=1, mechanic=TaskMechanic.NUMERIC, ku_ids=())
    pred = engine.predict({}, task, datetime.now(timezone.utc))
    assert 0 <= pred.p_correct <= 1
    assert engine.update({}, task, pred, 1.0, datetime.now(timezone.utc)) == ()
