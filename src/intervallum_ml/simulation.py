"""Synthetic world generator and recovery checks.

This module exists for a specific scientific reason: every estimator must be
validated on data with a known ground truth before it is trusted on real
students.  The simulator creates latent abilities, item parameters, memory
decay and noisy responses, then checks whether the pipeline recovers them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

from .engine import IntervallumEngine
from .mathutils import clamp, sigmoid, softmin
from .types import (
    Attempt,
    KnowledgeUnit,
    KUType,
    Rating,
    Task,
    TaskMechanic,
    UserKUState,
)


@dataclass(frozen=True)
class SyntheticWorld:
    knowledge_units: dict[int, KnowledgeUnit]
    tasks: dict[int, Task]
    true_theta: dict[tuple[str, int], float]
    true_stability: dict[tuple[str, int], float]
    true_b: dict[int, float]
    true_a: dict[int, float]
    true_tau: float
    true_gamma: float
    users: tuple[str, ...]


@dataclass(frozen=True)
class SimulationResult:
    world: SyntheticWorld
    attempts: tuple[Attempt, ...]
    final_states: dict[str, dict[int, UserKUState]]


def build_world(
    *,
    n_users: int = 60,
    n_kus: int = 40,
    n_tasks: int = 200,
    seed: int = 7,
) -> SyntheticWorld:
    rng = np.random.default_rng(seed)
    kus: dict[int, KnowledgeUnit] = {}
    for i in range(1, n_kus + 1):
        section = 1 + (i - 1) * 4 // n_kus
        parents: tuple[int, ...] = ()
        if i > 4 and rng.random() < 0.45:
            parents = (int(rng.integers(max(1, i - 6), i)),)
        kus[i] = KnowledgeUnit(
            id=i,
            code=f"ku_{i}",
            name=f"Knowledge unit {i}",
            ku_type=KUType.HYBRID if i % 3 else KUType.PROCEDURAL,
            topic_id=section,
            section_id=section,
            ege_weight=float(rng.uniform(0.5, 3.0)),
            parent_ids=parents,
            neighbor_ids=(),
        )

    mechanics = list(TaskMechanic)
    tasks: dict[int, Task] = {}
    for i in range(1, n_tasks + 1):
        k = int(rng.integers(1, 4))
        ku_ids = tuple(sorted(rng.choice(list(kus), size=k, replace=False).tolist()))
        mech = mechanics[int(rng.integers(0, len(mechanics)))]
        c = {
            TaskMechanic.MCQ_SINGLE: 0.25,
            TaskMechanic.MCQ_MULTI: 0.05,
            TaskMechanic.PREDICT_CHANGE: 0.11,
        }.get(mech, 0.0)
        b = float(rng.normal(0.0, 1.0))
        a = float(rng.uniform(0.7, 2.2))
        expert = int(clamp(round(3 + b * 1.2), 1, 5))
        tasks[i] = Task(
            id=i,
            mechanic=mech,
            ku_ids=ku_ids,
            c_guess=c,
            t_expected_sec=float(rng.uniform(30, 180)),
            b_difficulty=b,
            a_discrimination=a,
            expert_difficulty=expert,
            topic_id=kus[ku_ids[0]].topic_id,
            section_id=kus[ku_ids[0]].section_id,
            text="Synthetic task " + "x" * int(rng.integers(20, 200)),
        )

    users = tuple(f"u{i}" for i in range(n_users))
    true_theta: dict[tuple[str, int], float] = {}
    true_stability: dict[tuple[str, int], float] = {}
    for u in users:
        base = float(rng.normal(0.0, 0.8))
        for k in kus:
            true_theta[(u, k)] = float(base + rng.normal(0.0, 0.8))
            true_stability[(u, k)] = float(rng.uniform(2.0, 30.0))

    return SyntheticWorld(
        knowledge_units=kus,
        tasks=tasks,
        true_theta=true_theta,
        true_stability=true_stability,
        true_b={i: t.b_difficulty for i, t in tasks.items()},
        true_a={i: t.a_discrimination for i, t in tasks.items()},
        true_tau=2.2,
        true_gamma=0.9,
        users=users,
    )


def _true_retrievability(world: SyntheticWorld, user: str, ku_id: int, now: datetime, start: datetime) -> float:
    stability = world.true_stability[(user, ku_id)]
    elapsed = max(0.0, (now - start).total_seconds() / 86400.0)
    # A simple exponential forgetting curve for the simulator.  The production
    # model uses FSRS; the point of the simulation is to recover the latent
    # ability and item parameters, not to assume FSRS is the truth.
    return float(0.9 ** (elapsed / max(stability, 1e-6)))


def simulate(
    world: SyntheticWorld,
    *,
    attempts_per_user: int = 40,
    days: int = 30,
    seed: int = 11,
) -> SimulationResult:
    rng = np.random.default_rng(seed)
    engine = IntervallumEngine(world.knowledge_units, world.tasks)
    start = datetime(2026, 9, 1, tzinfo=timezone.utc)
    states_by_user: dict[str, dict[int, UserKUState]] = {
        u: engine.new_user_state() for u in world.users
    }
    all_attempts: list[Attempt] = []

    for u in world.users:
        states = states_by_user[u]
        for step in range(attempts_per_user):
            now = start + timedelta(
                days=float(rng.uniform(0, days)),
                minutes=float(rng.uniform(0, 1440)),
            )
            # Mildly difficulty-targeted sampling: random tasks from the pool,
            # weighted toward predicted success around 0.7.  This mirrors a
            # product that does not expose only trivial items.
            task_ids = list(world.tasks)
            sampled = rng.choice(task_ids, size=min(20, len(task_ids)), replace=False)
            best_task = None
            best_distance = float("inf")
            for tid in sampled:
                task = world.tasks[int(tid)]
                true_vals = []
                for ku_id in task.ku_ids:
                    st = states.get(ku_id)
                    if st is not None and st.memory is not None and st.memory.is_started():
                        r = engine.memory.retrievability(st.memory, now)
                    else:
                        r = 1.0
                    true_vals.append(
                        world.true_theta[(u, ku_id)] + world.true_gamma * math.log(max(r, 1e-6))
                    )
                theta_task = softmin(true_vals, world.true_tau, task.weights())
                p_star = sigmoid(task.a_discrimination * (theta_task - task.b_difficulty))
                p = task.c_guess + (1 - task.c_guess) * p_star
                distance = abs(p - 0.72)
                if distance < best_distance:
                    best_distance = distance
                    best_task = task
            if best_task is None:
                continue
            task = best_task
            true_vals = []
            for ku_id in task.ku_ids:
                st = states.get(ku_id)
                if st is not None and st.memory is not None and st.memory.is_started():
                    r = engine.memory.retrievability(st.memory, now)
                else:
                    r = 1.0
                true_vals.append(
                    world.true_theta[(u, ku_id)] + world.true_gamma * math.log(max(r, 1e-6))
                )
            theta_task = softmin(true_vals, world.true_tau, task.weights())
            p_true = clamp(
                task.c_guess
                + (1 - task.c_guess)
                * sigmoid(task.a_discrimination * (theta_task - task.b_difficulty)),
                1e-6,
                1 - 1e-6,
            )
            y = float(rng.random() < p_true)
            duration_ratio = float(rng.lognormal(mean=0.25, sigma=0.35))
            duration_ms = int(max(1000, task.t_expected_sec * duration_ratio * 1000))
            hint_sum = 0.0
            if rng.random() < 0.18:
                hint_sum = float(rng.choice([0.25, 0.6]))
            if y >= 0.99 and duration_ratio < 0.7 and hint_sum == 0:
                self_grade = Rating.EASY
            elif y >= 0.99 and duration_ratio <= 1.8:
                self_grade = Rating.GOOD
            elif y >= 0.99:
                self_grade = Rating.HARD
            else:
                self_grade = Rating.AGAIN

            snapshot = engine.snapshot(states, now)
            attempt = Attempt(
                user_id=u,
                task_id=task.id,
                session_id=f"{u}-s{step // 8}",
                timestamp=now,
                raw_score=y,
                duration_ms=duration_ms,
                self_grade=self_grade,
                hints_used=(1,) if hint_sum else (),
                hint_weight_sum=hint_sum,
                pos_in_session=step % 8,
                session_elapsed_sec=float((step % 8) * 60),
                consecutive_errors=0,
                state_before=snapshot,
                propensity_score=1.0 / len(sampled),
            )
            engine.review_attempt(states, task, attempt)
            all_attempts.append(attempt)

    return SimulationResult(
        world=world,
        attempts=tuple(all_attempts),
        final_states=states_by_user,
    )


def recovery_report(result: SimulationResult) -> dict[str, float]:
    world = result.world
    theta_errors: list[float] = []
    for u, states in result.final_states.items():
        for ku_id, st in states.items():
            theta_errors.append(abs(st.theta - world.true_theta[(u, ku_id)]))
    return {
        "theta_mae": float(np.mean(theta_errors)) if theta_errors else 0.0,
        "n_attempts": float(len(result.attempts)),
    }
