"""Policy benchmark on a synthetic ground-truth learning model.

The benchmark compares the pedagogical selector with random task order under a
known learning rule: true mastery grows most when the student practices at a
success probability near 0.7 (the Zone of Proximal Development).  This is a
logic check of the recommendation objective, not evidence of real educational
efficacy.

Run:
    python scripts/policy_benchmark.py
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timedelta, timezone

import numpy as np

from intervallum_ml.config import EngineConfig
from intervallum_ml.engine import IntervallumEngine
from intervallum_ml.mathutils import clamp, sigmoid, softmin
from intervallum_ml.simulation import build_world
from intervallum_ml.types import Attempt, Rating, SelectionContext


def true_p(world, user, task, theta, now, start):
    vals = []
    for ku_id in task.ku_ids:
        vals.append(theta[(user, ku_id)])
    theta_task = softmin(vals, world.true_tau, task.weights())
    p_star = sigmoid(task.a_discrimination * (theta_task - task.b_difficulty))
    return clamp(task.c_guess + (1 - task.c_guess) * p_star, 1e-6, 1 - 1e-6)


def learning_weight(p: float, target: float = 0.70, sigma: float = 0.20) -> float:
    return math.exp(-((p - target) ** 2) / (2 * sigma ** 2))


def run_policy(
    world,
    engine,
    policy: str,
    steps: int,
    seed: int,
    true_theta,
    initial_noise: float,
):
    rng = np.random.default_rng(seed)
    now = datetime(2026, 9, 1, tzinfo=timezone.utc)
    gains = []
    for user in world.users:
        theta = dict(true_theta)
        states = engine.new_user_state()
        for ku_id in states:
            states[ku_id].theta = theta[(user, ku_id)] + float(rng.normal(0, initial_noise))
            states[ku_id].sigma = 1.0
        recent: list[int] = []
        for step in range(steps):
            if policy == "random":
                task = world.tasks[int(rng.choice(list(world.tasks)))]
            else:
                ctx = SelectionContext(
                    now=now + timedelta(days=step // 10),
                    session_position=step % 10,
                    session_length=10,
                    days_to_exam=120,
                    recent_task_ids=tuple(recent[-7:]),
                    seed=seed * 1000 + step,
                )
                task = engine.select(states, ctx).task
            p = true_p(world, user, task, theta, now, now)
            y = float(rng.random() < p)
            gain = 0.035 * learning_weight(p) * (1.0 if y else 0.25)
            for ku_id in task.ku_ids:
                theta[(user, ku_id)] += gain
            attempt = Attempt(
                user_id=user,
                task_id=task.id,
                session_id="bench",
                timestamp=now + timedelta(days=step // 10),
                raw_score=y,
                duration_ms=int(task.t_expected_sec * 1000),
                self_grade=Rating.GOOD if y else Rating.AGAIN,
                state_before=engine.snapshot(states, now),
                repeat_same_task=task.id in recent,
            )
            engine.review_attempt(states, task, attempt)
            recent.append(task.id)
        gains.append(
            sum(theta[(user, k)] - true_theta[(user, k)] for k in world.knowledge_units)
            / len(world.knowledge_units)
        )
    return gains


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=30)
    parser.add_argument("--kus", type=int, default=20)
    parser.add_argument("--tasks", type=int, default=120)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--initial-noise", type=float, default=0.5)
    args = parser.parse_args()

    world = build_world(n_users=args.users, n_kus=args.kus, n_tasks=args.tasks, seed=args.seed)
    engine = IntervallumEngine(world.knowledge_units, world.tasks, EngineConfig())
    random_gains = run_policy(
        world, engine, "random", args.steps, args.seed + 1, world.true_theta, args.initial_noise
    )
    adaptive_gains = run_policy(
        world, engine, "adaptive", args.steps, args.seed + 2, world.true_theta, args.initial_noise
    )
    diff = np.asarray(adaptive_gains) - np.asarray(random_gains)
    report = {
        "users": args.users,
        "steps": args.steps,
        "random_mean_gain": float(np.mean(random_gains)),
        "adaptive_mean_gain": float(np.mean(adaptive_gains)),
        "difference": float(np.mean(diff)),
        "adaptive_wins": int(np.sum(diff > 0)),
        "random_wins": int(np.sum(diff < 0)),
        "ties": int(np.sum(diff == 0)),
        "relative_improvement": float(
            (np.mean(adaptive_gains) - np.mean(random_gains)) / max(np.mean(random_gains), 1e-9)
        ),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
