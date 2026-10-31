"""Conjunctive 3PL IRT mastery layer.

A task is treated as a conjunction of requirements: a student who knows
Newton's second law but cannot project a vector onto an axis should fail a
compound task.  The smooth minimum (softmin) implements that "weakest link"
behaviour without the discontinuity of a hard min.

The online update is an Elo/Glicko-style Bayesian step with responsibility
weights, so an error is attributed mostly to the weakest knowledge unit that
the task requires.
"""

from __future__ import annotations

import math
from typing import Mapping, MutableMapping, Sequence

from .config import MasteryConfig
from .mathutils import clamp, safe_div, sigmoid, softmax, softmin
from .memory import MemoryScheduler
from .types import (
    MasteryUpdate,
    Task,
    TaskPrediction,
    UserKUState,
)


class MasteryEngine:
    def __init__(
        self,
        cfg: MasteryConfig | None = None,
        memory: MemoryScheduler | None = None,
        gamma: float = 0.8,
    ) -> None:
        self.cfg = cfg or MasteryConfig()
        self.memory = memory or MemoryScheduler()
        self.gamma = gamma

    def effective_theta(self, state: UserKUState, now) -> float:
        if state.memory is None or not state.memory.is_started():
            return state.theta
        r = self.memory.retrievability(state.memory, now)
        r = clamp(r, self.cfg.min_retrievability, 1.0)
        return state.theta + self.gamma * math.log(r)

    def _ensure_state(
        self, states: MutableMapping[int, UserKUState], ku_id: int
    ) -> UserKUState:
        if ku_id not in states:
            states[ku_id] = UserKUState(ku_id=ku_id)
        return states[ku_id]

    def predict(
        self,
        states: MutableMapping[int, UserKUState],
        task: Task,
        now,
    ) -> TaskPrediction:
        if not task.ku_ids:
            theta_task = 0.0
            p_star = sigmoid(-task.b_difficulty)
            p_correct = task.c_guess + (1.0 - task.c_guess) * p_star
            return TaskPrediction(
                task_id=task.id,
                theta_task=theta_task,
                p_star=p_star,
                p_correct=clamp(p_correct, 0.0, 1.0),
                sigma_task=1.0,
                effective_thetas={},
                responsibilities={},
            )

        effective: dict[int, float] = {}
        sigmas: list[float] = []
        for ku_id in task.ku_ids:
            st = self._ensure_state(states, ku_id)
            effective[ku_id] = self.effective_theta(st, now)
            sigmas.append(st.sigma)

        weights = task.weights()
        theta_task = softmin([effective[k] for k in task.ku_ids], self.cfg.tau, weights)
        p_star = sigmoid(task.a_discrimination * (theta_task - task.b_difficulty))
        c = clamp(task.c_guess, 0.0, 0.99)
        p_correct = clamp(c + (1.0 - c) * p_star, 1e-6, 1.0 - 1e-6)

        # Responsibility: on failure, the weakest KU receives most of the blame.
        logits = [-self.cfg.tau * effective[k] for k in task.ku_ids]
        resp = softmax(logits, temperature=1.0)
        responsibilities = {k: float(resp[i]) for i, k in enumerate(task.ku_ids)}
        if weights:
            w = [max(0.0, x) for x in weights]
            total = sum(w)
            if total > 0:
                responsibilities = {
                    k: responsibilities[k] * (w[i] / total)
                    for i, k in enumerate(task.ku_ids)
                }
                norm = sum(responsibilities.values())
                if norm > 0:
                    responsibilities = {k: v / norm for k, v in responsibilities.items()}

        weight_sum = sum(weights) if weights else float(len(sigmas))
        if weight_sum <= 0:
            weight_sum = float(len(sigmas)) or 1.0
        sigma_task = math.sqrt(
            sum(
                (weights[i] if weights else 1.0) * sigmas[i] ** 2
                for i in range(len(sigmas))
            )
            / weight_sum
        )
        return TaskPrediction(
            task_id=task.id,
            theta_task=theta_task,
            p_star=p_star,
            p_correct=p_correct,
            sigma_task=sigma_task,
            effective_thetas=effective,
            responsibilities=responsibilities,
        )

    def update(
        self,
        states: MutableMapping[int, UserKUState],
        task: Task,
        prediction: TaskPrediction,
        y_eff: float,
        now,
        *,
        repeat_same_task: bool = False,
    ) -> tuple[MasteryUpdate, ...]:
        if not task.ku_ids:
            return ()

        p = clamp(prediction.p_correct, 1e-6, 1.0 - 1e-6)
        c = clamp(task.c_guess, 0.0, 0.99)
        p_star = clamp(prediction.p_star, 1e-6, 1.0 - 1e-6)
        g_i = clamp(safe_div((1.0 - c) * p_star, p, 1.0), 0.0, 1.0)
        delta = clamp(y_eff, 0.0, 1.0) - p
        updates: list[MasteryUpdate] = []

        for ku_id in task.ku_ids:
            st = self._ensure_state(states, ku_id)
            w = max(0.0, prediction.responsibilities.get(ku_id, 0.0))
            before_theta = st.theta
            before_sigma = st.sigma

            step = (
                self.cfg.learning_rate
                * (st.sigma ** 2)
                * task.a_discrimination
                * g_i
                * w
                * delta
            )
            if repeat_same_task:
                step *= self.cfg.repeat_task_theta_discount
            st.theta = clamp(before_theta + step, self.cfg.theta_min, self.cfg.theta_max)

            info_gain = (
                (task.a_discrimination ** 2)
                * (g_i ** 2)
                * (w ** 2)
                * p
                * (1.0 - p)
                * self.cfg.info_gain_scale
            )
            if repeat_same_task:
                info_gain *= self.cfg.repeat_task_theta_discount
            precision = 1.0 / max(before_sigma ** 2, 1e-9) + info_gain
            st.sigma = clamp(
                1.0 / math.sqrt(max(precision, 1e-9)),
                self.cfg.sigma_min,
                self.cfg.sigma_max,
            )
            st.attempts_count += 1
            st.last_seen_at = now

            updates.append(
                MasteryUpdate(
                    ku_id=ku_id,
                    before_theta=before_theta,
                    after_theta=st.theta,
                    before_sigma=before_sigma,
                    after_sigma=st.sigma,
                    responsibility=w,
                )
            )
        return tuple(updates)

    def mastery_probability(self, state: UserKUState, now) -> float:
        return sigmoid(self.effective_theta(state, now))

    def section_ability(
        self,
        states: Mapping[int, UserKUState],
        section_ku_ids: Sequence[int],
        now,
    ) -> float:
        if not section_ku_ids:
            return 0.0
        values = []
        for ku_id in section_ku_ids:
            st = states.get(ku_id)
            values.append(self.effective_theta(st, now) if st else 0.0)
        return sum(values) / len(values)
