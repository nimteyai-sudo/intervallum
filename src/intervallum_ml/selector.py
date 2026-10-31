"""Pedagogical task selection.

This is deliberately a deterministic, explainable scorer rather than a generic
contextual bandit.  The objective is learning per minute, not immediate
correctness.  Exploration is a small controlled term, and every decision logs
the score components so that policy value can later be estimated offline.
"""

from __future__ import annotations

import math
from typing import Mapping, MutableMapping

import numpy as np

from .config import SelectorConfig
from .irt import MasteryEngine
from .mathutils import clamp, gumbel_noise, sigmoid, softmax
from .types import (
    KnowledgeUnit,
    PolicyReason,
    SelectionContext,
    SelectionResult,
    Task,
    TaskScore,
    UserKUState,
)


class PedagogicalSelector:
    def __init__(
        self,
        cfg: SelectorConfig,
        knowledge_units: Mapping[int, KnowledgeUnit],
        tasks: Mapping[int, Task],
        mastery: MasteryEngine,
    ) -> None:
        self.cfg = cfg
        self.ku = knowledge_units
        self.tasks = tasks
        self.mastery = mastery

    # ------------------------------------------------------------------
    # Candidate filtering
    # ------------------------------------------------------------------
    def _prerequisites_ok(
        self,
        task: Task,
        states: Mapping[int, UserKUState],
        now,
    ) -> bool:
        for ku_id in task.ku_ids:
            ku = self.ku.get(ku_id)
            if ku is None:
                continue
            for parent_id in ku.parent_ids:
                parent = states.get(parent_id)
                if parent is None or parent.attempts_count < 3:
                    continue
                if sigmoid(self.mastery.effective_theta(parent, now)) < self.cfg.prerequisite_success:
                    return False
        return True

    def _candidates(
        self,
        states: Mapping[int, UserKUState],
        ctx: SelectionContext,
    ) -> list[Task]:
        recent = set(ctx.recent_task_ids)
        all_tasks = [t for t in self.tasks.values() if not t.is_quarantined]
        filtered: list[Task] = []
        for task in all_tasks:
            if task.id in ctx.teacher_assignment_ids:
                # A teacher assignment is never silently dropped because of a
                # cooldown or a prerequisite gate; the UI can add scaffolding.
                filtered.append(task)
                continue
            if task.id in recent:
                continue
            if not self._prerequisites_ok(task, states, ctx.now):
                continue
            filtered.append(task)
        if filtered:
            return filtered

        # Never leave the student without a task: fall back to all non-quarantined
        # tasks, preferring ones that were not just shown.
        fallback = [t for t in all_tasks if t.id not in recent] or all_tasks
        if not fallback:
            raise ValueError("No tasks available")
        return fallback

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def _p_target(self, ctx: SelectionContext) -> float:
        if ctx.session_position <= 1:
            return self.cfg.p_target_warmup
        if (
            ctx.consecutive_errors >= self.cfg.fatigue_after_errors
            or ctx.session_elapsed_sec >= self.cfg.fatigue_after_seconds
        ):
            return self.cfg.p_target_fatigue
        return self.cfg.p_target_base

    def _exam_urgency(self, theta: float, days_to_exam: int) -> float:
        if days_to_exam > 90:
            return 1.0
        if days_to_exam > 45:
            return 1.0
        # Sprint mode: push quick wins, deprioritise hopeless hard topics.
        if theta < -1.0:
            return 0.2
        if -0.5 <= theta <= 0.5:
            return 2.0
        return 1.0

    def _diversity_penalty(self, task: Task, ctx: SelectionContext) -> float:
        penalty = 0.0
        mechanics = ctx.recent_mechanics
        if mechanics:
            if mechanics[-1] == task.mechanic:
                penalty += self.cfg.diversity_same_mechanic
            if len(mechanics) >= 2 and mechanics[-2] == task.mechanic:
                penalty += self.cfg.diversity_two_mechanics
        if ctx.recent_ku_sets and set(task.ku_ids) == set(ctx.recent_ku_sets[-1]):
            penalty += self.cfg.diversity_same_kus
        return penalty

    def score_task(
        self,
        task: Task,
        states: Mapping[int, UserKUState],
        ctx: SelectionContext,
        rng: np.random.Generator,
    ) -> TaskScore:
        pred = self.mastery.predict(states, task, ctx.now)  # type: ignore[arg-type]
        p_target = self._p_target(ctx)
        zpd = math.exp(-((pred.p_star - p_target) ** 2) / (2.0 * self.cfg.zpd_sigma ** 2))

        memory = 0.0
        for ku_id in task.ku_ids:
            st = states.get(ku_id)
            if st is None or st.memory is None or not st.memory.is_started():
                continue
            r = self.mastery.memory.retrievability(st.memory, ctx.now)
            if r < self.cfg.memory_threshold:
                need = clamp(
                    (0.95 - r) / self.cfg.memory_scale,
                    0.0,
                    self.cfg.memory_cap,
                )
                memory = max(memory, need)

        exam_num = 0.0
        exam_den = 0.0
        for i, ku_id in enumerate(task.ku_ids):
            ku = self.ku.get(ku_id)
            if ku is None:
                continue
            st = states.get(ku_id)
            theta = self.mastery.effective_theta(st, ctx.now) if st else 0.0
            mastery = sigmoid(theta)
            weight = task.weights()[i] if task.weights() else 1.0
            value = max(0.0, ku.ege_weight) * weight
            urgency = self._exam_urgency(theta, ctx.days_to_exam)
            exam_num += value * (1.0 - mastery) * urgency
            exam_den += value
        exam = exam_num / exam_den if exam_den > 0 else 0.0

        diversity = self._diversity_penalty(task, ctx)
        sigma_mean = 0.0
        if task.ku_ids:
            sigma_mean = sum(
                states[k].sigma if k in states else 1.0 for k in task.ku_ids
            ) / len(task.ku_ids)
        uncertainty = self.cfg.uncertainty_bonus * sigma_mean
        noise = float(gumbel_noise(rng, 1)[0]) * ctx.exploration_scale
        exploration = uncertainty + noise

        base_utility = (
            self.cfg.w_zpd * zpd
            + self.cfg.w_memory * memory
            + self.cfg.w_exam * exam
            - self.cfg.w_diversity * diversity
            + uncertainty
        )
        if task.id in ctx.teacher_assignment_ids:
            base_utility += self.cfg.teacher_bonus
        utility = base_utility + noise

        # Hard-ish review quota: after the quota is exhausted, review-heavy
        # tasks are filtered out in ``select`` when non-review alternatives
        # exist.  Keeping the raw score here makes ``rank`` diagnostic.
        reason = self._reason(
            task, ctx, zpd, memory, exam, exploration
        )
        components = {
            "zpd": zpd,
            "memory": memory,
            "exam": exam,
            "diversity": diversity,
            "exploration": exploration,
            "sigma_mean": sigma_mean,
        }
        return TaskScore(
            task_id=task.id,
            utility=float(utility),
            zpd=float(zpd),
            memory=float(memory),
            exam=float(exam),
            diversity_penalty=float(diversity),
            exploration=float(exploration),
            p_correct=float(pred.p_correct),
            p_star=float(pred.p_star),
            theta_task=float(pred.theta_task),
            reason=reason,
            components=components,
            base_utility=float(base_utility),
        )

    def _reason(
        self,
        task: Task,
        ctx: SelectionContext,
        zpd: float,
        memory: float,
        exam: float,
        exploration: float,
    ) -> PolicyReason:
        if task.id in ctx.teacher_assignment_ids:
            return PolicyReason.TEACHER_ASSIGNMENT
        weighted = {
            PolicyReason.REVIEW_DUE: self.cfg.w_memory * memory,
            PolicyReason.WEAK_SKILL: self.cfg.w_zpd * zpd,
            PolicyReason.EXAM_VALUE: self.cfg.w_exam * exam,
            PolicyReason.EXPLORATION: exploration,
        }
        best = max(weighted, key=weighted.get)
        if weighted[best] <= 0:
            return PolicyReason.BALANCED
        if best == PolicyReason.REVIEW_DUE and memory <= 0.4:
            return PolicyReason.WEAK_SKILL if weighted[PolicyReason.WEAK_SKILL] > 0 else PolicyReason.BALANCED
        if best == PolicyReason.EXAM_VALUE and exam <= 0.25:
            return PolicyReason.WEAK_SKILL if weighted[PolicyReason.WEAK_SKILL] > 0 else PolicyReason.BALANCED
        return best

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def rank(
        self,
        states: MutableMapping[int, UserKUState],
        ctx: SelectionContext,
    ) -> list[TaskScore]:
        candidates = self._candidates(states, ctx)
        rng = np.random.default_rng(ctx.seed)
        scores = [self.score_task(t, states, ctx, rng) for t in candidates]
        scores.sort(key=lambda s: s.utility, reverse=True)
        return scores

    def select(
        self,
        states: MutableMapping[int, UserKUState],
        ctx: SelectionContext,
    ) -> SelectionResult:
        scores = self.rank(states, ctx)
        if not scores:
            raise ValueError("No candidate tasks")
        quota = max(1, int(round(self.cfg.max_review_share * max(ctx.session_length, 1))))
        eligible = scores
        if ctx.review_count >= quota:
            non_review = [s for s in scores if s.memory <= 0.4]
            if non_review:
                eligible = non_review
        chosen = eligible[0]
        if ctx.exploration_scale <= 1e-9:
            propensity = 1.0
        else:
            # Gumbel-max: argmax(base + scale * Gumbel) samples from
            # softmax(base / scale).  This is the exact logging propensity.
            base = np.asarray([s.base_utility for s in eligible], dtype=float)
            probs = softmax(base.tolist(), temperature=ctx.exploration_scale)
            propensity = float(probs[0])
        task = self.tasks[chosen.task_id]
        return SelectionResult(
            task=task,
            score=chosen,
            propensity=propensity,
            ranked=tuple(scores),
            candidates=len(scores),
        )
