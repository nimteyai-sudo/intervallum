"""High-level orchestration of the Intervallum adaptive engine.

The engine is deliberately synchronous and dependency-light.  In production the
same logic can run in a TypeScript fast path; this Python implementation is the
reference behaviour and the source of truth for tests, simulations and the
offline calibration pipeline.
"""

from __future__ import annotations

from datetime import datetime
from typing import Iterable, Mapping, MutableMapping

from .config import EngineConfig
from .irt import MasteryEngine
from .memory import MemoryScheduler
from .readiness import ExamReadinessProjector
from .selector import PedagogicalSelector
from .telemetry import scrub_attempt
from .types import (
    Attempt,
    AttemptResult,
    ExamBlueprintItem,
    KnowledgeUnit,
    KUType,
    MemoryState,
    MemoryUpdate,
    ReadinessProjection,
    SelectionContext,
    SelectionResult,
    StateSnapshot,
    Task,
    UserKUState,
)


class IntervallumEngine:
    def __init__(
        self,
        knowledge_units: Mapping[int, KnowledgeUnit],
        tasks: Mapping[int, Task],
        cfg: EngineConfig | None = None,
    ) -> None:
        self.ku = dict(knowledge_units)
        self.tasks = dict(tasks)
        self.cfg = cfg or EngineConfig()
        self.memory = MemoryScheduler(self.cfg.memory)
        self.mastery = MasteryEngine(self.cfg.mastery, self.memory)
        self.selector = PedagogicalSelector(
            cfg=self.cfg.selector,
            knowledge_units=self.ku,
            tasks=self.tasks,
            mastery=self.mastery,
        )
        self.readiness = ExamReadinessProjector(self.cfg.readiness, self.mastery)

    # ------------------------------------------------------------------
    # State creation and snapshots
    # ------------------------------------------------------------------
    def new_user_state(self) -> dict[int, UserKUState]:
        return {ku_id: UserKUState(ku_id=ku_id) for ku_id in self.ku}

    def snapshot(self, states: Mapping[int, UserKUState], now: datetime) -> dict[int, StateSnapshot]:
        out: dict[int, StateSnapshot] = {}
        for ku_id, st in states.items():
            r = 0.0
            if st.memory is not None and st.memory.is_started():
                r = self.memory.retrievability(st.memory, now)
            out[ku_id] = StateSnapshot(
                theta=st.theta,
                sigma=st.sigma,
                stability=st.memory.stability if st.memory else 0.0,
                difficulty=st.memory.difficulty if st.memory else 0.0,
                retrievability=r,
            )
        return out

    # ------------------------------------------------------------------
    # Attempt processing
    # ------------------------------------------------------------------
    def review_attempt(
        self,
        states: MutableMapping[int, UserKUState],
        task: Task,
        attempt: Attempt,
    ) -> AttemptResult:
        now = attempt.timestamp
        telemetry = scrub_attempt(
            raw_score=attempt.raw_score,
            duration_ms=attempt.duration_ms,
            self_grade=attempt.self_grade,
            task=task,
            hint_weight_sum=attempt.hint_weight_sum,
            cfg=self.cfg.telemetry,
        )
        prediction = self.mastery.predict(states, task, now)
        mastery_updates = self.mastery.update(
            states,
            task,
            prediction,
            telemetry.y_eff,
            now,
            repeat_same_task=attempt.repeat_same_task,
        )
        memory_updates = self._update_memory(
            states, task, attempt, telemetry.g_sync, prediction.responsibilities, now
        )
        return AttemptResult(
            telemetry=telemetry,
            mastery_updates=mastery_updates,
            memory_updates=memory_updates,
            p_correct_before=prediction.p_correct,
            p_star_before=prediction.p_star,
            diagnostics={
                "theta_task": prediction.theta_task,
                "sigma_task": prediction.sigma_task,
                "y_eff": telemetry.y_eff,
                "g_sync": float(telemetry.g_sync),
            },
        )

    def _update_memory(
        self,
        states: MutableMapping[int, UserKUState],
        task: Task,
        attempt: Attempt,
        rating,
        responsibilities: Mapping[int, float],
        now: datetime,
    ) -> tuple[MemoryUpdate, ...]:
        updates: list[MemoryUpdate] = []
        for ku_id in task.ku_ids:
            ku = self.ku.get(ku_id)
            ku_type = ku.ku_type if ku else KUType.HYBRID
            if ku_type == KUType.PROCEDURAL:
                continue
            if responsibilities.get(ku_id, 0.0) < self.cfg.mastery.memory_update_threshold:
                continue
            st = states.get(ku_id)
            if st is None:
                st = UserKUState(ku_id=ku_id)
                states[ku_id] = st
            before = st.memory or MemoryState()
            r_before = self.memory.retrievability(before, now) if before.is_started() else 0.0
            after = self.memory.review(before, rating, now)
            r_after = self.memory.retrievability(after, now)
            st.memory = after
            updates.append(
                MemoryUpdate(
                    ku_id=ku_id,
                    rating=rating,
                    before=before,
                    after=after,
                    retrievability_before=r_before,
                    retrievability_after=r_after,
                    reviewed_at=now,
                )
            )
        return tuple(updates)

    # ------------------------------------------------------------------
    # Selection and forecasting
    # ------------------------------------------------------------------
    def select(
        self,
        states: MutableMapping[int, UserKUState],
        context: SelectionContext | None = None,
    ) -> SelectionResult:
        return self.selector.select(states, context or SelectionContext())

    def rank(
        self,
        states: MutableMapping[int, UserKUState],
        context: SelectionContext | None = None,
    ):
        return self.selector.rank(states, context or SelectionContext())

    def readiness_projection(
        self,
        states: Mapping[int, UserKUState],
        blueprint: Iterable[ExamBlueprintItem],
        exam_date: datetime,
        now: datetime | None = None,
        scale_points: Iterable[tuple[float, float]] | None = None,
    ) -> ReadinessProjection:
        return self.readiness.project(states, blueprint, exam_date, now, scale_points)

    # ------------------------------------------------------------------
    # Cold start
    # ------------------------------------------------------------------
    def apply_ku_cold_start(
        self,
        states: MutableMapping[int, UserKUState],
        new_ku_id: int,
    ) -> UserKUState:
        ku = self.ku.get(new_ku_id)
        if ku is None:
            raise KeyError(f"Unknown knowledge unit {new_ku_id}")
        parent_values = [states[p].theta for p in ku.parent_ids if p in states]
        neighbor_values = [states[p].theta for p in ku.neighbor_ids if p in states]
        if parent_values and neighbor_values:
            theta = 0.5 * (sum(parent_values) / len(parent_values)) + 0.5 * (
                sum(neighbor_values) / len(neighbor_values)
            )
        elif parent_values:
            theta = sum(parent_values) / len(parent_values)
        elif neighbor_values:
            theta = sum(neighbor_values) / len(neighbor_values)
        else:
            theta = 0.0
        st = UserKUState(ku_id=new_ku_id, theta=theta, sigma=0.85)
        states[new_ku_id] = st
        return st
