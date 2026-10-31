"""Cold-start utilities: meta-difficulty and adaptive placement.

At launch there are no attempt logs.  The system therefore needs:

* a meta-model that predicts item difficulty from task metadata;
* a short adaptive placement test that gives every knowledge unit a reasonable
  prior instead of starting at zero;
* cold-start rules for new tasks and new knowledge units.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, MutableMapping, Sequence

import numpy as np

from .engine import IntervallumEngine
from .irt import MasteryEngine
from .mathutils import clamp
from .types import (
    Attempt,
    KnowledgeUnit,
    PlacementResult,
    Task,
    UserKUState,
)

MECHANIC_PRIOR = {
    "mcq_single": -0.5,
    "mcq_multi": -0.2,
    "numeric": 0.2,
    "formula_build": 0.3,
    "formula_derive": 0.4,
    "find_error": 0.5,
    "draw_forces": 0.5,
    "dim_check": -0.5,
    "predict_change": 0.0,
    "match_order": 0.2,
    "graph_read": 0.3,
    "free_response": 0.5,
}


def task_features(task: Task) -> list[float]:
    """Feature vector used by the meta-difficulty model.

    Keep this stable: the same vector must be computed at task creation and at
    calibration time.  Categorical values are encoded as small numeric priors
    rather than a large one-hot matrix so that the model works with tiny data.
    """

    mechanic = task.mechanic.value if hasattr(task.mechanic, "value") else str(task.mechanic)
    return [
        1.0,
        (task.expert_difficulty - 3.0) / 2.0,
        float(len(task.ku_ids)),
        float(task.c_guess),
        math.log1p(max(task.t_expected_sec, 1.0)) / 5.0,
        MECHANIC_PRIOR.get(mechanic, 0.0),
        (task.topic_id or 0) / 100.0,
        (task.section_id or 0) / 10.0,
        len(task.text) / 500.0,
    ]


@dataclass
class MetaDifficultyModel:
    """Ridge regression from task metadata to IRT difficulty ``b``."""

    coefficients: np.ndarray | None = None
    ridge: float = 1.0

    def fit(self, tasks: Sequence[Task], b: Sequence[float] | None = None) -> "MetaDifficultyModel":
        if not tasks:
            self.coefficients = np.zeros(9)
            return self
        X = np.asarray([task_features(t) for t in tasks], dtype=float)
        y = np.asarray(
            b if b is not None else [(t.expert_difficulty - 3.0) * 0.8 for t in tasks],
            dtype=float,
        )
        # Do not regularise the intercept.
        penalty = np.eye(X.shape[1]) * self.ridge
        penalty[0, 0] = 0.0
        self.coefficients = np.linalg.solve(X.T @ X + penalty, X.T @ y)
        return self

    def predict(self, task: Task) -> float:
        if self.coefficients is None:
            return (task.expert_difficulty - 3.0) * 0.8
        x = np.asarray(task_features(task), dtype=float)
        return float(x @ self.coefficients)


class AdaptivePlacement:
    """Short computerized adaptive placement test.

    The goal is not to measure precisely; it is to avoid starting every student
    at ``theta = 0``.  Seven or eight items covering the four EGE sections are
    enough to set useful priors.
    """

    def __init__(
        self,
        mastery: MasteryEngine,
        section_by_ku: Mapping[int, int],
        n_items: int = 8,
    ) -> None:
        self.mastery = mastery
        self.section_by_ku = section_by_ku
        self.n_items = n_items

    def plan(
        self,
        tasks: Sequence[Task],
        states: MutableMapping[int, UserKUState],
        now,
        *,
        sections: Sequence[int] | None = None,
        exclude: set[int] | None = None,
    ) -> list[Task]:
        sections = list(sections or sorted(set(self.section_by_ku.values())))
        exclude = set(exclude or set())
        available = [t for t in tasks if not t.is_quarantined and t.id not in exclude]
        chosen: list[Task] = []
        covered: set[int] = set()

        # 1. One item per section, closest to p_star = 0.5.
        for section in sections:
            pool = [
                t
                for t in available
                if t.id not in {x.id for x in chosen}
                and any(self.section_by_ku.get(k) == section for k in t.ku_ids)
            ]
            if not pool:
                continue
            best = min(pool, key=lambda t: abs(self.mastery.predict(states, t, now).p_star - 0.5))
            chosen.append(best)
            covered.add(section)

        # 2. Fill remaining slots by maximum information, with a small
        # preference for uncovered sections.
        while len(chosen) < self.n_items:
            pool = [t for t in available if t.id not in {x.id for x in chosen}]
            if not pool:
                break
            def key(t: Task) -> float:
                pred = self.mastery.predict(states, t, now)
                info = abs(pred.p_star - 0.5)
                task_sections = {self.section_by_ku.get(k) for k in t.ku_ids}
                uncovered_bonus = -0.15 if task_sections - covered else 0.0
                return info + uncovered_bonus
            best = min(pool, key=key)
            chosen.append(best)
            covered |= {self.section_by_ku.get(k) for k in best.ku_ids}
        return chosen[: self.n_items]

    def apply(
        self,
        engine: IntervallumEngine,
        states: MutableMapping[int, UserKUState],
        task_attempts: Sequence[tuple[Task, Attempt]],
        *,
        prior_mean: float = 0.0,
        same_section_corr: float = 0.6,
        cross_section_corr: float = 0.3,
    ) -> PlacementResult:
        for task, attempt in task_attempts:
            engine.review_attempt(states, task, attempt)

        # Section means from directly tested units.
        section_values: dict[int, list[float]] = {}
        for task, _ in task_attempts:
            for ku_id in task.ku_ids:
                st = states.get(ku_id)
                if st is None:
                    continue
                section = self.section_by_ku.get(ku_id, 0)
                section_values.setdefault(section, []).append(st.theta)
        section_mean = {s: sum(v) / len(v) for s, v in section_values.items() if v}
        global_mean = (
            sum(section_mean.values()) / len(section_mean) if section_mean else prior_mean
        )

        theta_updates: dict[int, float] = {}
        for ku_id, st in states.items():
            if st.attempts_count > 0:
                continue
            section = self.section_by_ku.get(ku_id, 0)
            if section in section_mean:
                target = prior_mean + same_section_corr * (section_mean[section] - prior_mean)
            else:
                target = prior_mean + cross_section_corr * (global_mean - prior_mean)
            st.theta = clamp(target, engine.cfg.mastery.theta_min, engine.cfg.mastery.theta_max)
            st.sigma = max(st.sigma, 0.85)
            theta_updates[ku_id] = st.theta

        estimated_sections = {s: v for s, v in section_mean.items()}
        return PlacementResult(
            selected_task_ids=tuple(t.id for t, _ in task_attempts),
            theta_updates=theta_updates,
            estimated_sections=estimated_sections,
        )


def cold_start_task(task: Task, meta: MetaDifficultyModel) -> Task:
    """Return a copy of a new task with a meta-predicted difficulty."""

    from dataclasses import replace

    return replace(task, b_difficulty=meta.predict(task), a_discrimination=1.0)


def cold_start_ku(
    states: MutableMapping[int, UserKUState],
    ku: KnowledgeUnit,
) -> UserKUState:
    """Initialise a new knowledge unit from its graph neighbours."""

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
    st = UserKUState(ku_id=ku.id, theta=theta, sigma=0.85)
    states[ku.id] = st
    return st
