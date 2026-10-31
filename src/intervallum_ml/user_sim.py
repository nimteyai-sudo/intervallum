"""Behavioural student simulator for end-to-end validation without real users.

The simulator is deliberately more than a random-response generator.  Each
simulated student has:

* latent knowledge per knowledge unit;
* memory stability and forgetting;
* motivation, fatigue and anxiety;
* hint propensity, guessing propensity and self-rating bias;
* a response-time model;
* a learning rule with a Zone of Proximal Development peak;
* optional dropout.

It can run on the synthetic world or on the real EGE content bank.  It is a
logic and robustness test, not evidence of real educational efficacy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Sequence

import numpy as np

from .config import EngineConfig
from .content import ContentBank
from .engine import IntervallumEngine
from .mathutils import brier_score, clamp, expected_calibration_error, sigmoid, softmin
from .simulation import SyntheticWorld
from .types import (
    Attempt,
    Rating,
    SelectionContext,
    Task,
    UserKUState,
)


@dataclass(frozen=True)
class StudentProfile:
    user_id: str
    ability_mean: float = 0.0
    ability_sd: float = 0.8
    motivation: float = 0.75
    anxiety: float = 0.3
    hint_propensity: float = 0.2
    guess_propensity: float = 0.05
    overconfidence: float = 0.15
    fatigue_rate: float = 0.03
    forgetting_multiplier: float = 1.0
    device_width: int = 390
    label: str = "typical"


@dataclass(frozen=True)
class SimulatedResponse:
    task_id: int
    p_base: float
    p_effective: float
    correct: bool
    hint_weight: float
    duration_ms: int
    self_grade: Rating
    raw_score: float


@dataclass
class SimulatedStudent:
    profile: StudentProfile
    true_theta: dict[int, float]
    true_stability: dict[int, float]
    last_review: dict[int, datetime]
    engine_states: dict[int, UserKUState]
    fatigue: float = 0.0
    motivation: float = 0.75
    history: list[SimulatedResponse] = field(default_factory=list)

    def true_retrievability(self, ku_id: int, now: datetime) -> float:
        last = self.last_review.get(ku_id)
        if last is None:
            return 1.0
        days = max(0.0, (now - last).total_seconds() / 86400.0)
        stability = max(0.1, self.true_stability.get(ku_id, 10.0))
        stability *= self.profile.forgetting_multiplier
        return float(0.9 ** (days / stability))

    def true_effective_theta(self, ku_id: int, now: datetime, gamma: float) -> float:
        r = clamp(self.true_retrievability(ku_id, now), 0.01, 1.0)
        return self.true_theta[ku_id] + gamma * math.log(r)

    def predict_true(
        self,
        task: Task,
        world: SyntheticWorld,
        now: datetime,
    ) -> float:
        vals = [
            self.true_effective_theta(k, now, world.true_gamma)
            for k in task.ku_ids
        ]
        theta_task = softmin(vals, world.true_tau, task.weights())
        p_star = sigmoid(task.a_discrimination * (theta_task - task.b_difficulty))
        return clamp(task.c_guess + (1 - task.c_guess) * p_star, 1e-6, 1 - 1e-6)

    def respond(
        self,
        task: Task,
        world: SyntheticWorld,
        now: datetime,
        rng: np.random.Generator,
    ) -> SimulatedResponse:
        p_base = self.predict_true(task, world, now)
        hint_probability = clamp(
            self.profile.hint_propensity
            * (1.0 - p_base)
            * (0.5 + self.profile.anxiety)
            * (1.3 - 0.4 * self.profile.motivation),
            0.0,
            1.0,
        )
        hint_weight = 0.0
        if rng.random() < hint_probability:
            hint_weight = float(rng.choice([0.25, 0.40, 0.60]))
        p_after_hint = clamp(p_base + hint_weight * 0.55, 0.0, 0.99)
        # A guesser moves toward the guessing floor.
        p_effective = clamp(
            (1.0 - self.profile.guess_propensity) * p_after_hint
            + self.profile.guess_propensity * task.c_guess,
            1e-6,
            1 - 1e-6,
        )
        correct = bool(rng.random() < p_effective)

        ratio = (
            (0.8 + 1.4 * (1.0 - p_base))
            * (1.0 + 0.35 * self.fatigue)
            * float(rng.lognormal(0.0, 0.25))
        )
        if hint_weight > 0:
            ratio += 0.35
        duration_ms = int(
            min(600_000, max(2_000, task.t_expected_sec * ratio * 1000))
        )

        if correct and ratio < 0.7:
            grade = Rating.EASY
        elif correct and ratio < 1.8:
            grade = Rating.GOOD
        elif correct:
            grade = Rating.HARD
        else:
            grade = Rating.AGAIN
        if rng.random() < self.profile.overconfidence:
            grade = Rating(min(4, int(grade) + 1))
        if rng.random() < self.profile.anxiety * 0.35:
            grade = Rating(max(1, int(grade) - 1))

        return SimulatedResponse(
            task_id=task.id,
            p_base=p_base,
            p_effective=p_effective,
            correct=correct,
            hint_weight=hint_weight,
            duration_ms=duration_ms,
            self_grade=grade,
            raw_score=1.0 if correct else 0.0,
        )

    def learn(
        self,
        task: Task,
        response: SimulatedResponse,
        world: SyntheticWorld,
        now: datetime,
    ) -> None:
        zpd = math.exp(-((response.p_base - 0.70) ** 2) / (2 * 0.12 ** 2))
        gain = (
            0.040
            * zpd
            * (1.0 - 0.5 * response.hint_weight)
            * self.motivation
            * (1.0 - 0.25 * self.fatigue)
        )
        if not response.correct:
            gain *= 0.30
        weights = task.weights() or tuple(1.0 for _ in task.ku_ids)
        total = sum(weights) or 1.0
        for i, ku_id in enumerate(task.ku_ids):
            share = weights[i] / total
            self.true_theta[ku_id] = clamp(
                self.true_theta[ku_id] + gain * share,
                -5.0,
                5.0,
            )
            if response.correct:
                self.true_stability[ku_id] = min(
                    365.0,
                    max(0.5, self.true_stability[ku_id] * (1.0 + 0.35 * zpd)),
                )
            else:
                self.true_stability[ku_id] = max(
                    0.5, self.true_stability[ku_id] * 0.85
                )
            self.last_review[ku_id] = now

        self.fatigue = clamp(
            self.fatigue
            + self.profile.fatigue_rate
            + 0.03 * (1.0 - response.p_base)
            + 0.02 * response.hint_weight,
            0.0,
            2.0,
        )
        self.motivation = clamp(
            self.motivation - 0.005 * (1.0 - response.p_base),
            0.15,
            1.0,
        )


@dataclass(frozen=True)
class SessionResult:
    user_id: str
    profile: str
    policy: str
    attempts: int
    completed: bool
    true_gain: float
    estimated_gain: float
    mean_true_p: float
    mean_estimated_p: float
    hint_rate: float
    mean_duration_sec: float
    true_retention: float
    estimated_retention: float
    brier: float
    ece: float
    theta_mae: float


def _init_student(
    user_id: str,
    profile: StudentProfile,
    world: SyntheticWorld,
    engine: IntervallumEngine,
    initial_noise: float,
    rng: np.random.Generator,
) -> SimulatedStudent:
    true_theta = {
        ku_id: world.true_theta[(user_id, ku_id)]
        for ku_id in world.knowledge_units
    }
    true_stability = {
        ku_id: world.true_stability[(user_id, ku_id)]
        for ku_id in world.knowledge_units
    }
    states = engine.new_user_state()
    for ku_id, st in states.items():
        # A good diagnostic gives a noisy but per-KU estimate, not one global
        # mean copied to every unit.
        st.theta = true_theta[ku_id] + float(rng.normal(0, initial_noise))
        st.sigma = 1.0
    return SimulatedStudent(
        profile=profile,
        true_theta=true_theta,
        true_stability=true_stability,
        last_review={},
        engine_states=states,
        motivation=profile.motivation,
    )


def _choose_task(
    policy: str,
    engine: IntervallumEngine,
    student: SimulatedStudent,
    world: SyntheticWorld,
    ctx: SelectionContext,
    recent: Sequence[int],
    rng: np.random.Generator,
) -> Task:
    tasks = list(world.tasks.values())
    if policy == "adaptive":
        return engine.select(student.engine_states, ctx).task
    if policy == "random":
        pool = [t for t in tasks if t.id not in recent] or tasks
        return pool[int(rng.integers(len(pool)))]
    if policy in {"easy", "hard"}:
        idx = rng.choice(len(tasks), size=min(60, len(tasks)), replace=False)
        pool = [tasks[int(i)] for i in idx]
        preds = [
            (t, engine.mastery.predict(student.engine_states, t, ctx.now).p_correct)
            for t in pool
        ]
        if policy == "easy":
            return max(preds, key=lambda x: x[1])[0]
        # Avoid impossible tasks: aim for the lowest p above 0.15.
        viable = [x for x in preds if x[1] >= 0.15] or preds
        return min(viable, key=lambda x: x[1])[0]
    if policy == "topic_order":
        ordered = sorted(tasks, key=lambda t: (t.topic_id or 0, t.id))
        start = len(recent) % len(ordered)
        return ordered[start]
    raise ValueError(f"Unknown policy: {policy}")


def simulate_student(
    world: SyntheticWorld,
    profile: StudentProfile,
    *,
    policy: str = "adaptive",
    sessions: int = 4,
    tasks_per_session: int = 10,
    days_between_sessions: int = 7,
    initial_noise: float = 1.0,
    seed: int = 42,
    engine_config: EngineConfig | None = None,
) -> SessionResult:
    rng = np.random.default_rng(seed)
    engine = IntervallumEngine(
        world.knowledge_units, world.tasks, engine_config or EngineConfig()
    )
    student = _init_student(
        profile.user_id, profile, world, engine, initial_noise, rng
    )
    initial_theta = dict(student.true_theta)
    initial_estimated = {
        k: st.theta for k, st in student.engine_states.items()
    }
    start = datetime(2026, 9, 1, tzinfo=timezone.utc)
    true_ps: list[float] = []
    estimated_ps: list[float] = []
    hints = 0
    durations: list[float] = []
    recent: list[int] = []
    attempts = 0
    completed_sessions = 0
    for session in range(sessions):
        student.fatigue *= 0.5
        now = start + timedelta(days=session * days_between_sessions)
        session_completed = True
        for step in range(tasks_per_session):
            ctx = SelectionContext(
                now=now + timedelta(minutes=step * 4),
                session_position=step,
                session_length=tasks_per_session,
                consecutive_errors=0,
                session_elapsed_sec=float(step * 240),
                days_to_exam=max(1, (sessions - session) * days_between_sessions),
                recent_task_ids=tuple(recent[-7:]),
                device_width=profile.device_width,
                seed=seed * 1000 + session * 100 + step,
            )
            task = _choose_task(
                policy, engine, student, world, ctx, recent, rng
            )
            response = student.respond(task, world, ctx.now, rng)
            student.learn(task, response, world, ctx.now)
            snapshot = engine.snapshot(student.engine_states, ctx.now)
            attempt = Attempt(
                user_id=profile.user_id,
                task_id=task.id,
                session_id=f"{profile.user_id}-s{session}",
                timestamp=ctx.now,
                raw_score=response.raw_score,
                duration_ms=response.duration_ms,
                self_grade=response.self_grade,
                hint_weight_sum=response.hint_weight,
                hints_used=(1,) if response.hint_weight else (),
                pos_in_session=step,
                session_elapsed_sec=ctx.session_elapsed_sec,
                state_before=snapshot,
                repeat_same_task=task.id in recent,
            )
            engine.review_attempt(student.engine_states, task, attempt)
            true_ps.append(response.p_base)
            estimated_ps.append(
                engine.mastery.predict(student.engine_states, task, ctx.now).p_correct
            )
            hints += 1 if response.hint_weight else 0
            durations.append(response.duration_ms / 1000.0)
            recent.append(task.id)
            student.history.append(response)
            attempts += 1
            if student.fatigue > 1.0 and rng.random() < 0.20:
                session_completed = False
                break
        if session_completed:
            completed_sessions += 1

    final_now = start + timedelta(days=sessions * days_between_sessions)
    true_gain = sum(
        student.true_theta[k] - initial_theta[k] for k in world.knowledge_units
    ) / max(len(world.knowledge_units), 1)
    estimated_gain = sum(
        student.engine_states[k].theta - initial_estimated[k]
        for k in world.knowledge_units
    ) / max(len(world.knowledge_units), 1)
    true_retention = sum(
        student.true_retrievability(k, final_now) for k in world.knowledge_units
    ) / max(len(world.knowledge_units), 1)
    estimated_retention = sum(
        (
            engine.memory.retrievability(student.engine_states[k].memory, final_now)
            if student.engine_states[k].memory is not None
            and student.engine_states[k].memory.is_started()
            else 0.0
        )
        for k in world.knowledge_units
    ) / max(len(world.knowledge_units), 1)
    theta_mae = sum(
        abs(student.engine_states[k].theta - student.true_theta[k])
        for k in world.knowledge_units
    ) / max(len(world.knowledge_units), 1)
    return SessionResult(
        user_id=profile.user_id,
        profile=profile.label,
        policy=policy,
        attempts=attempts,
        completed=completed_sessions == sessions,
        true_gain=true_gain,
        estimated_gain=estimated_gain,
        mean_true_p=sum(true_ps) / len(true_ps) if true_ps else 0.0,
        mean_estimated_p=(
            sum(estimated_ps) / len(estimated_ps) if estimated_ps else 0.0
        ),
        hint_rate=hints / max(attempts, 1),
        mean_duration_sec=(
            sum(durations) / len(durations) if durations else 0.0
        ),
        true_retention=true_retention,
        estimated_retention=estimated_retention,
        brier=brier_score(true_ps, estimated_ps),
        ece=expected_calibration_error(true_ps, estimated_ps),
        theta_mae=theta_mae,
    )


def world_from_content(
    bank: ContentBank,
    *,
    n_users: int = 20,
    seed: int = 42,
) -> SyntheticWorld:
    """Create a latent ground truth over the real content bank.

    The true item difficulties start from the imported heuristic priors and are
    perturbed so that the simulator is not merely reproducing the same
    parameters the engine uses.
    """

    rng = np.random.default_rng(seed)
    users = tuple(f"real_u{i}" for i in range(n_users))
    true_theta: dict[tuple[str, int], float] = {}
    true_stability: dict[tuple[str, int], float] = {}
    for user in users:
        base = float(rng.normal(0, 0.8))
        for ku_id in bank.knowledge_units:
            true_theta[(user, ku_id)] = float(base + rng.normal(0, 0.8))
            true_stability[(user, ku_id)] = float(rng.uniform(2.0, 30.0))
    true_b = {
        task_id: float(task.b_difficulty + rng.normal(0, 0.25))
        for task_id, task in bank.tasks.items()
    }
    true_a = {
        task_id: float(max(0.5, min(2.5, task.a_discrimination + rng.normal(0, 0.1))))
        for task_id, task in bank.tasks.items()
    }
    return SyntheticWorld(
        knowledge_units=dict(bank.knowledge_units),
        tasks=dict(bank.tasks),
        true_theta=true_theta,
        true_stability=true_stability,
        true_b=true_b,
        true_a=true_a,
        true_tau=2.0,
        true_gamma=0.8,
        users=users,
    )


def make_profiles(world: SyntheticWorld, seed: int = 42) -> list[StudentProfile]:
    rng = np.random.default_rng(seed)
    profiles = []
    for user in world.users:
        mean = sum(
            world.true_theta[(user, k)] for k in world.knowledge_units
        ) / len(world.knowledge_units)
        profiles.append(
            StudentProfile(
                user_id=user,
                ability_mean=mean,
                motivation=float(rng.uniform(0.50, 0.95)),
                anxiety=float(rng.uniform(0.05, 0.80)),
                hint_propensity=float(rng.uniform(0.05, 0.45)),
                guess_propensity=float(rng.uniform(0.0, 0.20)),
                overconfidence=float(rng.uniform(0.0, 0.35)),
                fatigue_rate=float(rng.uniform(0.015, 0.05)),
                forgetting_multiplier=float(rng.uniform(0.7, 1.3)),
                label="typical",
            )
        )
    return profiles


def make_edge_profiles(user_id: str = "edge") -> list[StudentProfile]:
    common = dict(
        user_id=user_id,
        motivation=0.75,
        anxiety=0.3,
        hint_propensity=0.2,
        guess_propensity=0.05,
        overconfidence=0.15,
        fatigue_rate=0.03,
        forgetting_multiplier=1.0,
    )
    return [
        StudentProfile(**{**common, "label": "strong", "motivation": 0.95, "anxiety": 0.1}),  # type: ignore[arg-type]
        StudentProfile(**{**common, "label": "weak", "motivation": 0.55, "anxiety": 0.7}),  # type: ignore[arg-type]
        StudentProfile(**{**common, "label": "anxious", "anxiety": 0.95, "overconfidence": 0.0}),  # type: ignore[arg-type]
        StudentProfile(**{**common, "label": "guesser", "guess_propensity": 0.45, "motivation": 0.35}),  # type: ignore[arg-type]
        StudentProfile(**{**common, "label": "hint_abuser", "hint_propensity": 0.9}),  # type: ignore[arg-type]
        StudentProfile(**{**common, "label": "overconfident", "overconfidence": 0.9}),  # type: ignore[arg-type]
        StudentProfile(**{**common, "label": "fatigued", "fatigue_rate": 0.12, "motivation": 0.4}),  # type: ignore[arg-type]
    ]


def run_cohort(
    world: SyntheticWorld,
    profiles: Sequence[StudentProfile],
    *,
    policy: str,
    sessions: int = 4,
    tasks_per_session: int = 10,
    days_between_sessions: int = 7,
    initial_noise: float = 1.0,
    seed: int = 42,
) -> list[SessionResult]:
    return [
        simulate_student(
            world,
            p,
            policy=policy,
            sessions=sessions,
            tasks_per_session=tasks_per_session,
            days_between_sessions=days_between_sessions,
            initial_noise=initial_noise,
            seed=seed + i,
        )
        for i, p in enumerate(profiles)
    ]


def summarise(results: Sequence[SessionResult]) -> dict:
    n = len(results)
    if n == 0:
        return {}
    return {
        "students": n,
        "mean_true_gain": sum(r.true_gain for r in results) / n,
        "mean_estimated_gain": sum(r.estimated_gain for r in results) / n,
        "mean_true_p": sum(r.mean_true_p for r in results) / n,
        "mean_estimated_p": sum(r.mean_estimated_p for r in results) / n,
        "mean_hint_rate": sum(r.hint_rate for r in results) / n,
        "mean_duration_sec": sum(r.mean_duration_sec for r in results) / n,
        "mean_true_retention": sum(r.true_retention for r in results) / n,
        "mean_estimated_retention": sum(r.estimated_retention for r in results) / n,
        "mean_brier": sum(r.brier for r in results) / n,
        "mean_ece": sum(r.ece for r in results) / n,
        "mean_theta_mae": sum(r.theta_mae for r in results) / n,
        "completion_rate": sum(1 for r in results if r.completed) / n,
    }
