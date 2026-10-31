"""FastAPI reference service.

This is intentionally an in-memory adapter, not the production persistence
layer.  It demonstrates the exact request/response contract that a TypeScript
backend can implement.  Production should replace ``InMemoryStore`` with
PostgreSQL and Redis while keeping the engine calls identical.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import EngineConfig
from .content import load_engine
from .engine import IntervallumEngine
from .grading import grade_answer
from .simulation import build_world
from .types import (
    Attempt,
    ExamBlueprintItem,
    KUType,
    Rating,
    SelectionContext,
    Task,
    TaskMechanic,
    UserKUState,
)


class KUOut(BaseModel):
    id: int
    code: str
    name: str
    ku_type: KUType
    topic_id: int
    section_id: int
    ege_weight: float


class TaskOut(BaseModel):
    id: int
    mechanic: TaskMechanic
    ku_ids: list[int]
    c_guess: float
    t_expected_sec: float
    b_difficulty: float
    a_discrimination: float
    expert_difficulty: int
    is_quarantined: bool


class StateOut(BaseModel):
    ku_id: int
    theta: float
    sigma: float
    stability: float
    difficulty: float
    attempts_count: int


class StateIn(BaseModel):
    ku_id: int
    theta: float = 0.0
    sigma: float = 1.0
    stability: float = 0.0
    difficulty: float = 5.0
    attempts_count: int = 0


class SelectRequest(BaseModel):
    user_id: str
    states: list[StateIn] | None = None
    session_position: int = 0
    session_length: int = 10
    consecutive_errors: int = 0
    session_elapsed_sec: float = 0.0
    review_count: int = 0
    days_to_exam: int = 180
    teacher_assignment_ids: list[int] = Field(default_factory=list)
    recent_task_ids: list[int] = Field(default_factory=list)
    recent_mechanics: list[TaskMechanic] = Field(default_factory=list)
    recent_ku_sets: list[list[int]] = Field(default_factory=list)
    device_width: int = 390
    seed: int | None = None


class ScoreOut(BaseModel):
    task_id: int
    utility: float
    zpd: float
    memory: float
    exam: float
    diversity_penalty: float
    exploration: float
    p_correct: float
    p_star: float
    theta_task: float
    reason: str


class SelectResponse(BaseModel):
    task: TaskOut
    score: ScoreOut
    propensity: float
    candidates: int
    ranked: list[ScoreOut]


class AttemptIn(BaseModel):
    user_id: str
    task_id: int
    session_id: str = "demo"
    timestamp: datetime | None = None
    raw_score: float | None = None
    answer: str | None = None
    duration_ms: int
    self_grade: Rating
    hints_used: list[int] = Field(default_factory=list)
    hint_weight_sum: float = 0.0
    pos_in_session: int = 0
    session_elapsed_sec: float = 0.0
    consecutive_errors: int = 0
    propensity_score: float = 1.0
    repeat_same_task: bool = False
    teacher_assigned: bool = False


class GradeRequest(BaseModel):
    task_id: int
    answer: str


class GradeResponse(BaseModel):
    task_id: int
    raw_score: float
    correct: bool
    normalized_user: str
    normalized_correct: str
    method: str
    needs_manual_review: bool


class AttemptOut(BaseModel):
    y_eff: float
    g_sync: int
    p_correct_before: float
    p_star_before: float
    mastery_updates: list[dict]
    memory_updates: list[dict]
    diagnostics: dict


class BlueprintItemIn(BaseModel):
    position: int
    max_points: float
    ku_ids: list[int]
    b_exam: float = 0.0
    a_exam: float = 1.0
    c_exam: float = 0.0


class ReadinessRequest(BaseModel):
    user_id: str
    blueprint: list[BlueprintItemIn]
    exam_date: datetime
    states: list[StateIn] | None = None


class ReadinessOut(BaseModel):
    expected_raw: float
    raw_low: float
    raw_high: float
    expected_scaled: float
    scaled_low: float
    scaled_high: float
    confidence: float
    variance: float


class InMemoryStore:
    def __init__(self, engine: IntervallumEngine) -> None:
        self.engine = engine
        self.states: dict[str, dict[int, UserKUState]] = {}

    def get(self, user_id: str) -> dict[int, UserKUState]:
        if user_id not in self.states:
            self.states[user_id] = self.engine.new_user_state()
        return self.states[user_id]

    def apply(self, user_id: str, rows: list[StateIn]) -> None:
        states = self.get(user_id)
        for row in rows:
            st = states.setdefault(row.ku_id, UserKUState(ku_id=row.ku_id))
            st.theta = row.theta
            st.sigma = row.sigma
            st.attempts_count = row.attempts_count
            if st.memory is None and row.stability > 0:
                from .types import MemoryState

                now = datetime.now(timezone.utc)
                st.memory = MemoryState(
                    stability=row.stability,
                    difficulty=row.difficulty,
                    state=2,
                    last_review_at=now,
                    due_at=now,
                )


def _task_out(task: Task) -> TaskOut:
    return TaskOut(
        id=task.id,
        mechanic=task.mechanic,
        ku_ids=list(task.ku_ids),
        c_guess=task.c_guess,
        t_expected_sec=task.t_expected_sec,
        b_difficulty=task.b_difficulty,
        a_discrimination=task.a_discrimination,
        expert_difficulty=task.expert_difficulty,
        is_quarantined=task.is_quarantined,
    )


def _score_out(score) -> ScoreOut:
    return ScoreOut(
        task_id=score.task_id,
        utility=score.utility,
        zpd=score.zpd,
        memory=score.memory,
        exam=score.exam,
        diversity_penalty=score.diversity_penalty,
        exploration=score.exploration,
        p_correct=score.p_correct,
        p_star=score.p_star,
        theta_task=score.theta_task,
        reason=score.reason.value if hasattr(score.reason, "value") else str(score.reason),
    )


def create_app(engine: IntervallumEngine | None = None) -> FastAPI:
    if engine is None:
        candidates = [
            Path(__file__).resolve().parents[2] / "content" / "fipi_physics_2026.json",
            Path.cwd() / "content" / "fipi_physics_2026.json",
            Path("/app/content/fipi_physics_2026.json"),
        ]
        bank_path = next((p for p in candidates if p.exists()), None)
        if bank_path is not None:
            engine, _bank = load_engine(bank_path, EngineConfig())
        else:
            world = build_world(n_users=2, n_kus=12, n_tasks=60)
            engine = IntervallumEngine(world.knowledge_units, world.tasks, EngineConfig())
    store = InMemoryStore(engine)
    app = FastAPI(title="Intervallum ML API", version="0.1.0")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "knowledge_units": len(engine.ku), "tasks": len(engine.tasks)}

    @app.get("/v1/tasks", response_model=list[TaskOut])
    def tasks() -> list[TaskOut]:
        return [_task_out(t) for t in engine.tasks.values()]

    @app.post("/v1/select", response_model=SelectResponse)
    def select(req: SelectRequest) -> SelectResponse:
        states = store.get(req.user_id)
        if req.states is not None:
            store.apply(req.user_id, req.states)
            states = store.get(req.user_id)
        ctx = SelectionContext(
            session_position=req.session_position,
            session_length=req.session_length,
            consecutive_errors=req.consecutive_errors,
            session_elapsed_sec=req.session_elapsed_sec,
            review_count=req.review_count,
            days_to_exam=req.days_to_exam,
            teacher_assignment_ids=frozenset(req.teacher_assignment_ids),
            recent_task_ids=tuple(req.recent_task_ids),
            recent_mechanics=tuple(req.recent_mechanics),
            recent_ku_sets=tuple(frozenset(x) for x in req.recent_ku_sets),
            device_width=req.device_width,
            seed=req.seed,
        )
        try:
            result = engine.select(states, ctx)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return SelectResponse(
            task=_task_out(result.task),
            score=_score_out(result.score),
            propensity=result.propensity,
            candidates=result.candidates,
            ranked=[_score_out(s) for s in result.ranked],
        )

    @app.post("/v1/grade", response_model=GradeResponse)
    def grade(req: GradeRequest) -> GradeResponse:
        task = engine.tasks.get(req.task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="Task not found")
        result = grade_answer(task, req.answer)
        return GradeResponse(
            task_id=result.task_id,
            raw_score=result.raw_score,
            correct=result.correct,
            normalized_user=result.normalized_user,
            normalized_correct=result.normalized_correct,
            method=result.method,
            needs_manual_review=result.needs_manual_review,
        )

    @app.post("/v1/attempt", response_model=AttemptOut)
    def attempt(req: AttemptIn) -> AttemptOut:
        states = store.get(req.user_id)
        task = engine.tasks.get(req.task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="Task not found")
        if req.raw_score is None:
            if req.answer is None:
                raise HTTPException(
                    status_code=422,
                    detail="Provide either raw_score or answer for grading",
                )
            req.raw_score = grade_answer(task, req.answer).raw_score
        now = req.timestamp or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        snapshot = engine.snapshot(states, now)
        a = Attempt(
            user_id=req.user_id,
            task_id=req.task_id,
            session_id=req.session_id,
            timestamp=now,
            raw_score=req.raw_score,
            duration_ms=req.duration_ms,
            self_grade=req.self_grade,
            hints_used=tuple(req.hints_used),
            hint_weight_sum=req.hint_weight_sum,
            pos_in_session=req.pos_in_session,
            session_elapsed_sec=req.session_elapsed_sec,
            consecutive_errors=req.consecutive_errors,
            state_before=snapshot,
            propensity_score=req.propensity_score,
            repeat_same_task=req.repeat_same_task,
            teacher_assigned=req.teacher_assigned,
        )
        result = engine.review_attempt(states, task, a)
        return AttemptOut(
            y_eff=result.telemetry.y_eff,
            g_sync=int(result.telemetry.g_sync),
            p_correct_before=result.p_correct_before,
            p_star_before=result.p_star_before,
            mastery_updates=[
                {
                    "ku_id": u.ku_id,
                    "before_theta": u.before_theta,
                    "after_theta": u.after_theta,
                    "before_sigma": u.before_sigma,
                    "after_sigma": u.after_sigma,
                    "responsibility": u.responsibility,
                }
                for u in result.mastery_updates
            ],
            memory_updates=[
                {
                    "ku_id": u.ku_id,
                    "rating": int(u.rating),
                    "stability_before": u.before.stability,
                    "stability_after": u.after.stability,
                    "due_at": u.after.due_at.isoformat() if u.after.due_at else None,
                }
                for u in result.memory_updates
            ],
            diagnostics=dict(result.diagnostics),
        )

    @app.post("/v1/readiness", response_model=ReadinessOut)
    def readiness(req: ReadinessRequest) -> ReadinessOut:
        states = store.get(req.user_id)
        if req.states is not None:
            store.apply(req.user_id, req.states)
            states = store.get(req.user_id)
        blueprint = [
            ExamBlueprintItem(
                position=x.position,
                max_points=x.max_points,
                ku_ids=tuple(x.ku_ids),
                b_exam=x.b_exam,
                a_exam=x.a_exam,
                c_exam=x.c_exam,
            )
            for x in req.blueprint
        ]
        exam_date = req.exam_date
        if exam_date.tzinfo is None:
            exam_date = exam_date.replace(tzinfo=timezone.utc)
        result = engine.readiness_projection(states, blueprint, exam_date)
        return ReadinessOut(
            expected_raw=result.expected_raw,
            raw_low=result.raw_low,
            raw_high=result.raw_high,
            expected_scaled=result.expected_scaled,
            scaled_low=result.scaled_low,
            scaled_high=result.scaled_high,
            confidence=result.confidence,
            variance=result.variance,
        )

    return app


app = create_app()
