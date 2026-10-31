"""Core domain types for the Intervallum adaptive learning engine.

The module deliberately keeps the ML core independent from HTTP, database and
frontend concerns.  All identifiers are integers or opaque strings; no personal
data belongs here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, IntEnum
from typing import Mapping


class Rating(IntEnum):
    """FSRS rating.  The numeric values match the FSRS specification."""

    AGAIN = 1
    HARD = 2
    GOOD = 3
    EASY = 4


class KUType(str, Enum):
    """Two aspects of a knowledge unit.

    Declarative units are recalled facts/formulae and are scheduled by FSRS.
    Procedural units are skills and are tracked by the IRT mastery model.
    A unit may be hybrid when a formula is both remembered and applied.
    """

    DECLARATIVE = "declarative"
    PROCEDURAL = "procedural"
    HYBRID = "hybrid"


class TaskMechanic(str, Enum):
    MCQ_SINGLE = "mcq_single"
    MCQ_MULTI = "mcq_multi"
    NUMERIC = "numeric"
    FORMULA_BUILD = "formula_build"
    FORMULA_DERIVE = "formula_derive"
    FIND_ERROR = "find_error"
    DRAW_FORCES = "draw_forces"
    DIM_CHECK = "dim_check"
    PREDICT_CHANGE = "predict_change"
    MATCH_ORDER = "match_order"
    GRAPH_READ = "graph_read"
    FREE_RESPONSE = "free_response"


class PolicyReason(str, Enum):
    REVIEW_DUE = "review_due"
    WEAK_SKILL = "weak_skill"
    EXAM_VALUE = "exam_value"
    TEACHER_ASSIGNMENT = "teacher_assignment"
    EXPLORATION = "exploration"
    PREREQUISITE = "prerequisite"
    BALANCED = "balanced"


@dataclass(frozen=True, slots=True)
class KnowledgeUnit:
    id: int
    code: str
    name: str
    ku_type: KUType
    topic_id: int
    section_id: int
    ege_weight: float = 1.0
    parent_ids: tuple[int, ...] = ()
    neighbor_ids: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class Task:
    id: int
    mechanic: TaskMechanic
    ku_ids: tuple[int, ...]
    ku_weights: tuple[float, ...] = ()
    c_guess: float = 0.0
    t_expected_sec: float = 60.0
    b_difficulty: float = 0.0
    a_discrimination: float = 1.0
    expert_difficulty: int = 3
    is_quarantined: bool = False
    topic_id: int | None = None
    section_id: int | None = None
    text: str = ""
    source: str = ""
    version: int = 1
    media: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    answer_raw: str = ""
    answer_type: str = ""
    answer_value: float | None = None
    answer_digits: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.c_guess <= 1.0:
            raise ValueError("c_guess must be in [0, 1]")
        if self.t_expected_sec <= 0:
            raise ValueError("t_expected_sec must be positive")
        if self.ku_weights and len(self.ku_weights) != len(self.ku_ids):
            raise ValueError("ku_weights must match ku_ids")
        if not 0.1 <= self.a_discrimination <= 5.0:
            raise ValueError("a_discrimination must be in [0.1, 5.0]")

    def weights(self) -> tuple[float, ...]:
        if self.ku_weights:
            return self.ku_weights
        if not self.ku_ids:
            return ()
        return tuple(1.0 for _ in self.ku_ids)


@dataclass(slots=True)
class MemoryState:
    """FSRS memory state for one user x knowledge-unit pair."""

    stability: float = 0.0
    difficulty: float = 5.0
    state: int = 1  # fsrs.State: 1 Learning, 2 Review, 3 Relearning
    step: int | None = None
    last_review_at: datetime | None = None
    due_at: datetime | None = None

    def is_started(self) -> bool:
        return self.last_review_at is not None and self.stability > 0


@dataclass(slots=True)
class UserKUState:
    """Mutable per-user, per-knowledge-unit state.

    ``theta`` and ``sigma`` describe procedural mastery on a logit scale.
    ``memory`` describes declarative recall and is only present for
    declarative/hybrid units.
    """

    ku_id: int
    theta: float = 0.0
    sigma: float = 1.0
    memory: MemoryState | None = None
    attempts_count: int = 0
    last_seen_at: datetime | None = None

    def copy(self) -> "UserKUState":
        from copy import deepcopy

        return deepcopy(self)


@dataclass(frozen=True, slots=True)
class StateSnapshot:
    theta: float
    sigma: float
    stability: float
    difficulty: float
    retrievability: float


@dataclass(frozen=True, slots=True)
class Attempt:
    """One task attempt.  ``state_before`` must be captured before updates."""

    user_id: str
    task_id: int
    session_id: str
    timestamp: datetime
    raw_score: float
    duration_ms: int
    self_grade: Rating
    hints_used: tuple[int, ...] = ()
    hint_weight_sum: float = 0.0
    pos_in_session: int = 0
    session_elapsed_sec: float = 0.0
    consecutive_errors: int = 0
    state_before: Mapping[int, StateSnapshot] = field(default_factory=dict)
    propensity_score: float = 1.0
    policy_version: str = "dev"
    device_width: int = 390
    repeat_same_task: bool = False
    teacher_assigned: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.raw_score <= 1.0:
            raise ValueError("raw_score must be in [0, 1]")
        if self.duration_ms < 0:
            raise ValueError("duration_ms must be non-negative")


@dataclass(frozen=True, slots=True)
class TelemetryResult:
    y_eff: float
    g_sync: Rating
    hint_multiplier: float
    time_penalty: float
    objective_grade: Rating
    self_grade: Rating


@dataclass(frozen=True, slots=True)
class TaskPrediction:
    task_id: int
    theta_task: float
    p_star: float
    p_correct: float
    sigma_task: float
    effective_thetas: Mapping[int, float]
    responsibilities: Mapping[int, float]


@dataclass(frozen=True, slots=True)
class MemoryUpdate:
    ku_id: int
    rating: Rating
    before: MemoryState
    after: MemoryState
    retrievability_before: float
    retrievability_after: float
    reviewed_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class ReviewRecord:
    """One FSRS review event, used by the optional FSRS weight optimizer."""

    ku_id: int
    rating: Rating
    reviewed_at: datetime
    duration_ms: int | None = None


@dataclass(frozen=True, slots=True)
class MasteryUpdate:
    ku_id: int
    before_theta: float
    after_theta: float
    before_sigma: float
    after_sigma: float
    responsibility: float


@dataclass(frozen=True, slots=True)
class AttemptResult:
    telemetry: TelemetryResult
    mastery_updates: tuple[MasteryUpdate, ...]
    memory_updates: tuple[MemoryUpdate, ...]
    p_correct_before: float
    p_star_before: float
    diagnostics: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TaskScore:
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
    reason: PolicyReason
    components: Mapping[str, float] = field(default_factory=dict)
    base_utility: float = 0.0


@dataclass(frozen=True, slots=True)
class SelectionResult:
    task: Task
    score: TaskScore
    propensity: float
    ranked: tuple[TaskScore, ...]
    candidates: int


@dataclass(frozen=True, slots=True)
class SelectionContext:
    now: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    session_position: int = 0
    session_length: int = 10
    consecutive_errors: int = 0
    session_elapsed_sec: float = 0.0
    review_count: int = 0
    days_to_exam: int = 180
    teacher_assignment_ids: frozenset[int] = frozenset()
    recent_task_ids: tuple[int, ...] = ()
    recent_mechanics: tuple[TaskMechanic, ...] = ()
    recent_ku_sets: tuple[frozenset[int], ...] = ()
    device_width: int = 390
    exploration_scale: float = 0.03
    seed: int | None = None


@dataclass(frozen=True, slots=True)
class ExamBlueprintItem:
    position: int
    max_points: float
    ku_ids: tuple[int, ...]
    ku_weights: tuple[float, ...] = ()
    b_exam: float = 0.0
    a_exam: float = 1.0
    c_exam: float = 0.0

    def weights(self) -> tuple[float, ...]:
        if self.ku_weights:
            return self.ku_weights
        return tuple(1.0 for _ in self.ku_ids)


@dataclass(frozen=True, slots=True)
class ReadinessProjection:
    expected_raw: float
    raw_low: float
    raw_high: float
    expected_scaled: float
    scaled_low: float
    scaled_high: float
    confidence: float
    by_position: Mapping[int, float]
    variance: float


@dataclass(frozen=True, slots=True)
class ExamScalePoint:
    raw: float
    scaled: float


@dataclass(frozen=True, slots=True)
class PlacementResult:
    selected_task_ids: tuple[int, ...]
    theta_updates: Mapping[int, float]
    estimated_sections: Mapping[int, float]


@dataclass(frozen=True, slots=True)
class ItemHealth:
    task_id: int
    attempts: int
    point_biserial: float
    discrimination: float
    median_time_ratio: float
    quarantine: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    b: Mapping[int, float]
    a: Mapping[int, float]
    tau: float
    gamma: float
    loss: float
    brier: float
    ece: float
    n_attempts: int
    n_items: int
    converged: bool
