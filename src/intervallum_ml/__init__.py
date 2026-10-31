"""Intervallum adaptive learning engine.

Public entry points:
    IntervallumEngine — orchestration
    MemoryScheduler — FSRS memory layer
    MasteryEngine — conjunctive 3PL IRT mastery
    PedagogicalSelector — task selection
    ExamReadinessProjector — EGE readiness forecast
    ItemCalibrator — offline item calibration
"""

from .calibration import ItemCalibrator, calibrate_exam_blueprint, detect_bad_items
from .coldstart import (
    AdaptivePlacement,
    MetaDifficultyModel,
    cold_start_ku,
    cold_start_task,
)
from .config import EngineConfig
from .content import (
    ContentBank,
    ContentValidationReport,
    load_content,
    load_engine,
    validate_content,
)
from .engine import IntervallumEngine
from .evaluation import (
    PredictionMetrics,
    doubly_robust,
    effective_sample_size,
    evaluate_predictions,
    inverse_propensity_score,
    temporal_user_split,
)
from .fsrs_optimizer import FSRSOptimizationResult, optimize_fsrs_parameters
from .grading import GradeResult, grade_answer, normalize_text
from .irt import MasteryEngine
from .memory import MemoryScheduler
from .model_store import ModelArtifact, load_model, save_model
from .readiness import ExamReadinessProjector
from .selector import PedagogicalSelector
from .simulation import build_world, recovery_report, simulate
from .state_store import (
    state_from_dict,
    state_to_dict,
    states_from_dict,
    states_to_dict,
)
from .telemetry import scrub_attempt
from .types import (
    Attempt,
    AttemptResult,
    CalibrationReport,
    ExamBlueprintItem,
    ItemHealth,
    KnowledgeUnit,
    KUType,
    MasteryUpdate,
    MemoryState,
    MemoryUpdate,
    PolicyReason,
    Rating,
    ReadinessProjection,
    ReviewRecord,
    SelectionContext,
    SelectionResult,
    StateSnapshot,
    Task,
    TaskMechanic,
    TaskPrediction,
    TaskScore,
    TelemetryResult,
    UserKUState,
)
from .user_sim import (
    SimulatedResponse,
    SimulatedStudent,
    StudentProfile,
    make_edge_profiles,
    make_profiles,
    run_cohort,
    simulate_student,
    summarise,
    world_from_content,
)

__all__ = [
    "IntervallumEngine",
    "MemoryScheduler",
    "ModelArtifact",
    "save_model",
    "load_model",
    "state_to_dict",
    "state_from_dict",
    "states_to_dict",
    "states_from_dict",
    "MasteryEngine",
    "optimize_fsrs_parameters",
    "FSRSOptimizationResult",
    "PedagogicalSelector",
    "ExamReadinessProjector",
    "ItemCalibrator",
    "AdaptivePlacement",
    "MetaDifficultyModel",
    "EngineConfig",
    "ContentBank",
    "ContentValidationReport",
    "load_content",
    "load_engine",
    "validate_content",
    "Attempt",
    "AttemptResult",
    "CalibrationReport",
    "ExamBlueprintItem",
    "ItemHealth",
    "KnowledgeUnit",
    "KUType",
    "MasteryUpdate",
    "MemoryState",
    "MemoryUpdate",
    "ReviewRecord",
    "StudentProfile",
    "SimulatedStudent",
    "SimulatedResponse",
    "simulate_student",
    "run_cohort",
    "summarise",
    "make_profiles",
    "make_edge_profiles",
    "world_from_content",
    "PolicyReason",
    "Rating",
    "ReadinessProjection",
    "SelectionContext",
    "SelectionResult",
    "StateSnapshot",
    "Task",
    "TaskMechanic",
    "TaskPrediction",
    "TaskScore",
    "TelemetryResult",
    "UserKUState",
    "scrub_attempt",
    "detect_bad_items",
    "calibrate_exam_blueprint",
    "cold_start_task",
    "cold_start_ku",
    "evaluate_predictions",
    "temporal_user_split",
    "inverse_propensity_score",
    "doubly_robust",
    "effective_sample_size",
    "PredictionMetrics",
    "GradeResult",
    "grade_answer",
    "normalize_text",
    "build_world",
    "simulate",
    "recovery_report",
]
