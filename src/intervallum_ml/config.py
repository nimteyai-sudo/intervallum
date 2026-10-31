"""Tunable configuration for the Intervallum ML engine.

Every hyper-parameter that a product or research decision might change lives
here.  The defaults are intentionally conservative for a cold-start product:
explainable models, mild exploration, and a hard cap on review workload.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class TelemetryConfig:
    hint_multiplier_floor: float = 0.0
    fast_answer_ratio: float = 0.35
    slow_answer_ratio: float = 5.0
    time_penalty_floor: float = 0.7
    time_penalty_start: float = 2.0
    hard_hint_sum: float = 0.5
    confident_fast_ratio: float = 0.7
    slow_but_correct_ratio: float = 1.8
    self_weight: float = 0.30
    objective_weight: float = 0.70


@dataclass(frozen=True, slots=True)
class MasteryConfig:
    tau: float = 2.0
    learning_rate: float = 0.35
    sigma_min: float = 0.25
    sigma_max: float = 1.5
    theta_min: float = -4.0
    theta_max: float = 4.0
    min_retrievability: float = 0.01
    info_gain_scale: float = 1.0
    memory_update_threshold: float = 0.15
    repeat_task_theta_discount: float = 0.25


@dataclass(frozen=True, slots=True)
class MemoryConfig:
    desired_retention: float = 0.9
    maximum_interval: int = 36500
    enable_fuzzing: bool = False
    learning_steps_minutes: tuple[int, ...] = (1, 10)
    relearning_steps_minutes: tuple[int, ...] = (10,)
    # Optional FSRS-6 weights produced by the offline optimizer.  None means
    # "use the package defaults".
    fsrs_parameters: tuple[float, ...] | None = None
    # If FSRS is unavailable, fixed intervals are used as a safe fallback.
    fallback_intervals_days: tuple[float, ...] = (0.007, 1.0, 3.0, 7.0)


@dataclass(frozen=True, slots=True)
class SelectorConfig:
    w_zpd: float = 0.35
    w_memory: float = 0.30
    w_exam: float = 0.25
    w_diversity: float = 0.10
    zpd_sigma: float = 0.15
    p_target_base: float = 0.72
    p_target_warmup: float = 0.85
    p_target_fatigue: float = 0.88
    fatigue_after_errors: int = 2
    fatigue_after_seconds: float = 720.0
    memory_threshold: float = 0.90
    memory_scale: float = 0.25
    memory_cap: float = 1.5
    max_review_share: float = 0.40
    prerequisite_success: float = 0.65
    cooldown_days: int = 7
    teacher_bonus: float = 10.0
    diversity_same_mechanic: float = 0.5
    diversity_two_mechanics: float = 2.0
    diversity_same_kus: float = 0.4
    uncertainty_bonus: float = 0.10
    review_backlog_share: float = 0.40


@dataclass(frozen=True, slots=True)
class ReadinessConfig:
    interval_z: float = 1.2815515655446004  # 80% two-sided
    min_confidence: float = 0.05
    max_confidence: float = 0.98
    confidence_attempts_scale: float = 120.0
    confidence_sigma_scale: float = 1.0
    default_scale_points: tuple[tuple[float, float], ...] = (
        (0.0, 0.0),
        (54.0, 100.0),
    )


@dataclass(frozen=True, slots=True)
class CalibrationConfig:
    lambda_b: float = 1.0
    lambda_a: float = 5.0
    lambda_tau: float = 0.5
    lambda_gamma: float = 2.0
    a_min: float = 0.3
    a_max: float = 3.0
    b_min: float = -4.0
    b_max: float = 4.0
    tau_min: float = 0.5
    tau_max: float = 5.0
    gamma_min: float = 0.0
    gamma_max: float = 2.0
    min_attempts_for_tau: int = 1000
    min_attempts_for_gamma: int = 3000
    max_iter: int = 300
    verbose: bool = False


@dataclass(frozen=True, slots=True)
class EngineConfig:
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    mastery: MasteryConfig = field(default_factory=MasteryConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    selector: SelectorConfig = field(default_factory=SelectorConfig)
    readiness: ReadinessConfig = field(default_factory=ReadinessConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
