"""Offline calibration and content-health diagnostics.

The calibration fits item difficulty/discrimination and the two global
parameters (``tau`` for conjunctive strictness and ``gamma`` for memory decay)
against logged attempts.  It is a compact, explainable IRT model rather than a
deep knowledge-tracing network, which is the right choice for the first
50k--300k attempts of a new product.

The optimization uses L-BFGS-B with analytic gradients.  No PyTorch dependency
is required, so the pipeline runs on a modest CI runner and on a school server.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from .coldstart import MetaDifficultyModel
from .config import CalibrationConfig
from .mathutils import (
    brier_score,
    clamp,
    expected_calibration_error,
    point_biserial,
    sigmoid,
    softmin,
)
from .types import (
    Attempt,
    CalibrationReport,
    ExamBlueprintItem,
    ItemHealth,
    Task,
)


@dataclass
class _AttemptData:
    item_index: int
    theta: np.ndarray
    log_r: np.ndarray
    weights: np.ndarray
    c_guess: float
    y: float


def _softmin_derivative(
    x: np.ndarray, tau: float, weights: np.ndarray
) -> tuple[float, np.ndarray, float]:
    """Return theta_task, d theta / d x, d theta / d tau."""

    w = np.asarray(weights, dtype=float)
    total = float(np.sum(w))
    if total <= 0:
        w = np.full(x.shape, 1.0 / x.size, dtype=float)
    else:
        w = w / total
    tau = max(float(tau), 1e-6)
    m = float(np.max(-tau * x))
    e = np.exp(-tau * x - m)
    s = float(np.sum(w * e))
    if s <= 0:
        s = 1e-300
    L = m + math.log(s)
    theta = -L / tau
    q = w * e / s
    e_x = float(np.sum(q * x))
    d_tau = L / (tau ** 2) + e_x / tau
    return theta, q, d_tau


class ItemCalibrator:
    def __init__(
        self,
        cfg: CalibrationConfig | None = None,
        meta: MetaDifficultyModel | None = None,
    ) -> None:
        self.cfg = cfg or CalibrationConfig()
        self.meta = meta

    def _prepare(
        self,
        tasks: Mapping[int, Task],
        attempts: Sequence[Attempt],
    ) -> tuple[list[int], dict[int, int], list[_AttemptData], np.ndarray, np.ndarray]:
        item_ids = sorted({a.task_id for a in attempts if a.task_id in tasks})
        index = {task_id: i for i, task_id in enumerate(item_ids)}
        data: list[_AttemptData] = []
        b_meta = np.asarray(
            [
                self.meta.predict(tasks[tid]) if self.meta is not None else tasks[tid].b_difficulty
                for tid in item_ids
            ],
            dtype=float,
        )
        a_init = np.asarray([tasks[tid].a_discrimination for tid in item_ids], dtype=float)
        for attempt in attempts:
            task = tasks.get(attempt.task_id)
            if task is None or not task.ku_ids:
                continue
            thetas: list[float] = []
            log_rs: list[float] = []
            for ku_id in task.ku_ids:
                snap = attempt.state_before.get(ku_id)
                if snap is None:
                    thetas.append(0.0)
                    log_rs.append(0.0)
                    continue
                thetas.append(snap.theta)
                # If memory has not started, there is no decay term.
                log_rs.append(math.log(snap.retrievability) if snap.retrievability > 0 else 0.0)
            weights = np.asarray(task.weights() or [1.0] * len(thetas), dtype=float)
            data.append(
                _AttemptData(
                    item_index=index[attempt.task_id],
                    theta=np.asarray(thetas, dtype=float),
                    log_r=np.asarray(log_rs, dtype=float),
                    weights=weights,
                    c_guess=clamp(task.c_guess, 0.0, 0.99),
                    y=clamp(attempt.raw_score, 0.0, 1.0),
                )
            )
        return item_ids, index, data, b_meta, a_init

    def _unpack(
        self, params: np.ndarray, n_items: int
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        b = params[:n_items]
        a = params[n_items : 2 * n_items]
        tau = float(params[-2])
        gamma = float(params[-1])
        return b, a, tau, gamma

    def fit(
        self,
        tasks: Mapping[int, Task],
        attempts: Sequence[Attempt],
        *,
        tau0: float = 2.0,
        gamma0: float = 0.8,
    ) -> CalibrationReport:
        item_ids, index, data, b_meta, a_init = self._prepare(tasks, attempts)
        if not item_ids or not data:
            return CalibrationReport(
                b={},
                a={},
                tau=tau0,
                gamma=gamma0,
                loss=0.0,
                brier=0.0,
                ece=0.0,
                n_attempts=0,
                n_items=0,
                converged=True,
            )
        n_items = len(item_ids)
        b_init = np.asarray([tasks[tid].b_difficulty for tid in item_ids], dtype=float)
        # Blend expert and meta prior for a stable start.
        b_init = 0.5 * b_init + 0.5 * b_meta
        b_init = np.clip(b_init, self.cfg.b_min, self.cfg.b_max)
        a_init = np.clip(a_init, self.cfg.a_min, self.cfg.a_max)
        tau0 = float(np.clip(tau0, self.cfg.tau_min, self.cfg.tau_max))
        gamma0 = float(np.clip(gamma0, self.cfg.gamma_min, self.cfg.gamma_max))
        x0 = np.concatenate([b_init, a_init, np.asarray([tau0, gamma0])])

        bounds = (
            [(self.cfg.b_min, self.cfg.b_max)] * n_items
            + [(self.cfg.a_min, self.cfg.a_max)] * n_items
            + [
                (
                    (tau0, tau0)
                    if len(data) < self.cfg.min_attempts_for_tau
                    else (self.cfg.tau_min, self.cfg.tau_max)
                ),
                (
                    (gamma0, gamma0)
                    if len(data) < self.cfg.min_attempts_for_gamma
                    else (self.cfg.gamma_min, self.cfg.gamma_max)
                ),
            ]
        )

        def objective(params: np.ndarray) -> tuple[float, np.ndarray]:
            b, a, tau, gamma = self._unpack(params, n_items)
            grad = np.zeros_like(params)
            loss = 0.0
            for d in data:
                x = d.theta + gamma * d.log_r
                theta_task, q, d_tau = _softmin_derivative(x, tau, d.weights)
                z = a[d.item_index] * (theta_task - b[d.item_index])
                p_star = sigmoid(z)
                p = clamp(d.c_guess + (1.0 - d.c_guess) * p_star, 1e-9, 1.0 - 1e-9)
                y = d.y
                loss += -(y * math.log(p) + (1.0 - y) * math.log(1.0 - p))
                dL_dp = (p - y) / (p * (1.0 - p))
                dL_dz = dL_dp * (1.0 - d.c_guess) * p_star * (1.0 - p_star)
                grad_b = dL_dz * (-a[d.item_index])
                grad_a = dL_dz * (theta_task - b[d.item_index])
                grad_theta = dL_dz * a[d.item_index]
                grad[d.item_index] += grad_b
                grad[n_items + d.item_index] += grad_a
                grad[-1] += grad_theta * float(np.sum(q * d.log_r))
                grad[-2] += grad_theta * d_tau

            # Priors: items with little data stay near the meta prediction.
            loss += float(np.sum(self.cfg.lambda_b * (b - b_meta) ** 2))
            grad[:n_items] += 2.0 * self.cfg.lambda_b * (b - b_meta)
            loss += float(np.sum(self.cfg.lambda_a * (a - 1.0) ** 2))
            grad[n_items : 2 * n_items] += 2.0 * self.cfg.lambda_a * (a - 1.0)
            loss += self.cfg.lambda_tau * (tau - tau0) ** 2
            grad[-2] += 2.0 * self.cfg.lambda_tau * (tau - tau0)
            loss += self.cfg.lambda_gamma * (gamma - gamma0) ** 2
            grad[-1] += 2.0 * self.cfg.lambda_gamma * (gamma - gamma0)
            return float(loss), grad

        result = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": self.cfg.max_iter},
        )
        b, a, tau, gamma = self._unpack(result.x, n_items)

        # Diagnostic metrics on the calibrated model.
        y_true: list[float] = []
        p_pred: list[float] = []
        for d in data:
            x = d.theta + gamma * d.log_r
            theta_task = softmin(x.tolist(), tau, d.weights.tolist())
            p_star = sigmoid(a[d.item_index] * (theta_task - b[d.item_index]))
            p = clamp(d.c_guess + (1.0 - d.c_guess) * p_star, 1e-9, 1.0 - 1e-9)
            y_true.append(d.y)
            p_pred.append(p)

        return CalibrationReport(
            b={tid: float(b[i]) for tid, i in index.items()},
            a={tid: float(a[i]) for tid, i in index.items()},
            tau=float(tau),
            gamma=float(gamma),
            loss=float(result.fun),
            brier=brier_score(y_true, p_pred),
            ece=expected_calibration_error(y_true, p_pred),
            n_attempts=len(data),
            n_items=n_items,
            converged=bool(result.success),
        )


def detect_bad_items(
    tasks: Mapping[int, Task],
    attempts: Sequence[Attempt],
    *,
    calibrated_a: Mapping[int, float] | None = None,
    min_attempts: int = 15,
    pbis_threshold: float = 0.0,
    discrimination_threshold: float = 0.3,
    time_ratio_threshold: float = 2.5,
) -> list[ItemHealth]:
    """Flag tasks whose empirical behaviour contradicts the intended skill.

    A negative point-biserial correlation is the strongest signal: stronger
    students fail the item more often than weaker students, which almost always
    means a wrong key, an ambiguous wording, or a broken interactive.
    """

    by_task: dict[int, list[Attempt]] = {}
    for a in attempts:
        by_task.setdefault(a.task_id, []).append(a)

    out: list[ItemHealth] = []
    for task_id, rows in by_task.items():
        task = tasks.get(task_id)
        if task is None:
            continue
        n = len(rows)
        if n < min_attempts:
            continue
        item_scores = [a.raw_score for a in rows]
        abilities = []
        for a in rows:
            vals = [s.theta for s in a.state_before.values()]
            abilities.append(sum(vals) / len(vals) if vals else 0.0)
        pbis = point_biserial(item_scores, abilities)
        ratios = [
            (a.duration_ms / 1000.0) / max(task.t_expected_sec, 1e-9)
            for a in rows
        ]
        median_ratio = float(np.median(ratios)) if ratios else 0.0
        discrimination = (
            calibrated_a.get(task_id, task.a_discrimination)
            if calibrated_a is not None
            else task.a_discrimination
        )
        reasons: list[str] = []
        if pbis < pbis_threshold:
            reasons.append("negative_point_biserial")
        if discrimination < discrimination_threshold and n >= 50:
            reasons.append("low_discrimination")
        if median_ratio > time_ratio_threshold:
            reasons.append("slow_or_broken_interaction")
        out.append(
            ItemHealth(
                task_id=task_id,
                attempts=n,
                point_biserial=float(pbis),
                discrimination=float(discrimination),
                median_time_ratio=float(median_ratio),
                quarantine=bool(reasons),
                reasons=tuple(reasons),
            )
        )
    return sorted(out, key=lambda x: (not x.quarantine, x.point_biserial))


def calibrate_exam_blueprint(
    blueprint: Sequence[ExamBlueprintItem],
    predicted_raw: Sequence[float],
    actual_raw: Sequence[float],
) -> tuple[list[ExamBlueprintItem], float]:
    """Fit a single difficulty shift for the exam blueprint.

    This is the anchor calibration described in the architecture: the app's
    tasks are usually easier than a real exam, so the blueprint difficulty is
    shifted until mean predicted raw score matches observed mock-exam scores.
    """

    from dataclasses import replace

    pred = np.asarray(predicted_raw, dtype=float)
    actual = np.asarray(actual_raw, dtype=float)
    if pred.size == 0 or actual.size == 0:
        return list(blueprint), 0.0

    def loss(delta: float) -> float:
        shifted = pred + delta
        return float(np.mean((shifted - actual) ** 2))

    result = minimize_scalar(loss, bounds=(-2.0, 2.0), method="bounded")
    delta = float(result.x)
    # Shift each blueprint item's difficulty in the opposite direction: a
    # positive delta means predictions were too low, so the exam is easier
    # than assumed and b should decrease.
    shifted = [replace(item, b_exam=item.b_exam - delta) for item in blueprint]
    return shifted, delta
