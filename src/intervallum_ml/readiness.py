"""Exam readiness projection.

The projector does not regress "percent correct in app" onto an EGE score.
Instead it simulates the official 26-position blueprint: for every position it
estimates the probability of earning its points from the student's effective
knowledge state at the exam date, then aggregates expectation and variance.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Iterable, Mapping, Sequence

from .config import ReadinessConfig
from .irt import MasteryEngine
from .mathutils import clamp, sigmoid, softmin
from .types import (
    ExamBlueprintItem,
    ReadinessProjection,
    UserKUState,
)


def _piecewise_scale(raw: float, points: Sequence[tuple[float, float]]) -> float:
    pts = sorted((float(x), float(y)) for x, y in points)
    if not pts:
        return raw
    if raw <= pts[0][0]:
        return pts[0][1]
    if raw >= pts[-1][0]:
        return pts[-1][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= raw <= x1:
            if x1 == x0:
                return y1
            t = (raw - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return pts[-1][1]


class ExamReadinessProjector:
    def __init__(self, cfg: ReadinessConfig, mastery: MasteryEngine) -> None:
        self.cfg = cfg
        self.mastery = mastery

    def project(
        self,
        states: Mapping[int, UserKUState],
        blueprint: Iterable[ExamBlueprintItem],
        exam_date: datetime,
        now: datetime | None = None,
        scale_points: Iterable[tuple[float, float]] | None = None,
    ) -> ReadinessProjection:
        now = now or datetime.now(timezone.utc)
        if exam_date.tzinfo is None:
            exam_date = exam_date.replace(tzinfo=timezone.utc)
        points = tuple(scale_points) if scale_points is not None else self.cfg.default_scale_points
        total_max = 0.0
        expected_raw = 0.0
        variance = 0.0
        by_position: dict[int, float] = {}
        sigmas: list[float] = []
        attempts = 0

        for item in blueprint:
            total_max += item.max_points
            if not item.ku_ids:
                c = clamp(item.c_exam, 0.0, 0.99)
                p = clamp(c + (1 - c) * sigmoid(-item.b_exam), 0, 1)
                expected = item.max_points * p
                expected_raw += expected
                variance += (item.max_points ** 2) * p * (1 - p)
                by_position[item.position] = expected
                continue

            thetas: list[float] = []
            item_sigmas: list[float] = []
            for ku_id in item.ku_ids:
                st = states.get(ku_id)
                if st is None:
                    thetas.append(0.0)
                    item_sigmas.append(1.0)
                    continue
                thetas.append(self.mastery.effective_theta(st, exam_date))
                item_sigmas.append(st.sigma)
                attempts += st.attempts_count
            sigma_task = math.sqrt(
                sum(s ** 2 for s in item_sigmas) / max(len(item_sigmas), 1)
            )
            sigmas.append(sigma_task)
            theta_task = softmin(thetas, self.mastery.cfg.tau, item.weights())
            p_star = sigmoid(item.a_exam * (theta_task - item.b_exam))
            c = clamp(item.c_exam, 0.0, 0.99)
            p = clamp(c + (1.0 - c) * p_star, 1e-6, 1.0 - 1e-6)
            expected = item.max_points * p
            expected_raw += expected
            # Binomial noise plus parameter uncertainty.
            variance += (item.max_points ** 2) * (
                p * (1.0 - p)
                + (item.a_exam * p * (1.0 - p) * sigma_task) ** 2
            )
            by_position[item.position] = expected

        sd = math.sqrt(max(variance, 0.0))
        z = self.cfg.interval_z
        raw_low = clamp(expected_raw - z * sd, 0.0, total_max)
        raw_high = clamp(expected_raw + z * sd, 0.0, total_max)
        scaled = _piecewise_scale(expected_raw, points)
        scaled_low = _piecewise_scale(raw_low, points)
        scaled_high = _piecewise_scale(raw_high, points)

        mean_sigma = sum(sigmas) / len(sigmas) if sigmas else 1.0
        sigma_conf = clamp(1.0 - mean_sigma / max(self.cfg.confidence_sigma_scale, 1e-6), 0.0, 1.0)
        attempt_conf = clamp(
            attempts / (attempts + self.cfg.confidence_attempts_scale), 0.0, 1.0
        )
        confidence = clamp(
            0.6 * sigma_conf + 0.4 * attempt_conf,
            self.cfg.min_confidence,
            self.cfg.max_confidence,
        )

        return ReadinessProjection(
            expected_raw=float(expected_raw),
            raw_low=float(raw_low),
            raw_high=float(raw_high),
            expected_scaled=float(scaled),
            scaled_low=float(scaled_low),
            scaled_high=float(scaled_high),
            confidence=float(confidence),
            by_position=by_position,
            variance=float(variance),
        )
