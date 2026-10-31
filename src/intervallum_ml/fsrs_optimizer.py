"""Optional FSRS weight optimization.

FSRS ships with excellent global defaults, so per-user optimization is a
phase-2 feature.  This module wraps the official optimizer when the optional
``torch``/``pandas`` dependencies are installed and degrades gracefully when
they are not.

Install the optional extra with:

    pip install -e ".[fsrs-optimizer]"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .types import Rating, ReviewRecord


@dataclass(frozen=True)
class FSRSOptimizationResult:
    parameters: tuple[float, ...] | None
    n_reviews: int
    available: bool
    message: str


def optimize_fsrs_parameters(
    records: Sequence[ReviewRecord],
    *,
    verbose: bool = False,
) -> FSRSOptimizationResult:
    """Optimize FSRS-6 weights from a review history.

    The official optimizer requires at least 512 review-state reviews; below
    that threshold it returns the global defaults.  We keep that behaviour and
    report it explicitly instead of silently returning a bad fit.
    """

    if not records:
        return FSRSOptimizationResult(None, 0, False, "no review records")
    try:
        from fsrs import Rating as FSRSRating
        from fsrs import ReviewLog
        from fsrs.optimizer import Optimizer
    except Exception as exc:  # pragma: no cover - depends on optional extras
        return FSRSOptimizationResult(
            None, len(records), False, f"optimizer unavailable: {exc}"
        )

    mapping = {
        Rating.AGAIN: FSRSRating.Again,
        Rating.HARD: FSRSRating.Hard,
        Rating.GOOD: FSRSRating.Good,
        Rating.EASY: FSRSRating.Easy,
    }
    logs = [
        ReviewLog(
            card_id=int(r.ku_id),
            rating=mapping[r.rating],
            review_datetime=r.reviewed_at,
            review_duration=r.duration_ms,
        )
        for r in records
    ]
    try:
        optimizer = Optimizer(logs)
        params = optimizer.compute_optimal_parameters(verbose=verbose)
    except Exception as exc:  # pragma: no cover - optional dependency/runtime
        return FSRSOptimizationResult(
            None, len(records), False, f"optimization failed: {exc}"
        )
    return FSRSOptimizationResult(tuple(float(x) for x in params), len(records), True, "ok")
