"""FSRS memory layer.

We wrap the MIT-licensed ``fsrs`` Python package.  The public ``MemoryState``
is deliberately small so that the same state can be stored in PostgreSQL and
consumed by a TypeScript fast path later.

FSRS-6 is used by default because the installed package is mature and stable.
The architecture's FSRS-5 recommendation is a conservative fallback; the
scheduler is isolated behind this interface so the model version can be
changed without touching the rest of the engine.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fsrs import Card, Scheduler
from fsrs import Rating as FSRSRating
from fsrs import State as FSRSState

from .config import MemoryConfig
from .mathutils import clamp
from .types import MemoryState, Rating

_FSRS_TO_RATING = {
    FSRSRating.Again: Rating.AGAIN,
    FSRSRating.Hard: Rating.HARD,
    FSRSRating.Good: Rating.GOOD,
    FSRSRating.Easy: Rating.EASY,
}
_RATING_TO_FSRS = {
    Rating.AGAIN: FSRSRating.Again,
    Rating.HARD: FSRSRating.Hard,
    Rating.GOOD: FSRSRating.Good,
    Rating.EASY: FSRSRating.Easy,
}


class MemoryScheduler:
    """Thin, deterministic wrapper around FSRS."""

    def __init__(self, cfg: MemoryConfig | None = None) -> None:
        self.cfg = cfg or MemoryConfig()
        kwargs = {}
        if self.cfg.fsrs_parameters is not None:
            kwargs["parameters"] = self.cfg.fsrs_parameters
        self._scheduler = Scheduler(
            desired_retention=self.cfg.desired_retention,
            learning_steps=tuple(timedelta(minutes=m) for m in self.cfg.learning_steps_minutes),
            relearning_steps=tuple(timedelta(minutes=m) for m in self.cfg.relearning_steps_minutes),
            maximum_interval=self.cfg.maximum_interval,
            enable_fuzzing=self.cfg.enable_fuzzing,
            **kwargs,
        )

    @staticmethod
    def _utc(now: datetime) -> datetime:
        if now.tzinfo is None:
            return now.replace(tzinfo=timezone.utc)
        return now.astimezone(timezone.utc)

    def _to_card(self, state: MemoryState, ku_id: int, now: datetime) -> Card:
        if not state.is_started():
            return Card(card_id=ku_id, due=self._utc(now))
        return Card(
            card_id=ku_id,
            state=FSRSState(state.state),
            step=state.step,
            stability=state.stability,
            difficulty=state.difficulty,
            due=self._utc(state.due_at or now),
            last_review=self._utc(state.last_review_at or now),
        )

    @staticmethod
    def _from_card(card: Card) -> MemoryState:
        return MemoryState(
            stability=float(card.stability or 0.0),
            difficulty=float(card.difficulty or 5.0),
            state=int(card.state.value if isinstance(card.state, FSRSState) else card.state),
            step=card.step,
            last_review_at=card.last_review,
            due_at=card.due,
        )

    def retrievability(self, state: MemoryState, now: datetime) -> float:
        if not state.is_started():
            return 0.0
        card = self._to_card(state, 0, now)
        return float(self._scheduler.get_card_retrievability(card, self._utc(now)))

    def review(self, state: MemoryState, rating: Rating, now: datetime) -> MemoryState:
        now = self._utc(now)
        card = self._to_card(state, 0, now)
        reviewed, _log = self._scheduler.review_card(
            card, _RATING_TO_FSRS[rating], review_datetime=now
        )
        return self._from_card(reviewed)

    def review_with_interval(
        self, state: MemoryState, rating: Rating, now: datetime
    ) -> tuple[MemoryState, float]:
        before = self._utc(now)
        after = self.review(state, rating, now)
        interval = 0.0
        if after.due_at is not None:
            interval = max(0.0, (after.due_at - before).total_seconds() / 86400.0)
        return after, interval

    def fallback_review(self, state: MemoryState, rating: Rating, now: datetime) -> MemoryState:
        """Deterministic fallback when FSRS is unavailable.

        This is intentionally simple and is not used in normal operation.  It
        preserves the state machine and avoids losing a review if the optional
        dependency is missing.
        """

        now = self._utc(now)
        intervals = self.cfg.fallback_intervals_days
        if rating == Rating.AGAIN:
            stability = max(0.007, state.stability * 0.25 if state.stability else 0.007)
            due = now + timedelta(days=intervals[0])
            return MemoryState(
                stability=stability,
                difficulty=clamp(state.difficulty + 0.8, 1.0, 10.0),
                state=3,
                step=0,
                last_review_at=now,
                due_at=due,
            )
        factor = {Rating.HARD: 1.15, Rating.GOOD: 1.8, Rating.EASY: 2.5}[rating]
        stability = max(0.1, (state.stability or 0.1) * factor)
        due = now + timedelta(days=min(stability, float(self.cfg.maximum_interval)))
        return MemoryState(
            stability=stability,
            difficulty=clamp(state.difficulty - {Rating.HARD: 0.1, Rating.GOOD: 0.0, Rating.EASY: -0.2}[rating], 1.0, 10.0),
            state=2,
            step=None,
            last_review_at=now,
            due_at=due,
        )
