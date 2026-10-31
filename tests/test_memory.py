from __future__ import annotations

from datetime import datetime, timezone

from intervallum_ml.memory import MemoryScheduler
from intervallum_ml.types import MemoryState, Rating


def test_first_review_creates_stability_and_due_date():
    s = MemoryScheduler()
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    after = s.review(MemoryState(), Rating.GOOD, now)
    assert after.stability > 0
    assert after.last_review_at == now
    assert after.due_at is not None and after.due_at > now
    assert after.is_started()


def test_retrievability_decreases_over_time():
    s = MemoryScheduler()
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    state = s.review(MemoryState(), Rating.GOOD, now)
    r0 = s.retrievability(state, now)
    r1 = s.retrievability(state, datetime(2026, 2, 1, tzinfo=timezone.utc))
    assert 0 < r1 < r0 <= 1.0


def test_again_reduces_stability():
    s = MemoryScheduler()
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    good = s.review(MemoryState(), Rating.GOOD, now)
    again = s.review(good, Rating.AGAIN, datetime(2026, 1, 10, tzinfo=timezone.utc))
    assert again.stability < good.stability
    assert again.difficulty >= good.difficulty


def test_deterministic_when_fuzzing_disabled():
    s = MemoryScheduler()
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    a = s.review(MemoryState(), Rating.GOOD, now)
    b = s.review(MemoryState(), Rating.GOOD, now)
    assert a.stability == b.stability
    assert a.difficulty == b.difficulty
    assert a.due_at == b.due_at


def test_unstarted_retrievability_is_zero():
    s = MemoryScheduler()
    assert s.retrievability(MemoryState(), datetime.now(timezone.utc)) == 0.0
