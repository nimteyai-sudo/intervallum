from __future__ import annotations

from datetime import datetime, timezone

from intervallum_ml.fsrs_optimizer import optimize_fsrs_parameters
from intervallum_ml.types import Rating, ReviewRecord


def test_fsrs_optimizer_handles_empty_records():
    result = optimize_fsrs_parameters([])
    assert not result.available
    assert result.parameters is None


def test_fsrs_optimizer_degrades_without_optional_torch():
    records = [
        ReviewRecord(
            ku_id=1,
            rating=Rating.GOOD,
            reviewed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            duration_ms=30000,
        )
    ]
    result = optimize_fsrs_parameters(records)
    # With torch installed the optimizer may return defaults for tiny data;
    # without torch it must fail gracefully rather than crash.
    assert result.n_reviews == 1
    assert isinstance(result.message, str)
