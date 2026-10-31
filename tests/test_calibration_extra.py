from __future__ import annotations

from datetime import datetime, timezone

from intervallum_ml.calibration import calibrate_exam_blueprint
from intervallum_ml.memory import MemoryScheduler
from intervallum_ml.types import ExamBlueprintItem, MemoryState, Rating


def test_memory_fallback_keeps_state_machine():
    s = MemoryScheduler()
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    good = s.fallback_review(MemoryState(), Rating.GOOD, now)
    assert good.stability > 0
    again = s.fallback_review(good, Rating.AGAIN, datetime(2026, 1, 2, tzinfo=timezone.utc))
    assert again.state == 3
    assert again.stability < good.stability


def test_blueprint_calibration_shifts_difficulty():
    blueprint = [
        ExamBlueprintItem(position=i, max_points=2, ku_ids=(1,), b_exam=0.0)
        for i in range(1, 5)
    ]
    shifted, delta = calibrate_exam_blueprint(
        blueprint,
        predicted_raw=[10.0, 20.0, 30.0],
        actual_raw=[15.0, 25.0, 35.0],
    )
    assert delta > 0
    assert shifted[0].b_exam < blueprint[0].b_exam
