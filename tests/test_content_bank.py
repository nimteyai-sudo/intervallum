from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from intervallum_ml.config import EngineConfig
from intervallum_ml.content import load_content, load_engine
from intervallum_ml.types import (
    Attempt,
    ExamBlueprintItem,
    Rating,
    SelectionContext,
)

ROOT = Path(__file__).parents[1]
BANK = ROOT / "content" / "fipi_physics_2026.json"


def test_real_bank_is_structurally_valid():
    bank = load_content(BANK)
    report = bank.validation_report()
    assert report.ok, [i.message for i in report.errors[:10]]
    assert report.task_count >= 1000
    assert report.ku_count >= 20
    assert report.media_tasks >= 400
    assert len(report.errors) == 0


def test_every_task_has_a_gradeable_answer():
    bank = load_content(BANK)
    for task in bank.tasks.values():
        assert task.answer_raw, task.id
        assert task.answer_type, task.id
        if task.answer_type == "numeric":
            assert task.answer_value is not None, task.id
        elif task.answer_type in {"single_digit", "multi_digit", "code"}:
            assert task.answer_digits, task.id
        elif task.answer_type == "measurement":
            assert task.answer_raw, task.id


def test_media_paths_exist():
    bank = load_content(BANK)
    missing = []
    for task in bank.tasks.values():
        for media in task.media:
            path = Path(media)
            if not path.is_absolute():
                path = ROOT / path
            if not path.exists():
                missing.append((task.id, media))
    assert not missing


def test_no_synthetic_placeholders():
    bank = load_content(BANK)
    assert not any("Synthetic task" in t.text for t in bank.tasks.values())


def test_position_coverage_reports_missing_positions():
    bank = load_content(BANK)
    positions = {
        int(tag.split(":")[1])
        for t in bank.tasks.values()
        for tag in t.tags
        if tag.startswith("ege_position:")
    }
    assert positions == set(range(1, 21)) - {6, 18}


def test_real_bank_selection_and_attempt_smoke():
    engine, bank = load_engine(BANK, EngineConfig())
    states = engine.new_user_state()
    rng = np.random.default_rng(42)
    now = datetime(2026, 10, 31, tzinfo=timezone.utc)
    selected = []
    for step in range(30):
        ctx = SelectionContext(
            now=now,
            session_position=step % 10,
            session_length=10,
            review_count=step // 3,
            days_to_exam=40,
            seed=step,
        )
        result = engine.select(states, ctx)
        assert result.task.id in bank.tasks
        assert 0 < result.propensity <= 1
        assert not result.task.is_quarantined
        selected.append(result.task.id)
        p = result.score.p_correct
        y = float(rng.random() < p)
        attempt = Attempt(
            user_id="u",
            task_id=result.task.id,
            session_id="s",
            timestamp=now,
            raw_score=y,
            duration_ms=int(result.task.t_expected_sec * 1000),
            self_grade=Rating.GOOD if y else Rating.AGAIN,
            state_before=engine.snapshot(states, now),
        )
        engine.review_attempt(states, result.task, attempt)
    assert len(set(selected)) >= 10
    for st in states.values():
        assert math.isfinite(st.theta)
        assert math.isfinite(st.sigma)


def test_real_bank_readiness_projection():
    engine, bank = load_engine(BANK, EngineConfig())
    states = engine.new_user_state()
    # Use the position KUs as a 20-position blueprint.
    position_kus = [100 + i for i in range(1, 21)]
    blueprint = [
        ExamBlueprintItem(
            position=i,
            max_points=1.0,
            ku_ids=(position_kus[i - 1],),
            b_exam=0.0,
            a_exam=1.0,
        )
        for i in range(1, 21)
    ]
    proj = engine.readiness_projection(
        states,
        blueprint,
        datetime(2027, 6, 1, tzinfo=timezone.utc),
        now=datetime(2026, 10, 31, tzinfo=timezone.utc),
    )
    assert 0 <= proj.expected_raw <= 20
    assert 0 <= proj.raw_low <= proj.expected_raw <= proj.raw_high <= 20
    assert 0 <= proj.confidence <= 1


def test_duplicate_task_ids_are_rejected(tmp_path):
    import json

    import pytest

    data = json.loads(BANK.read_text(encoding="utf-8"))
    data["tasks"].append(dict(data["tasks"][0]))
    path = tmp_path / "dup.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate task id"):
        load_content(path)


def test_duplicate_ku_ids_are_rejected(tmp_path):
    import json

    import pytest

    data = json.loads(BANK.read_text(encoding="utf-8"))
    data["knowledge_units"].append(dict(data["knowledge_units"][0]))
    path = tmp_path / "dup.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate knowledge-unit id"):
        load_content(path)


def test_new_user_selection_targets_a_reasonable_success_zone():
    engine, bank = load_engine(BANK, EngineConfig())
    states = engine.new_user_state()
    probs = []
    for i in range(200):
        result = engine.select(
            states,
            SelectionContext(
                session_position=i % 10,
                session_length=10,
                days_to_exam=40,
                seed=i,
            ),
        )
        probs.append(result.score.p_correct)
    mean_p = sum(probs) / len(probs)
    assert 0.60 <= mean_p <= 0.80, mean_p
    assert min(probs) >= 0.20
    assert max(probs) <= 0.95
