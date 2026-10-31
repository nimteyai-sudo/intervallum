from __future__ import annotations

from datetime import datetime, timezone

from intervallum_ml.model_store import ModelArtifact, load_model, save_model
from intervallum_ml.state_store import (
    state_from_dict,
    state_to_dict,
    states_from_dict,
    states_to_dict,
)
from intervallum_ml.types import CalibrationReport, MemoryState, UserKUState


def test_state_roundtrip():
    st = UserKUState(
        ku_id=1,
        theta=1.2,
        sigma=0.4,
        memory=MemoryState(
            stability=12.5,
            difficulty=4.2,
            state=2,
            step=None,
            last_review_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            due_at=datetime(2026, 1, 10, tzinfo=timezone.utc),
        ),
        attempts_count=7,
        last_seen_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )
    data = state_to_dict(st)
    restored = state_from_dict(data)
    assert restored.ku_id == st.ku_id
    assert restored.theta == st.theta
    assert restored.sigma == st.sigma
    assert restored.memory is not None
    assert restored.memory.stability == st.memory.stability
    assert restored.memory.due_at == st.memory.due_at


def test_states_roundtrip():
    states = {1: UserKUState(1, theta=0.5), 2: UserKUState(2, theta=-0.5)}
    data = states_to_dict(states)
    restored = states_from_dict(data)
    assert set(restored) == {1, 2}
    assert restored[1].theta == 0.5


def test_model_artifact_roundtrip(tmp_path):
    report = CalibrationReport(
        b={1: 0.3, 2: -0.4},
        a={1: 1.2, 2: 0.9},
        tau=2.1,
        gamma=0.8,
        loss=12.3,
        brier=0.18,
        ece=0.03,
        n_attempts=1000,
        n_items=2,
        converged=True,
    )
    artifact = ModelArtifact.from_report(report, version="test")
    path = save_model(tmp_path / "model.json", artifact)
    loaded = load_model(path)
    assert loaded.version == "test"
    assert loaded.tau == 2.1
    assert loaded.b[1] == 0.3
    assert loaded.metrics["brier"] == 0.18
