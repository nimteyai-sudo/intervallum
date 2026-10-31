from __future__ import annotations

from fastapi.testclient import TestClient

from intervallum_ml.api import create_app
from intervallum_ml.config import EngineConfig
from intervallum_ml.engine import IntervallumEngine
from intervallum_ml.types import KnowledgeUnit, KUType, Task, TaskMechanic


def make_app():
    kus = {i: KnowledgeUnit(i, f"ku{i}", f"KU {i}", KUType.HYBRID, 1, 1) for i in range(1, 4)}
    tasks = {
        1: Task(1, TaskMechanic.MCQ_SINGLE, (1,), c_guess=0.25, t_expected_sec=30),
        2: Task(2, TaskMechanic.NUMERIC, (2,), t_expected_sec=60),
    }
    engine = IntervallumEngine(kus, tasks, EngineConfig())
    return create_app(engine)


def test_health_and_tasks():
    client = TestClient(make_app())
    assert client.get("/health").json()["status"] == "ok"
    assert len(client.get("/v1/tasks").json()) == 2


def test_select_and_attempt_flow():
    client = TestClient(make_app())
    sel = client.post(
        "/v1/select",
        json={"user_id": "u1", "session_position": 0, "days_to_exam": 30, "seed": 1},
    )
    assert sel.status_code == 200, sel.text
    data = sel.json()
    task_id = data["task"]["id"]
    assert data["propensity"] > 0

    resp = client.post(
        "/v1/attempt",
        json={
            "user_id": "u1",
            "task_id": task_id,
            "raw_score": 1.0,
            "duration_ms": 20000,
            "self_grade": 3,
            "timestamp": "2026-01-01T00:00:00+00:00",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["y_eff"] == 1.0
    assert body["g_sync"] in (3, 4)


def test_readiness_endpoint():
    client = TestClient(make_app())
    payload = {
        "user_id": "u1",
        "exam_date": "2027-06-01T00:00:00+00:00",
        "blueprint": [
            {"position": i, "max_points": 2, "ku_ids": [1, 2], "b_exam": 0, "a_exam": 1}
            for i in range(1, 27)
        ],
    }
    resp = client.post("/v1/readiness", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["raw_low"] <= body["expected_raw"] <= body["raw_high"]


def test_default_app_uses_real_content_bank():
    client = TestClient(create_app())
    health = client.get("/health").json()
    assert health["tasks"] >= 1000
    assert health["knowledge_units"] >= 20


def test_grade_endpoint_and_answer_based_attempt():
    # The test app tasks have no answer metadata, so use the real default app
    # for the grading path.
    real = TestClient(create_app())
    # Find a task with a known single-digit answer by using the content bank.
    from pathlib import Path as P

    from intervallum_ml.content import load_content

    bank = load_content(P(__file__).parents[1] / "content" / "fipi_physics_2026.json")
    sample = next(t for t in bank.tasks.values() if t.answer_type == "single_digit")
    resp = real.post("/v1/grade", json={"task_id": sample.id, "answer": sample.answer_raw})
    assert resp.status_code == 200, resp.text
    assert resp.json()["raw_score"] == 1.0
    assert resp.json()["correct"] is True
    selected = real.post("/v1/select", json={"user_id": "real-user", "seed": 1})
    assert selected.status_code == 200, selected.text
    assert selected.json()["task"]["id"] in {t.id for t in bank.tasks.values()}
    attempt = real.post(
        "/v1/attempt",
        json={
            "user_id": "grader",
            "task_id": sample.id,
            "answer": sample.answer_raw,
            "duration_ms": 60000,
            "self_grade": 3,
        },
    )
    assert attempt.status_code == 200, attempt.text
    assert attempt.json()["y_eff"] == 1.0
