"""Offline calibration CLI.

Input format (JSON):

tasks.json:
[
  {"id": 1, "mechanic": "mcq_single", "ku_ids": [1, 2],
   "c_guess": 0.25, "t_expected_sec": 60, "b_difficulty": 0.0,
   "a_discrimination": 1.0, "expert_difficulty": 3}
]

attempts.json:
[
  {"user_id": "u1", "task_id": 1, "session_id": "s1",
   "timestamp": "2026-10-31T10:00:00+00:00", "raw_score": 1.0,
   "duration_ms": 42000, "self_grade": 3, "hint_weight_sum": 0.0,
   "state_before": {
      "1": {"theta": 0.1, "sigma": 0.9, "stability": 4.2,
            "difficulty": 5.0, "retrievability": 0.76}
   }}
]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from intervallum_ml.calibration import ItemCalibrator
from intervallum_ml.coldstart import MetaDifficultyModel
from intervallum_ml.model_store import ModelArtifact, save_model
from intervallum_ml.types import (
    Attempt,
    Rating,
    StateSnapshot,
    Task,
    TaskMechanic,
)


def _load_tasks(path: Path) -> dict[int, Task]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    tasks = {}
    for row in rows:
        tasks[int(row["id"])] = Task(
            id=int(row["id"]),
            mechanic=TaskMechanic(row["mechanic"]),
            ku_ids=tuple(int(x) for x in row["ku_ids"]),
            ku_weights=tuple(float(x) for x in row.get("ku_weights", [])),
            c_guess=float(row.get("c_guess", 0.0)),
            t_expected_sec=float(row.get("t_expected_sec", 60.0)),
            b_difficulty=float(row.get("b_difficulty", 0.0)),
            a_discrimination=float(row.get("a_discrimination", 1.0)),
            expert_difficulty=int(row.get("expert_difficulty", 3)),
            is_quarantined=bool(row.get("is_quarantined", False)),
            topic_id=row.get("topic_id"),
            section_id=row.get("section_id"),
            text=row.get("text", ""),
            source=row.get("source", ""),
        )
    return tasks


def _load_attempts(path: Path) -> list[Attempt]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    attempts = []
    for row in rows:
        ts = datetime.fromisoformat(row["timestamp"])
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        snapshots = {}
        for k, v in row.get("state_before", {}).items():
            snapshots[int(k)] = StateSnapshot(
                theta=float(v["theta"]),
                sigma=float(v.get("sigma", 1.0)),
                stability=float(v.get("stability", 0.0)),
                difficulty=float(v.get("difficulty", 5.0)),
                retrievability=float(v.get("retrievability", 0.0)),
            )
        attempts.append(
            Attempt(
                user_id=str(row["user_id"]),
                task_id=int(row["task_id"]),
                session_id=str(row.get("session_id", "unknown")),
                timestamp=ts,
                raw_score=float(row["raw_score"]),
                duration_ms=int(row["duration_ms"]),
                self_grade=Rating(int(row.get("self_grade", 3))),
                hints_used=tuple(int(x) for x in row.get("hints_used", [])),
                hint_weight_sum=float(row.get("hint_weight_sum", 0.0)),
                pos_in_session=int(row.get("pos_in_session", 0)),
                session_elapsed_sec=float(row.get("session_elapsed_sec", 0.0)),
                state_before=snapshots,
                propensity_score=float(row.get("propensity_score", 1.0)),
                policy_version=str(row.get("policy_version", "unknown")),
                repeat_same_task=bool(row.get("repeat_same_task", False)),
            )
        )
    return attempts


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate Intervallum item parameters")
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--attempts", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("model.json"))
    parser.add_argument("--tau0", type=float, default=2.0)
    parser.add_argument("--gamma0", type=float, default=0.8)
    parser.add_argument("--no-meta", action="store_true")
    args = parser.parse_args()

    tasks = _load_tasks(args.tasks)
    attempts = _load_attempts(args.attempts)
    meta = None if args.no_meta else MetaDifficultyModel().fit(list(tasks.values()))
    report = ItemCalibrator(meta=meta).fit(
        tasks, attempts, tau0=args.tau0, gamma0=args.gamma0
    )
    artifact = ModelArtifact.from_report(report)
    save_model(args.out, artifact)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "version": artifact.version,
                "n_attempts": report.n_attempts,
                "n_items": report.n_items,
                "tau": report.tau,
                "gamma": report.gamma,
                "brier": report.brier,
                "ece": report.ece,
                "converged": report.converged,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
