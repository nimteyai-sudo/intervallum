"""Command-line demo for the ML core."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from .calibration import ItemCalibrator, detect_bad_items
from .coldstart import MetaDifficultyModel
from .config import EngineConfig
from .engine import IntervallumEngine
from .simulation import build_world, recovery_report, simulate
from .types import ExamBlueprintItem, SelectionContext


def _demo(args: argparse.Namespace) -> None:
    world = build_world(n_users=args.users, n_kus=args.kus, n_tasks=args.tasks, seed=args.seed)
    result = simulate(world, attempts_per_user=args.attempts, seed=args.seed + 1)
    print("=== Synthetic world ===")
    print(f"users={len(world.users)} kus={len(world.knowledge_units)} tasks={len(world.tasks)}")
    print(f"attempts={len(result.attempts)}")
    print(f"recovery={json.dumps(recovery_report(result), indent=2)}")

    engine = IntervallumEngine(world.knowledge_units, world.tasks, EngineConfig())
    states = engine.new_user_state()
    ctx = SelectionContext(
        now=datetime(2026, 10, 31, tzinfo=timezone.utc),
        session_position=0,
        days_to_exam=40,
        seed=args.seed,
    )
    selection = engine.select(states, ctx)
    print("\n=== Selection ===")
    print(
        json.dumps(
            {
                "task_id": selection.task.id,
                "mechanic": selection.task.mechanic.value,
                "utility": selection.score.utility,
                "p_correct": selection.score.p_correct,
                "reason": selection.score.reason.value,
                "propensity": selection.propensity,
                "candidates": selection.candidates,
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    # Calibrate on synthetic attempts.  This is a smoke test, not a claim of
    # high accuracy on real data.
    meta = MetaDifficultyModel().fit(list(world.tasks.values()))
    calibrator = ItemCalibrator(meta=meta)
    report = calibrator.fit(world.tasks, result.attempts, tau0=world.true_tau, gamma0=world.true_gamma)
    print("\n=== Calibration ===")
    print(
        json.dumps(
            {
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
    bad = detect_bad_items(world.tasks, result.attempts)
    print(f"flagged_items={len([x for x in bad if x.quarantine])}")

    # A toy 26-position blueprint with equal weights.
    ku_ids = list(world.knowledge_units)[: min(8, len(world.knowledge_units))]
    blueprint = [
        ExamBlueprintItem(
            position=i,
            max_points=2.0,
            ku_ids=(ku_ids[i % len(ku_ids)],),
            b_exam=0.0,
            a_exam=1.0,
        )
        for i in range(1, 27)
    ]
    proj = engine.readiness_projection(
        states,
        blueprint,
        datetime(2027, 6, 1, tzinfo=timezone.utc),
        now=datetime(2026, 10, 31, tzinfo=timezone.utc),
    )
    print("\n=== Readiness ===")
    print(
        json.dumps(
            {
                "expected_raw": round(proj.expected_raw, 2),
                "raw_interval": [round(proj.raw_low, 2), round(proj.raw_high, 2)],
                "expected_scaled": round(proj.expected_scaled, 1),
                "scaled_interval": [round(proj.scaled_low, 1), round(proj.scaled_high, 1)],
                "confidence": round(proj.confidence, 3),
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Intervallum ML demo")
    parser.add_argument("--users", type=int, default=30)
    parser.add_argument("--kus", type=int, default=24)
    parser.add_argument("--tasks", type=int, default=120)
    parser.add_argument("--attempts", type=int, default=25)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    _demo(args)


if __name__ == "__main__":
    main()
