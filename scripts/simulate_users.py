"""Run the behavioural student simulator on the real EGE content bank.

Example:
    python scripts/simulate_users.py --students 12 --sessions 4 \
        --tasks 10 --policies random adaptive easy hard topic_order
"""

from __future__ import annotations

import argparse
import json

from intervallum_ml.content import load_content
from intervallum_ml.user_sim import (
    make_profiles,
    run_cohort,
    summarise,
    world_from_content,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--content", default="content/fipi_physics_2026.json")
    parser.add_argument("--students", type=int, default=12)
    parser.add_argument("--sessions", type=int, default=4)
    parser.add_argument("--tasks", type=int, default=10)
    parser.add_argument("--initial-noise", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["random", "adaptive"],
        choices=["random", "adaptive", "easy", "hard", "topic_order"],
    )
    args = parser.parse_args()

    bank = load_content(args.content)
    world = world_from_content(bank, n_users=args.students, seed=args.seed)
    profiles = make_profiles(world, seed=args.seed + 1)
    report = {}
    for i, policy in enumerate(args.policies):
        results = run_cohort(
            world,
            profiles,
            policy=policy,
            sessions=args.sessions,
            tasks_per_session=args.tasks,
            initial_noise=args.initial_noise,
            seed=args.seed + 100 + i,
        )
        report[policy] = summarise(results)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
