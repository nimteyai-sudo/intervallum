from __future__ import annotations

from dataclasses import asdict

from intervallum_ml.simulation import build_world
from intervallum_ml.user_sim import (
    make_edge_profiles,
    make_profiles,
    run_cohort,
    simulate_student,
    summarise,
    world_from_content,
)


def test_simulator_is_deterministic():
    world = build_world(n_users=2, n_kus=6, n_tasks=20, seed=1)
    profile = make_profiles(world, seed=2)[0]
    a = simulate_student(world, profile, policy="adaptive", sessions=2, tasks_per_session=6, seed=3)
    b = simulate_student(world, profile, policy="adaptive", sessions=2, tasks_per_session=6, seed=3)
    assert asdict(a) == asdict(b)


def test_adaptive_policy_beats_random_on_small_cohort():
    world = build_world(n_users=8, n_kus=10, n_tasks=40, seed=1)
    profiles = make_profiles(world, seed=2)
    adaptive = summarise(
        run_cohort(world, profiles, policy="adaptive", sessions=3, tasks_per_session=8, seed=3)
    )
    random = summarise(
        run_cohort(world, profiles, policy="random", sessions=3, tasks_per_session=8, seed=3)
    )
    assert adaptive["mean_true_gain"] > random["mean_true_gain"]


def test_edge_profiles_behave_differently():
    world = build_world(n_users=1, n_kus=8, n_tasks=30, seed=4)
    profiles = {p.label: p for p in make_edge_profiles("u0")}
    strong = simulate_student(world, profiles["strong"], policy="adaptive", sessions=2, tasks_per_session=8, seed=5)
    guesser = simulate_student(world, profiles["guesser"], policy="adaptive", sessions=2, tasks_per_session=8, seed=5)
    assert strong.true_gain > guesser.true_gain
    assert strong.mean_true_p >= guesser.mean_true_p


def test_world_from_content_has_latent_truth(tmp_path):
    from pathlib import Path

    from intervallum_ml.content import load_content

    bank = load_content(Path(__file__).parents[1] / "content" / "fipi_physics_2026.json")
    world = world_from_content(bank, n_users=3, seed=6)
    assert len(world.users) == 3
    assert len(world.knowledge_units) >= 20
    assert len(world.tasks) >= 1000
    assert world.true_theta
    assert world.true_b
