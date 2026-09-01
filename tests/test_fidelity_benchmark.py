"""Tests for Simulator Fidelity Benchmark v1."""

import json
import os

import pytest

from hsbg_coach.pace import load_pace
from ml.fidelity_benchmark import run_benchmark
from ml.fidelity_metrics import (aggregate_composition, aggregate_lobby_dynamics,
                                   aggregate_turn_curves, run_fidelity_rollouts)
from ml.fidelity_reference import (FIDELITY_BENCHMARK_VERSION,
                                     build_simulator_v1_contract,
                                     reference_at_exact,
                                     reference_fingerprints)


def test_reference_fingerprints_present():
    fps = reference_fingerprints()
    for key in ("firestone_pace", "firestone_final_boards", "bg_cards"):
        assert key in fps
        assert len(fps[key]) == 64


def test_simulator_v1_contract_fields():
    c = build_simulator_v1_contract(lobbies=10, evaluation_seed=0)
    assert c["fidelity_benchmark_version"] == FIDELITY_BENCHMARK_VERSION
    assert c["simulator_version"] == "Simulator v1"
    assert c["environment"]["env"] == "hsbg_coach.bg_env.BGEnv"
    assert "reference_data_fingerprints" in c
    assert "reference_label" in c["reference_metadata"]
    assert c["code_commit"]


def test_scaling_reference_unmeasured_turns_15_16():
    scaling = load_pace().get("scaling", {})
    assert reference_at_exact(scaling, 14) is not None
    assert reference_at_exact(scaling, 15) is None
    assert reference_at_exact(scaling, 16) is None


def test_game_length_one_per_lobby():
    rows = [
        {"lobby": 0, "turn": 1, "players_alive": 8.0},
        {"lobby": 0, "turn": 10, "players_alive": 2.0},
        {"lobby": 0, "turn": 5, "players_alive": 5.0},
        {"lobby": 1, "turn": 1, "players_alive": 8.0},
        {"lobby": 1, "turn": 20, "players_alive": 1.0},
    ]
    ld = aggregate_lobby_dynamics(rows)
    assert ld["avg_game_length"] == 15.0
    assert ld["median_game_length"] == 15.0
    assert ld["n_lobbies"] == 2


def test_turn_curve_no_extrapolation():
    rows = run_fidelity_rollouts(2, seed=0)
    curves = aggregate_turn_curves(rows)
    for t in (15, 16):
        row = curves.get(str(t))
        if not row:
            continue
        assert row["real_board_stats"] is None
        assert row["real_board_stats_status"] == "unmeasured"
        assert row["stats_ratio_sim_over_real"] is None
        assert row["stats_relative_error"] is None


def test_run_benchmark_smoke():
    result = run_benchmark(lobbies=2, seed=0)
    assert result["benchmark"] == FIDELITY_BENCHMARK_VERSION
    assert "reference_label" in result
    assert "turn_curves" in result
    comp = result["composition"]
    assert "sim_midgame_to_final_winner_coverage" in comp
    assert "final_winner_coverage" in comp
    assert comp["sim_midgame_to_final_winner_coverage"]["sim_n"] > 0
    assert comp["final_winner_coverage"]["sim_n"] > 0
    assert result["combat"]["measured_in_v1"] is False


def test_turn_curve_relative_error():
    rows = run_fidelity_rollouts(1, seed=0)
    curves = aggregate_turn_curves(rows)
    t14 = curves.get("14")
    if t14 and t14.get("real_board_stats"):
        assert t14["stats_ratio_sim_over_real"] > 0
        assert t14["real_board_stats_status"] == "measured"


def test_final_winner_composition_structure():
    rows = run_fidelity_rollouts(3, seed=0)
    comp = aggregate_composition(rows)
    final = comp["final_winner_coverage"]
    assert final["sim_n"] == 3
    assert final["real_n"] > 0
    assert len(final["sim_distribution"]) == final["sim_n"]


@pytest.mark.skipif(not os.path.isfile("results/sim_fidelity_v1/baseline.json"),
                    reason="baseline not generated yet")
def test_committed_baseline_schema():
    data = json.load(open("results/sim_fidelity_v1/baseline.json"))
    assert data["benchmark"] == FIDELITY_BENCHMARK_VERSION
    assert "reference_label" in data
    assert "turn_curves" in data
    assert data["turn_curves"]["14"]["stats_ratio_sim_over_real"] > 1.0
    t16 = data["turn_curves"].get("16")
    if t16:
        assert t16.get("real_board_stats") is None
        assert t16.get("stats_ratio_sim_over_real") is None
    assert "final_winner_coverage" in data["composition"]
