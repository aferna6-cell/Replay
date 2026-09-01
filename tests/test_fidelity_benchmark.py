"""Tests for Simulator Fidelity Benchmark v1."""

import json
import os

import pytest

from ml.fidelity_benchmark import run_benchmark
from ml.fidelity_metrics import aggregate_turn_curves, run_fidelity_rollouts
from ml.fidelity_reference import (FIDELITY_BENCHMARK_VERSION,
                                     build_simulator_v1_contract,
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
    assert c["code_commit"]


def test_run_benchmark_smoke():
    result = run_benchmark(lobbies=2, seed=0)
    assert result["benchmark"] == FIDELITY_BENCHMARK_VERSION
    assert "turn_curves" in result
    assert "composition" in result
    assert result["composition"]["sim_coverage_n"] > 0
    assert result["combat"]["measured_in_v1"] is False


def test_turn_curve_relative_error():
    rows = run_fidelity_rollouts(1, seed=0)
    curves = aggregate_turn_curves(rows)
    t14 = curves.get("14")
    if t14 and t14.get("real_board_stats"):
        assert t14["stats_ratio_sim_over_real"] > 0


@pytest.mark.skipif(not os.path.isfile("results/sim_fidelity_v1/baseline.json"),
                    reason="baseline not generated yet")
def test_committed_baseline_schema():
    data = json.load(open("results/sim_fidelity_v1/baseline.json"))
    assert data["benchmark"] == FIDELITY_BENCHMARK_VERSION
    assert "turn_curves" in data
    assert data["turn_curves"]["14"]["stats_ratio_sim_over_real"] > 1.0
