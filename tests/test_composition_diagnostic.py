"""Tests for Phase 2C composition assembly diagnostic."""

import pytest

from hsbg_coach.build_path import load_archetypes
from ml.composition_diagnostic import aggregate_diagnostics, recommend_intervention
from ml.composition_trace import run_traced_rollouts


def test_traced_rollouts_smoke():
    traces = run_traced_rollouts(2, seed=0, scaling_mode="residual")
    assert traces["lobbies"] == 2
    assert len(traces["events"]) > 0
    assert len(traces["player_finals"]) == 16
    ev = traces["events"][0]
    for key in ("lobby", "seat", "turn", "action", "shop_offered",
                "board_before", "board_after", "target"):
        assert key in ev


def test_aggregate_diagnostics_structure():
    traces = run_traced_rollouts(3, seed=1, scaling_mode="residual")
    report = aggregate_diagnostics(traces)
    assert report["n_lobbies"] == 3
    assert report["by_archetype"]
    assert report["recommended_phase_2d_intervention"]["intervention"]
    arch = load_archetypes()[0]
    row = report["by_archetype"][arch.key]
    for section in ("availability", "conversion", "assembly", "funnel", "classification"):
        assert section in row


def test_recommend_intervention_returns_one_choice():
    traces = run_traced_rollouts(2, seed=2)
    diag = aggregate_diagnostics(traces)
    rec = diag["recommended_phase_2d_intervention"]
    assert rec["intervention"] in (
        "shop_pool_fidelity",
        "build_aware_recruit_policy",
        "card_effect_fidelity",
        "triple_discover_fidelity",
    )
    assert "phase_2d_title" in rec
    assert "rationale" in rec


def test_phase_2c_runner_smoke():
    from ml.fidelity_phase_2c import run_phase_2c
    result = run_phase_2c(lobbies=2, seed=0, out_dir="/tmp/sim_fidelity_phase_2c_test")
    assert result["measurement_only"] is True
    assert result["simulator_version"] == "Simulator v1.1"
    assert "recommended_phase_2d_intervention" in result
