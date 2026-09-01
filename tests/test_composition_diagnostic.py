"""Tests for Phase 2C composition assembly diagnostic (v3)."""

import pytest

from hsbg_coach.build_path import load_archetypes
from ml.composition_diagnostic import (
    METHODOLOGY_VERSION,
    _WinnerFunnelState,
    aggregate_diagnostics,
    analyze_winner_funnel,
)
from ml.composition_trace import (
    board_fingerprint,
    run_plain_rollouts,
    run_traced_rollouts,
)


def test_traced_rollouts_smoke():
    traces = run_traced_rollouts(2, seed=0, scaling_mode="residual")
    assert traces["lobbies"] == 2
    assert len(traces["events"]) > 0
    assert len(traces["player_finals"]) == 16
    ev = traces["events"][0]
    for key in ("lobby", "seat", "turn", "action", "pre_shop",
                "legal_buy_slots", "shop_generation", "board_before",
                "target_before", "lobby_tribes"):
        assert key in ev


def test_traced_rollouts_equivalent_to_plain():
    n, seed = 8, 42
    plain = run_plain_rollouts(n, seed=seed, scaling_mode="residual")
    traces = run_traced_rollouts(n, seed=seed, scaling_mode="residual")
    traced = traces["player_finals"]
    assert len(plain) == len(traced)
    for p in plain:
        t = next(
            x for x in traced
            if x["lobby"] == p["lobby"] and x["seat"] == p["seat"])
        assert p["placement"] == t["placement"]
        assert p["final_board_fingerprint"] == t["final_board_fingerprint"]


def test_exposure_accounting_invariant():
    """fulfilled + rejected == legally_buyable for every funnel state."""
    traces = run_traced_rollouts(5, seed=3, scaling_mode="residual")
    arch = load_archetypes()[0]
    for view in ("broad_current_target", "seeded_current_target",
                 "committed_current_target", "final_target_hindsight"):
        for lobby in range(traces["lobbies"]):
            state = analyze_winner_funnel(traces, lobby, arch, view)
            if state is None:
                continue
            assert state.exposure_accounting_valid, (
                f"lobby={lobby} view={view} "
                f"fulfilled={state.fulfilled_exposures} "
                f"rejected={state.rejected_exposures} "
                f"legal={state.legally_buyable_exposures}")
    report = aggregate_diagnostics(traces)
    for view_name, view_data in report["winner_decision_funnel"].items():
        funnel = view_data.get("aggregate_funnel") or {}
        if not funnel:
            continue
        assert funnel.get("exposure_accounting_valid") is True, view_name


def test_purchase_fulfills_latched_exposure():
    """Purchase credits active generation even if infer_target changed."""
    state = _WinnerFunnelState(core={"CoreX"}, lobby_tribes=["Naga"])
    state.open_generation(5, 0, "naga_end_of_turn", {
        "CoreX": {"name": "CoreX", "attack": 2, "health": 2},
    })
    assert state.legally_buyable_exposures == 1
    state.note_core_purchase("CoreX", 5)
    state.close_generation()
    assert state.fulfilled_exposures == 1
    assert state.rejected_exposures == 0
    assert state.exposure_accounting_valid


def test_aggregate_diagnostics_v3_structure():
    traces = run_traced_rollouts(3, seed=1, scaling_mode="residual")
    report = aggregate_diagnostics(traces)
    assert report["methodology_version"] == METHODOLOGY_VERSION
    funnel = report["winner_decision_funnel"]
    for key in ("broad_current_target", "seeded_current_target",
                "committed_current_target", "final_target_hindsight"):
        assert key in funnel
        assert "view_label" in funnel[key]
    rec = report["recommended_phase_2d_intervention"]
    assert "funnel_seeded_current_target" in rec
    assert "funnel_committed_current_target" in rec


def test_seeded_subset_of_broad():
    traces = run_traced_rollouts(4, seed=9, scaling_mode="residual")
    report = aggregate_diagnostics(traces)
    broad = report["winner_decision_funnel"]["broad_current_target"]["aggregate_funnel"]
    seeded = report["winner_decision_funnel"]["seeded_current_target"]["aggregate_funnel"]
    if broad and seeded:
        assert seeded["legally_buyable_exposures"] <= broad["legally_buyable_exposures"]


def test_phase_2c_runner_smoke():
    from ml.fidelity_phase_2c import run_phase_2c
    result = run_phase_2c(
        lobbies=2, seed=0, out_dir="/tmp/sim_fidelity_phase_2c_test",
        require_clean_tree=False)
    assert result["measurement_only"] is True
    assert result["methodology_version"] == METHODOLOGY_VERSION
    assert result["implementation_commit"]
    assert "recommended_phase_2d_intervention" in result
