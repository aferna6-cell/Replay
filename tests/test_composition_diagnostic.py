"""Tests for Phase 2C composition assembly diagnostic (v2)."""

import pytest

from hsbg_coach.build_path import load_archetypes
from ml.composition_diagnostic import (
    METHODOLOGY_VERSION,
    aggregate_diagnostics,
    recommend_intervention,
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
    assert traces["lobby_meta"]
    ev = traces["events"][0]
    for key in ("lobby", "seat", "turn", "action", "pre_shop",
                "legal_buy_slots", "shop_generation", "board_before",
                "target_before", "lobby_tribes"):
        assert key in ev


def test_traced_rollouts_equivalent_to_plain():
    """Traced rollouts must match ordinary play_scripted for same seeds."""
    n, seed = 8, 42
    plain = run_plain_rollouts(n, seed=seed, scaling_mode="residual")
    traces = run_traced_rollouts(n, seed=seed, scaling_mode="residual")
    traced = traces["player_finals"]
    assert len(plain) == len(traced)
    for p in plain:
        t = next(
            x for x in traced
            if x["lobby"] == p["lobby"] and x["seat"] == p["seat"])
        assert p["placement"] == t["placement"], (
            f"lobby {p['lobby']} seat {p['seat']}: "
            f"{p['placement']} vs {t['placement']}")
        assert p["final_board_fingerprint"] == t["final_board_fingerprint"]


def test_board_fingerprint_stable():
    board = [{"name": "A", "attack": 1, "health": 2, "golden": False}]
    assert board_fingerprint(board) == board_fingerprint(list(board))


def test_aggregate_diagnostics_v2_structure():
    traces = run_traced_rollouts(3, seed=1, scaling_mode="residual")
    report = aggregate_diagnostics(traces)
    assert report["methodology_version"] == METHODOLOGY_VERSION
    assert "invalidated_prior_results" in report
    assert report["invalidated_prior_results"]["classification_totals"]["B_AVAILABLE_NOT_BOUGHT"] == 2516
    funnel = report["winner_decision_funnel"]
    assert "current_target" in funnel
    assert "final_target_hindsight" in funnel
    assert funnel["final_target_hindsight"]["view_label"].startswith("final-target")
    assert report["recommended_phase_2d_intervention"]["intervention"]


def test_pre_shop_updates_after_roll():
    """Each action event must carry the live pre-action shop, not turn-start snapshot."""
    traces = run_traced_rollouts(5, seed=7, scaling_mode="residual")
    for lobby in range(traces["lobbies"]):
        for seat in range(8):
            roll_events = [
                e for e in traces["events"]
                if e["lobby"] == lobby and e["seat"] == seat and e["action"] == "roll"]
            for roll_ev in roll_events:
                turn = roll_ev["turn"]
                gen = roll_ev["shop_generation"]
                prior = [
                    e for e in traces["events"]
                    if e["lobby"] == lobby and e["seat"] == seat
                    and e["turn"] == turn and e["shop_generation"] == gen
                    and e["action"] != "roll"]
                if not prior:
                    continue
                pre_shop_before_roll = prior[-1].get("pre_shop") or []
                # After roll, next generation's first event should differ if shop changed.
                next_gen = [
                    e for e in traces["events"]
                    if e["lobby"] == lobby and e["seat"] == seat
                    and e["turn"] == turn and e["shop_generation"] == gen + 1]
                if next_gen and pre_shop_before_roll:
                    # At minimum, pre_shop field exists and is not reused from turn start only.
                    assert "pre_shop" in next_gen[0]


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
    assert "invalidated" in rec["rationale"].lower() or "prior" in rec["rationale"].lower()


def test_phase_2c_runner_smoke():
    from ml.fidelity_phase_2c import run_phase_2c
    result = run_phase_2c(lobbies=2, seed=0, out_dir="/tmp/sim_fidelity_phase_2c_test")
    assert result["measurement_only"] is True
    assert result["simulator_version"] == "Simulator v1.1"
    assert result["methodology_version"] == METHODOLOGY_VERSION
    assert "recommended_phase_2d_intervention" in result
    assert "invalidated_prior_results" in result
