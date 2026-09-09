"""Tests for Phase 2F post-purchase core lifecycle diagnosis."""

from ml.core_lifecycle_diagnostic import (
    FATE_LABELS,
    METHODOLOGY_VERSION,
    _classify_fate,
    analyze_core_lifecycles,
    collect_fulfilled_seeded_purchases,
)
from ml.phase_2f_decision import evaluate_phase_2f_decision
from ml.composition_trace import run_traced_rollouts
from hsbg_coach.bg_env import seeded_core_stress_greedy_policy


def test_phase_2f_methodology_version():
    assert METHODOLOGY_VERSION == "2f_v2"
    assert len(FATE_LABELS) == 8


def test_classify_fate_played_never_sold_is_not_sold_later():
    fate = _classify_fate(
        played_turn=5, sell_turn=None, sold_same_turn=False, tripled=False,
        had_two_core_action=False, had_two_core_recruit_end=False,
        target_switched=False, seed_piece_lost=False, seed_cores_at_buy=True,
    )
    assert fate == "C_PLAYED_NO_PERSISTENT_ASSEMBLY"


def test_classify_fate_priority():
    assert _classify_fate(
        played_turn=5, sell_turn=5, sold_same_turn=True, tripled=False,
        had_two_core_action=False, had_two_core_recruit_end=False,
        target_switched=False, seed_piece_lost=False, seed_cores_at_buy=True,
    ) == "B_PLAYED_THEN_SOLD_SAME_TURN"
    assert _classify_fate(
        played_turn=5, sell_turn=None, sold_same_turn=False, tripled=False,
        had_two_core_action=True, had_two_core_recruit_end=False,
        target_switched=False, seed_piece_lost=False, seed_cores_at_buy=True,
    ) == "G_TWO_CORE_TRANSIENT"
    assert _classify_fate(
        played_turn=None, sell_turn=None, sold_same_turn=False, tripled=False,
        had_two_core_action=False, had_two_core_recruit_end=False,
        target_switched=True, seed_piece_lost=False, seed_cores_at_buy=True,
    ) == "A_BOUGHT_STUCK_IN_HAND"


def test_traced_events_include_hand_before():
    traces = run_traced_rollouts(1, seed=1000, scaling_mode="residual")
    assert "hand_before" in traces["events"][0]


def test_lifecycle_smoke_oracle():
    traces = run_traced_rollouts(
        4, seed=1000, policy=seeded_core_stress_greedy_policy,
        scaling_mode="residual")
    lifecycle = analyze_core_lifecycles(traces)
    assert lifecycle["methodology_version"] == METHODOLOGY_VERSION
    assert lifecycle["n_fulfilled_purchases"] >= 0
    if lifecycle["n_fulfilled_purchases"]:
        assert sum(lifecycle["fate_totals"].values()) == lifecycle["n_fulfilled_purchases"]
        for rec in lifecycle["purchases"]:
            assert rec["fate"] in FATE_LABELS


def test_lifecycle_latch_parity_smoke():
    from ml.core_lifecycle_diagnostic import lifecycle_meets_fulfillment_count
    traces = run_traced_rollouts(
        6, seed=1000, policy=seeded_core_stress_greedy_policy,
        scaling_mode="residual")
    assert lifecycle_meets_fulfillment_count(traces)


def test_phase_2f_decision_empty_cohort():
    out = evaluate_phase_2f_decision({"n_fulfilled_purchases": 0, "fate_totals": {}})
    assert out["decision_branch"] == "no_fulfilled_cohort"


def test_phase_2f_decision_hand_stuck():
    n = 10
    totals = {"A_BOUGHT_STUCK_IN_HAND": 6, "C_PLAYED_NO_PERSISTENT_ASSEMBLY": 4}
    out = evaluate_phase_2f_decision({
        "n_fulfilled_purchases": n,
        "fate_totals": totals,
        "funnel": {},
        "board_full_summary": {
            "stuck_in_hand": 6,
            "stuck_in_hand_and_board_full": 6,
            "sold_after_play": 0,
        },
    })
    assert out["decision_branch"] == "board_slot_play_policy"
    assert "full board" in out["recommended_next_step"]


def test_phase_2f_decision_sell_from_flags_not_fate_c():
    out = evaluate_phase_2f_decision({
        "n_fulfilled_purchases": 10,
        "fate_totals": {"C_PLAYED_NO_PERSISTENT_ASSEMBLY": 10},
        "funnel": {},
        "board_full_summary": {"sold_after_play": 4, "stuck_in_hand": 0},
    })
    assert out["decision_branch"] == "retention_aware_sell_policy"


def test_fidelity_phase_2f_runner_smoke(tmp_path):
    from tests.ml_testutil import require_ml
    require_ml()  # contract builder records a Torch/NumPy runtime fingerprint
    from ml.fidelity_phase_2f import run_phase_2f
    result = run_phase_2f(
        lobbies=2, seed=1000, out_dir=str(tmp_path / "p2f"),
        require_clean_tree=False)
    assert result["phase_2f_methodology_version"] == METHODOLOGY_VERSION
    assert result["decision"]["decision_branch"]
