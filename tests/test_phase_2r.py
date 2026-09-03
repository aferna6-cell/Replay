"""Tests for Phase 2R replacement-collapse mechanism diagnostic."""

from hsbg_coach.bg_env import (
    A_BUY0,
    A_PLAY0,
    A_SELL0,
    PHASE_2Q_RECRUIT_VALUE_STATS,
    EnvMinion,
    recruit_value_stats_enabled,
)
from ml.collapse_mechanism_diagnostic import (
    FORBIDDEN_RANGES,
    METHODOLOGY_VERSION,
    PHASE_2R_SEED,
    CollapseMechanismTracer,
    assert_seed_range_allowed,
    compare_control_treatment,
    diagnose_phase_2r,
    run_greedy_control,
    run_greedy_treatment,
    summarize_collapse_arm,
)


def test_methodology_is_2r_v1():
    assert METHODOLOGY_VERSION == "2r_v1"


def test_toggle_default_remains_off():
    assert PHASE_2Q_RECRUIT_VALUE_STATS is False
    with recruit_value_stats_enabled(True):
        from hsbg_coach.bg_env import PHASE_2Q_RECRUIT_VALUE_STATS as inner
        assert inner is True
    assert PHASE_2Q_RECRUIT_VALUE_STATS is False


def test_phase_2r_forbidden_ranges_cover_confirm_2q_and_priors():
    assert (11500, 11699) in FORBIDDEN_RANGES
    assert (13200, 13699) in FORBIDDEN_RANGES
    assert (12700, 13199) in FORBIDDEN_RANGES
    assert (12200, 12699) in FORBIDDEN_RANGES


def test_phase_2r_seed_guard():
    try:
        assert_seed_range_allowed(11500, 10)
        raise AssertionError("expected confirm-band overlap failure")
    except ValueError:
        pass
    try:
        assert_seed_range_allowed(13200, 500)
        raise AssertionError("expected 2Q-band overlap failure")
    except ValueError:
        pass
    assert_seed_range_allowed(PHASE_2R_SEED, 500)


def _view(m: EnvMinion) -> dict:
    return m.view()


def test_tracer_records_sell_buy_play_combat_loss():
    """Full-board sell → buy → play: incumbent combat 70, printed play 7."""
    tracer = CollapseMechanismTracer(0, 13700, "unit")
    board = [
        EnvMinion("a", "Scaled", 3, 40, 30, [], [],
                  recruit_attack=4, recruit_health=3)
    ] + [
        EnvMinion("b", f"Fill{i}", 2, 5, 5, [], [])
        for i in range(6)
    ]
    shop = [EnvMinion("c", "Shop", 4, 4, 3, [], [])]
    hand: list = []

    class _P:
        def __init__(self):
            self.board = list(board)
            self.hp = 30
            self.tier = 4

        def strength(self):
            return sum(m.attack + m.health for m in self.board)

    p = _P()
    tracer.begin_seat_recruit(0, 10, p)

    obs = {
        "board": [_view(m) for m in p.board],
        "shop": [_view(m) for m in shop],
        "hand": [],
    }
    tracer.before_action(0, 10, 0, obs, [True] * 28)
    # Sell the scaled incumbent (slot 0) while board is full.
    sold = p.board.pop(0)
    tracer.after_action(0, 10, 0, A_SELL0 + 0, False, p)

    obs2 = {
        "board": [_view(m) for m in p.board],
        "shop": [_view(m) for m in shop],
        "hand": [],
    }
    tracer.before_action(0, 10, 0, obs2, [True] * 28)
    bought = shop.pop(0)
    hand.append(bought)
    tracer.after_action(0, 10, 0, A_BUY0 + 0, False, p)

    obs3 = {
        "board": [_view(m) for m in p.board],
        "shop": [],
        "hand": [_view(m) for m in hand],
    }
    tracer.before_action(0, 10, 0, obs3, [True] * 28)
    played = hand.pop(0)
    p.board.append(played)
    tracer.after_action(0, 10, 0, A_PLAY0 + 0, False, p)
    tracer.end_seat_recruit(0, 10, p)

    assert len(tracer.replacement_events) == 1
    ev = tracer.replacement_events[0]
    assert ev["incumbent_combat"] == 70.0
    assert ev["incumbent_recruit"] == 7.0
    assert ev["candidate_recruit"] == 7.0
    assert ev["combat_loss"] == 63.0
    assert ev["recruit_delta"] == 0.0
    row = tracer._churn[(0, 10)]
    assert row["completed_replacements"] == 1
    assert row["combat_removed"] == 70.0
    assert row["recruit_gain_combat"] == 7.0
    assert sold.name == "Scaled"


def test_diagnose_routes_replacement_churn_loss():
    control = {
        "n_replacement_events": 100,
        "replacement_loss_distribution": {"mean": 5.0},
        "headline_t9_t12": {
            "mean_replacement_net": -2.0,
            "mean_combat_removed": 4.0,
            "mean_residual": 500.0,
        },
        "post_scale_fidelity": {
            "10": {"mean_post_scale_over_firestone": 0.95},
            "14": {"mean_post_scale_over_firestone": 1.8},
        },
        "symmetric_absolute_fidelity_turns_8_14": {
            "10": {"mean_post_scale_over_firestone": 0.95},
            "12": {"mean_post_scale_over_firestone": 1.4},
            "14": {"mean_post_scale_over_firestone": 1.8},
        },
        "lobby_dynamics": {
            "avg_game_length": 15.6,
            "sim_alive_by_turn": {str(t): 6.0 for t in range(8, 15)},
        },
        "per_turn_decomposition": {
            str(t): {
                "n_replacements": 20,
                "start_of_recruit": 900.0,
                "replacement_net_combat": -5.0,
                "other_recruit_delta": 0.0,
                "residual_scaling_recovery": 500.0,
                "post_scale": 1400.0,
                "residual_shrinkage_from_crater": 10.0,
            }
            for t in range(8, 15)
        },
    }
    treatment = {
        "n_replacement_events": 4000,
        "replacement_loss_distribution": {"mean": 80.0},
        "headline_t9_t12": {
            "mean_replacement_net": -280.0,
            "mean_combat_removed": 300.0,
            "mean_residual": 200.0,
        },
        "post_scale_fidelity": {
            "10": {"mean_post_scale_over_firestone": 0.47},
            "14": {"mean_post_scale_over_firestone": 0.11},
        },
        "symmetric_absolute_fidelity_turns_8_14": {
            "10": {"mean_post_scale_over_firestone": 0.47},
            "12": {"mean_post_scale_over_firestone": 0.16},
            "14": {"mean_post_scale_over_firestone": 0.11},
        },
        "lobby_dynamics": {
            "avg_game_length": 13.1,
            "sim_alive_by_turn": {str(t): 4.0 for t in range(8, 15)},
        },
        "per_turn_decomposition": {
            str(t): {
                "n_replacements": 400,
                "start_of_recruit": 850.0,
                "replacement_net_combat": -520.0,
                "other_recruit_delta": 0.0,
                "residual_scaling_recovery": 200.0,
                "post_scale": 530.0,
                "residual_shrinkage_from_crater": 200.0,
            }
            for t in range(8, 15)
        },
    }
    cmp = compare_control_treatment(control, treatment)
    assert cmp["deltas"]["n_replacements"] == 3900
    assert cmp["t9_t12_mean_hole"]["replacement_share"] is not None
    assert cmp["t9_t12_mean_hole"]["replacement_share"] > 0.5
    d = diagnose_phase_2r(cmp)
    assert d["primary_finding"] == "replacement_churn_loss_explains_macro_collapse"
    assert d["keep_pr_29_hold"] is True
    assert d["keep_pr_33_hold"] is True
    assert d["no_scaling_retune"] is True
    assert d["toggle_default_off"] is True


def test_diagnose_routes_residual_pace_coupling():
    control = {
        "n_replacement_events": 100,
        "replacement_loss_distribution": {"mean": 5.0},
        "headline_t9_t12": {"mean_replacement_net": -2.0, "mean_combat_removed": 4.0,
                            "mean_residual": 500.0},
        "post_scale_fidelity": {
            "10": {"mean_post_scale_over_firestone": 0.95},
            "14": {"mean_post_scale_over_firestone": 1.8},
        },
        "symmetric_absolute_fidelity_turns_8_14": {
            "10": {"mean_post_scale_over_firestone": 0.95},
            "12": {"mean_post_scale_over_firestone": 1.4},
            "14": {"mean_post_scale_over_firestone": 1.8},
        },
        "lobby_dynamics": {
            "avg_game_length": 15.6,
            "sim_alive_by_turn": {str(t): 6.0 for t in range(8, 15)},
        },
        "per_turn_decomposition": {
            str(t): {
                "n_replacements": 20,
                "start_of_recruit": 900.0,
                "replacement_net_combat": -5.0,
                "other_recruit_delta": 0.0,
                "residual_scaling_recovery": 600.0,
                "post_scale": 1500.0,
                "residual_shrinkage_from_crater": 10.0,
            }
            for t in range(8, 15)
        },
    }
    treatment = {
        "n_replacement_events": 150,
        "replacement_loss_distribution": {"mean": 8.0},
        "headline_t9_t12": {"mean_replacement_net": -10.0, "mean_combat_removed": 12.0,
                            "mean_residual": 50.0},
        "post_scale_fidelity": {
            "10": {"mean_post_scale_over_firestone": 0.40},
            "14": {"mean_post_scale_over_firestone": 0.20},
        },
        "symmetric_absolute_fidelity_turns_8_14": {
            "10": {"mean_post_scale_over_firestone": 0.40},
            "12": {"mean_post_scale_over_firestone": 0.20},
            "14": {"mean_post_scale_over_firestone": 0.20},
        },
        "lobby_dynamics": {
            "avg_game_length": 13.0,
            "sim_alive_by_turn": {str(t): 4.0 for t in range(8, 15)},
        },
        "per_turn_decomposition": {
            str(t): {
                "n_replacements": 25,
                "start_of_recruit": 880.0,
                "replacement_net_combat": -10.0,
                "other_recruit_delta": 0.0,
                "residual_scaling_recovery": 40.0,
                "post_scale": 910.0,
                "residual_shrinkage_from_crater": 20.0,
            }
            for t in range(8, 15)
        },
    }
    cmp = compare_control_treatment(control, treatment)
    d = diagnose_phase_2r(cmp)
    assert d["primary_finding"] == "residual_pace_coupling_dominates"


def test_greedy_smoke_two_lobbies():
    raw_c = run_greedy_control(2, 13700)
    raw_t = run_greedy_treatment(2, 13700)
    c = summarize_collapse_arm(raw_c)
    t = summarize_collapse_arm(raw_t)
    assert c["recruit_value_stats"] is False
    assert t["recruit_value_stats"] is True
    assert PHASE_2Q_RECRUIT_VALUE_STATS is False
    cmp = compare_control_treatment(c, t)
    assert "paired_post_scale_firestone" in cmp
    assert "paired_alive_curve" in cmp
    assert "10" in (c.get("per_turn_decomposition") or {})
    d = diagnose_phase_2r(cmp)
    assert d["measurement_only"] is True
    assert "11500" in d["confirm_seeds_reserved"]
