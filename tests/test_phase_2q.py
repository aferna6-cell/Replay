"""Tests for Phase 2Q recruit/combat representation split."""

from hsbg_coach.bg_env import (
    PHASE_2Q_RECRUIT_VALUE_STATS,
    EnvMinion,
    combat_raw,
    recruit_raw,
    recruit_value_stats_enabled,
    valuation_raw,
)
from ml.recruit_combat_split_diagnostic import (
    FORBIDDEN_RANGES,
    METHODOLOGY_VERSION,
    assert_seed_range_allowed,
    compare_control_treatment,
    run_greedy_control,
    run_greedy_treatment,
    summarize_split_arm,
)


def test_methodology_is_2q_v1():
    assert METHODOLOGY_VERSION == "2q_v1"


def test_phase_2q_forbidden_ranges_cover_confirm_and_priors():
    assert (11500, 11699) in FORBIDDEN_RANGES
    assert (12700, 13199) in FORBIDDEN_RANGES
    assert (12200, 12699) in FORBIDDEN_RANGES


def test_phase_2q_seed_guard():
    try:
        assert_seed_range_allowed(11500, 10)
        raise AssertionError("expected overlap failure")
    except ValueError:
        pass
    assert_seed_range_allowed(13200, 500)


def test_scaling_mutates_combat_only_not_recruit():
    m = EnvMinion("x", "X", 3, 4, 3, [], [])
    assert m.recruit_attack == 4 and m.recruit_health == 3
    m.attack = 40
    m.health = 30
    assert m.recruit_attack == 4 and m.recruit_health == 3
    view = m.view()
    assert view["attack"] == 40 and view["health"] == 30
    assert view["recruit_attack"] == 4 and view["recruit_health"] == 3
    assert combat_raw(view) == 70
    assert recruit_raw(view) == 7


def test_golden_doubles_both_combat_and_recruit():
    m = EnvMinion("x", "Tiny", 1, 4, 3, [], [])
    g = m.as_golden()
    assert g.golden is True
    assert g.attack == 8 and g.health == 6
    assert g.recruit_attack == 8 and g.recruit_health == 6
    # After synthetic combat inflation, golden still doubles recruit from recruit base.
    m.attack = 20
    m.health = 15
    g2 = m.as_golden()
    assert g2.attack == 40 and g2.health == 30
    assert g2.recruit_attack == 8 and g2.recruit_health == 6


def test_valuation_raw_respects_toggle():
    m = {"attack": 40, "health": 30, "recruit_attack": 4, "recruit_health": 3}
    assert PHASE_2Q_RECRUIT_VALUE_STATS is False
    assert valuation_raw(m) == 70.0
    with recruit_value_stats_enabled(True):
        assert valuation_raw(m) == 7.0
    assert valuation_raw(m) == 70.0


def test_residual_scaling_preserves_recruit_stats():
    from hsbg_coach.bg_env import BGEnv

    env = BGEnv(seed=0, scaling_mode="residual")
    env.reset()
    p = env.players[0]
    p.board = [EnvMinion("a", "A", 4, 10, 10, [], [])]
    before_r = (p.board[0].recruit_attack, p.board[0].recruit_health)
    # Force a mid/late turn so residual can apply.
    env.turn = 10
    env._end_of_turn_scaling_residual(p)
    assert (p.board[0].recruit_attack, p.board[0].recruit_health) == before_r
    # Combat may or may not grow depending on budget; recruit must not change.
    assert p.board[0].attack >= 1 and p.board[0].health >= 1


def test_compare_flags_post_scale_macro_collapse():
    control = {
        "recruit_delta_t9_t12": {"mean_t9_t12": 0.0},
        "full_board_replace_rate": 0.01,
        "valuation_scaling_blocked_pct_full_board": 0.8,
        "post_scale_fidelity": {
            "10": {"mean_post_scale_over_firestone": 0.95},
            "14": {"mean_post_scale_over_firestone": 1.8},
        },
        "symmetric_absolute_fidelity_turns_8_14": {
            "10": {"mean_post_scale_over_firestone": 0.95},
            "12": {"mean_post_scale_over_firestone": 1.4},
            "14": {"mean_post_scale_over_firestone": 1.8},
        },
        "lobby_dynamics": {"avg_game_length": 15.5},
    }
    treatment = {
        "recruit_delta_t9_t12": {"mean_t9_t12": -200.0},
        "full_board_replace_rate": 0.28,
        "valuation_scaling_blocked_pct_full_board": 0.0,
        "post_scale_fidelity": {
            "10": {"mean_post_scale_over_firestone": 0.47},
            "14": {"mean_post_scale_over_firestone": 0.11},
        },
        "symmetric_absolute_fidelity_turns_8_14": {
            "10": {"mean_post_scale_over_firestone": 0.47},
            "12": {"mean_post_scale_over_firestone": 0.16},
            "14": {"mean_post_scale_over_firestone": 0.11},
        },
        "lobby_dynamics": {"avg_game_length": 13.0},
    }
    cmp = compare_control_treatment(control, treatment)
    assert cmp["gates"]["full_board_replace_rate_increases"] is True
    assert cmp["gates"]["scaling_blocked_collapses"] is True
    assert cmp["gates"]["post_scale_macro_not_materially_worse"] is False
    assert cmp["gates"]["recruit_delta_t9_t12_increases"] is False
    from ml.recruit_combat_split_diagnostic import diagnose_phase_2q
    d = diagnose_phase_2q(cmp)
    assert d["primary_finding"] == (
        "replacement_unblocked_but_post_scale_macro_collapses"
    )


def test_greedy_treatment_smoke_two_lobbies():
    raw_c = run_greedy_control(2, 13200)
    raw_t = run_greedy_treatment(2, 13200)
    c = summarize_split_arm(raw_c)
    t = summarize_split_arm(raw_t)
    assert c["recruit_value_stats"] is False
    assert t["recruit_value_stats"] is True
    cmp = compare_control_treatment(c, t)
    assert "gates" in cmp
    assert cmp["gates_total"] == 5
