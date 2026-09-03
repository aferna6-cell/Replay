"""Tests for Phase 2R replacement churn / combat-loss diagnostic."""

from hsbg_coach.bg_env import PHASE_2Q_RECRUIT_VALUE_STATS
from ml.replacement_churn_diagnostic import (
    CHURN_EXPLAINS_FRACTION,
    FORBIDDEN_RANGES,
    METHODOLOGY_VERSION,
    PHASE_2R_SEED,
    assert_seed_range_allowed,
    compare_control_treatment,
    diagnose_phase_2r,
    run_greedy_control,
    run_greedy_treatment,
    summarize_churn_arm,
)


def test_methodology_is_2r_v1():
    assert METHODOLOGY_VERSION == "2r_v1"


def test_phase_2q_toggle_still_defaults_off():
    assert PHASE_2Q_RECRUIT_VALUE_STATS is False


def test_phase_2r_forbidden_ranges_cover_confirm_and_priors():
    assert (11500, 11699) in FORBIDDEN_RANGES
    assert (13200, 13699) in FORBIDDEN_RANGES  # 2Q consumed
    assert (12700, 13199) in FORBIDDEN_RANGES


def test_phase_2r_seed_guard():
    try:
        assert_seed_range_allowed(11500, 10)
        raise AssertionError("expected overlap failure")
    except ValueError:
        pass
    try:
        assert_seed_range_allowed(13200, 10)
        raise AssertionError("expected 2Q overlap failure")
    except ValueError:
        pass
    assert_seed_range_allowed(PHASE_2R_SEED, 500)


def test_diagnose_routes_churn_explains():
    greedy_cmp = {
        "deltas": {
            "full_board_replace_rate": 0.27,
            "sum_combat_strength_removed": 5.0e6,
            "mean_combat_loss_per_replacement": 40.0,
            "post_scale_over_firestone_t10": -0.48,
            "mean_game_length": -2.5,
            "treatment_post_stats_deficit_t10": 80.0,
            "excess_mean_net_loss_t10": 55.0,
            "cumulative_excess_net_loss_t8_t10": 90.0,
            "churn_explains_fraction_t10": 0.70,
        }
    }
    d = diagnose_phase_2r(greedy_cmp)
    assert d["primary_finding"] == "replacement_churn_loss_explains_macro_collapse"
    assert d["keep_pr_29_hold"] is True
    assert d["keep_pr_33_hold"] is True
    assert d["feature_toggle_default_off"] is True
    assert d["churn_explains_threshold"] == CHURN_EXPLAINS_FRACTION


def test_diagnose_routes_residual_coupling():
    greedy_cmp = {
        "deltas": {
            "full_board_replace_rate": 0.27,
            "sum_combat_strength_removed": 1.0e5,
            "mean_combat_loss_per_replacement": 10.0,
            "post_scale_over_firestone_t10": -0.48,
            "mean_game_length": -2.5,
            "treatment_post_stats_deficit_t10": 80.0,
            "excess_mean_net_loss_t10": 10.0,
            "cumulative_excess_net_loss_t8_t10": 15.0,
            "churn_explains_fraction_t10": 0.12,
        }
    }
    d = diagnose_phase_2r(greedy_cmp)
    assert d["primary_finding"] == "residual_or_pace_coupling_dominates"


def test_greedy_smoke_two_lobbies_instruments_replacements():
    raw_c = run_greedy_control(2, PHASE_2R_SEED)
    raw_t = run_greedy_treatment(2, PHASE_2R_SEED)
    c = summarize_churn_arm(raw_c)
    t = summarize_churn_arm(raw_t)
    assert c["recruit_value_stats"] is False
    assert t["recruit_value_stats"] is True
    assert "per_turn_decomposition" in c
    assert "8" in c["per_turn_decomposition"]
    assert "14" in c["per_turn_decomposition"]
    assert "replacement_loss_distribution" in c
    assert "post_scale_firestone_ratios" in c
    assert "alive_curve_t8_t14" in c
    cmp = compare_control_treatment(c, t)
    assert "paired_post_scale_firestone_ratios" in cmp
    assert "paired_alive_curve" in cmp
    assert "per_turn_decomposition_delta" in cmp
    assert "churn_explains_fraction_t10" in (cmp.get("deltas") or {})
    # Treatment should complete at least as many replacements on tiny sample
    # in the typical direction (not a hard gate — just sanity on keys).
    assert c["n_completed_replacements"] is not None
    assert t["n_completed_replacements"] is not None
    d = diagnose_phase_2r(cmp)
    assert d["primary_finding"] in {
        "replacement_churn_loss_explains_macro_collapse",
        "residual_or_pace_coupling_dominates",
        "churn_up_without_macro_collapse",
        "inconclusive",
    }
    # Toggle must remain OFF after arm contexts exit.
    assert PHASE_2Q_RECRUIT_VALUE_STATS is False
