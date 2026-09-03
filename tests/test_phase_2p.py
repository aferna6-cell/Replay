"""Tests for Phase 2P replacement-value contamination diagnostic."""

from ml.replacement_value_diagnostic import (
    FORBIDDEN_RANGES,
    assert_seed_range_allowed,
    diagnose_contamination,
    run_greedy_arm,
    summarize_arm,
)


def test_phase_2p_forbidden_ranges_cover_confirm_and_2o():
    assert (11500, 11699) in FORBIDDEN_RANGES
    assert (12200, 12699) in FORBIDDEN_RANGES


def test_phase_2p_seed_guard():
    try:
        assert_seed_range_allowed(11500, 10)
        raise AssertionError("expected overlap failure")
    except ValueError:
        pass
    assert_seed_range_allowed(12700, 500)


def test_diagnose_contamination_prefers_scaling_block():
    greedy = {
        "state_summary_by_turn": {
            "10": {
                "pct_scaling_blocked_upgrade_states": 0.6,
                "p_best_shop_gt_weakest_scaled": 0.05,
                "p_best_shop_gt_weakest_printed": 0.7,
            }
        }
    }
    phase_2j = {
        "state_summary_by_turn": {
            "10": {
                "pct_scaling_blocked_upgrade_states": 0.4,
                "p_best_shop_gt_weakest_scaled": 0.04,
                "p_best_shop_gt_weakest_printed": 0.5,
            }
        }
    }
    dec = diagnose_contamination(greedy, phase_2j)
    assert dec["primary_finding"] == "scaling_contamination_dominant"


def test_small_phase_2p_smoke():
    raw = run_greedy_arm(2, 12700)
    assert raw["state_rows"]
    assert raw["candidate_rows"]
    summary = summarize_arm(raw)
    assert summary["n_full_board_states"] > 0
    assert "10" in summary["state_summary_by_turn"]
