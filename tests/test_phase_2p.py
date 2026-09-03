"""Tests for Phase 2P replacement-value contamination diagnostic."""

from ml.replacement_value_diagnostic import (
    FORBIDDEN_RANGES,
    METHODOLOGY_VERSION,
    assert_seed_range_allowed,
    diagnose_contamination,
    run_greedy_arm,
    summarize_arm,
    _base_card_view,
)


def test_methodology_is_2p_v2():
    assert METHODOLOGY_VERSION == "2p_v2"


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


def test_golden_natural_printed_baseline_doubles_kb():
    """Golden 4/3 KB card → natural printed raw 14, not 7 (2p_v2)."""
    kb_id = {
        "X": type("CK", (), {
            "card_id": "X", "name": "Tiny", "attack": 4, "health": 3,
            "keywords": [], "text": "", "tribes": [],
        })(),
    }
    kb_name = {"Tiny": kb_id["X"]}
    golden_obs = {
        "name": "Tiny",
        "card_id": "X",
        "attack": 40,   # scaled combat stats
        "health": 30,
        "tags": {"PREMIUM": "1", "TECH_LEVEL": "1"},
    }
    base = _base_card_view(golden_obs, kb_id, kb_name)
    assert base["is_golden"] is True
    assert base["normal_raw"] == 7.0
    assert base["raw"] == 14.0
    assert base["attack"] == 8
    assert base["health"] == 6

    nongolden = {
        "name": "Tiny",
        "card_id": "X",
        "attack": 20,
        "health": 15,
        "tags": {"TECH_LEVEL": "1"},
    }
    base2 = _base_card_view(nongolden, kb_id, kb_name)
    assert base2["is_golden"] is False
    assert base2["normal_raw"] == 7.0
    assert base2["raw"] == 7.0


def test_diagnose_contamination_prefers_scaling_block():
    greedy = {
        "state_summary_by_turn": {
            "10": {
                "pct_scaling_blocked_upgrade_states": 0.6,
                "p_best_shop_gt_weakest_scaled": 0.05,
                "p_best_shop_gt_weakest_printed": 0.7,
                "all_full_board_states": {
                    "pct_scaling_blocked_upgrade_states": 0.6,
                    "p_best_shop_gt_weakest_scaled": 0.05,
                    "p_best_shop_gt_weakest_printed": 0.7,
                },
                "nongolden_weakest_states": {
                    "pct_scaling_blocked_upgrade_states": 0.55,
                    "p_best_shop_gt_weakest_scaled": 0.04,
                    "p_best_shop_gt_weakest_printed": 0.7,
                },
            }
        },
        "contamination_headline": {},
    }
    phase_2j = {
        "state_summary_by_turn": {
            "10": {
                "pct_scaling_blocked_upgrade_states": 0.4,
                "p_best_shop_gt_weakest_scaled": 0.04,
                "p_best_shop_gt_weakest_printed": 0.5,
                "all_full_board_states": {
                    "pct_scaling_blocked_upgrade_states": 0.4,
                    "p_best_shop_gt_weakest_scaled": 0.04,
                    "p_best_shop_gt_weakest_printed": 0.5,
                },
                "nongolden_weakest_states": {
                    "pct_scaling_blocked_upgrade_states": 0.38,
                    "p_best_shop_gt_weakest_scaled": 0.05,
                    "p_best_shop_gt_weakest_printed": 0.5,
                },
            }
        },
        "contamination_headline": {},
    }
    dec = diagnose_contamination(greedy, phase_2j)
    assert dec["primary_finding"] == "scaling_contamination_dominant"
    assert dec["survives_nongolden_weakest_filter"] is True


def test_small_phase_2p_smoke():
    raw = run_greedy_arm(2, 12700)
    assert raw["state_rows"]
    assert raw["candidate_rows"]
    summary = summarize_arm(raw)
    assert summary["n_full_board_states"] > 0
    assert "10" in summary["state_summary_by_turn"]
    t10 = summary["state_summary_by_turn"]["10"]
    assert "all_full_board_states" in t10
    assert "nongolden_weakest_states" in t10
    assert "contamination_headline" in summary
