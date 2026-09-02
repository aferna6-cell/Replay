"""Tests for Phase 2N catalogue classification + measurement guards."""

from hsbg_coach.bg_env import PHASE_2N_DEATH_RETURN, PHASE_2N_FREEZE_TOPUP, POOL_COPIES
from ml.fidelity_phase_2n import (
    METHODOLOGY_VERSION,
    PHASE_2N_SEED,
    assert_seed_range_allowed,
    evaluate_phase_2n_decision,
)
from ml.shop_pool_audit import audit_catalogue_synchronization, audit_rule_mismatches


def test_methodology_and_flags():
    assert METHODOLOGY_VERSION == "2n_v1"
    assert PHASE_2N_SEED == 11000
    assert PHASE_2N_DEATH_RETURN is True
    assert PHASE_2N_FREEZE_TOPUP is True
    assert POOL_COPIES[6] == 7


def test_rejects_dev_and_confirm_seeds():
    for seed in (8000, 9000, 10200, 11500):
        try:
            assert_seed_range_allowed(seed, 10)
            assert False, f"should reject {seed}"
        except RuntimeError:
            pass
    assert_seed_range_allowed(11000, 500)


def test_catalogue_clean_after_2n_a():
    cat = audit_catalogue_synchronization()
    assert cat.get("n_missing_from_kb", 0) == 0
    assert cat["status_counts"].get("MISSING_OR_INVALID_TIER", 0) == 0


def test_no_actionable_rule_mismatches_after_2n():
    rules = audit_rule_mismatches()
    assert rules["phase_2n_actionable_ids"] == []
    assert rules["n_phase_2n_actionable"] == 0


def test_decision_accept_when_clean():
    analysis = {
        "catalogue_synchronization": {
            "n_missing_from_kb": 0,
            "status_counts": {"IN_EXACT_CATALOGUE": 226},
        },
        "rule_mismatches": {
            "phase_2n_actionable_ids": [],
        },
        "live_calibration": {
            "primary_deal_level": {
                "raw_ratio_obs_over_exp": 0.95,
                "lobby_clustered": {
                    "raw_obs_minus_exp": {"ci95": [-1.0, 1.0]},
                },
            },
        },
        "headlines": {},
    }
    d = evaluate_phase_2n_decision(analysis)
    assert d["decision_branch"] == "accept_simulator_v1_x_candidate"
