"""Tests for Phase 2M shop/pool rules audit (measurement-only)."""

from hsbg_coach.bg_env import BGEnv, POOL_COPIES, SHOP_SLOTS
from ml.availability_decomposition import FROZEN_ALPHA, FROZEN_PRIOR_HASH
from ml.fidelity_phase_2k import load_frozen_prior
from ml.fidelity_phase_2m import assert_seed_range_allowed
from ml.phase_2m_decision import evaluate_phase_2m_decision
from ml.shop_pool_audit import (
    METHODOLOGY_VERSION,
    PHASE_2M_SEED,
    REF_POOL_COPIES,
    audit_catalogue_synchronization,
    audit_pool_contract,
    audit_rule_mismatches,
    expected_raw_live_deal,
    p_zero_live_deal,
)


def test_methodology_and_frozen():
    assert METHODOLOGY_VERSION == "2m_v1"
    assert PHASE_2M_SEED == 10200
    assert FROZEN_ALPHA == 0.5
    prior = load_frozen_prior()
    assert prior.content_hash_sha256() == FROZEN_PRIOR_HASH


def test_rejects_reserved_and_forbidden_seeds():
    for seed in (8000, 9000, 10000, 11000, 11500):
        try:
            assert_seed_range_allowed(seed, 10)
            assert False, f"should reject {seed}"
        except RuntimeError:
            pass
    assert_seed_range_allowed(10200, 500)


def test_p_zero_live_exact():
    # 2 copies of card in total 10; 3 slots → P(miss) = (8/10)*(7/9)*(6/8)
    p = p_zero_live_deal(2, 10, 3)
    assert abs(p - (8 / 10) * (7 / 9) * (6 / 8)) < 1e-12
    assert p_zero_live_deal(0, 10, 3) == 1.0
    assert p_zero_live_deal(10, 10, 3) == 0.0
    e = expected_raw_live_deal(2, 10, 3)
    assert abs(e - 3 * 2 / 10) < 1e-12


def test_catalogue_sync_flags_missing_kb():
    cat = audit_catalogue_synchronization()
    assert cat["n_core_slots"] > 0
    assert cat["n_missing_from_kb"] >= 1
    assert "MISSING_FROM_KB" in cat["status_counts"]
    assert "Captain Cookie" in cat["missing_from_kb_names"] or cat[
        "n_missing_from_kb"] > 0


def test_rule_mismatches_include_t6_copies():
    rules = audit_rule_mismatches()
    assert rules["n_demonstrated_mismatches"] >= 1
    assert "pool_copies_tier_6" in rules["demonstrated_ids"]
    assert POOL_COPIES[6] == 6
    assert REF_POOL_COPIES[6] == 7
    assert SHOP_SLOTS == audit_pool_contract()["simulator"]["SHOP_SLOTS"]


def test_pool_deal_hook_is_observational():
    """Hook must not change dealt shops / pool totals."""
    deals = []

    def hook(env, player, meta):
        deals.append(meta)

    env_a = BGEnv(seed=42)
    env_a.reset(seed=42)
    shop_a = [m.name for m in env_a.players[0].shop]
    pool_a = dict(env_a._pool)

    env_b = BGEnv(seed=42)
    env_b.pool_deal_hook = hook
    env_b._pool_audit_track_names = frozenset(["Alleycat"])
    env_b.reset(seed=42)
    shop_b = [m.name for m in env_b.players[0].shop]
    assert shop_a == shop_b
    assert pool_a == env_b._pool
    assert len(deals) == 8  # one reset deal per seat


def test_decision_multiple_mismatches():
    analysis = {
        "headlines": {
            "live_observed_zero_offer_rate": 1.0,
            "live_expected_zero_offer_rate": 0.7,
            "live_sum_expected_raw": 40.0,
            "live_sum_observed_raw": 0.0,
            "phase_2l_a1_share": 0.37,
            "phase_2l_a3_share": 0.63,
        },
        "rule_mismatches": {
            "n_demonstrated_mismatches": 4,
            "demonstrated_ids": [
                "pool_copies_tier_6",
                "elimination_no_return_to_pool",
                "freeze_no_topup",
            ],
        },
        "live_calibration": {"n_card_windows": 200},
        "catalogue_synchronization": {
            "status_share": {"MISSING_FROM_KB": 0.25},
        },
    }
    d = evaluate_phase_2m_decision(analysis)
    assert d["decision_branch"] == "multiple_substantial_mismatches"
    assert "scoped interventions" in d["recommended_next_step"]


def test_decision_live_consistent():
    analysis = {
        "headlines": {
            "live_observed_zero_offer_rate": 0.84,
            "live_expected_zero_offer_rate": 0.83,
            "live_sum_expected_raw": 10.0,
            "live_sum_observed_raw": 9.0,
            "phase_2l_a1_share": 0.1,
            "phase_2l_a3_share": 0.5,
        },
        "rule_mismatches": {
            "n_demonstrated_mismatches": 1,
            "demonstrated_ids": ["pool_copies_tier_6"],
        },
        "live_calibration": {"n_card_windows": 100},
        "catalogue_synchronization": {
            "status_share": {"MISSING_FROM_KB": 0.05},
        },
    }
    d = evaluate_phase_2m_decision(analysis)
    assert d["decision_branch"] == "scarcity_consistent_with_live_expectation"
