"""Tests for Phase 2M shop/pool rules audit (2m_v2)."""

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
    filter_post_assembly_deals,
    is_post_assembly_deal,
    p_hit_live_deal,
    p_zero_live_deal,
)


def test_methodology_and_frozen():
    assert METHODOLOGY_VERSION == "2m_v2"
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


def test_p_zero_and_hit_live_exact():
    p = p_zero_live_deal(2, 10, 3)
    assert abs(p - (8 / 10) * (7 / 9) * (6 / 8)) < 1e-12
    assert abs(p_hit_live_deal(2, 10, 3) - (1 - p)) < 1e-12
    assert p_zero_live_deal(0, 10, 3) == 1.0
    assert p_zero_live_deal(10, 10, 3) == 0.0
    e = expected_raw_live_deal(2, 10, 3)
    assert abs(e - 3 * 2 / 10) < 1e-12


def test_entry_turn_excluded_from_post_assembly():
    """Synthetic integrity: entry-turn deal must NOT count as post-assembly.

    turn 10: core appears → buy/play → end recruit → first 2-core state
    Phase 2M: turn-10 deal MUST NOT count; turn-11+ deals DO count.
    """
    entry_turn = 10
    deals = [
        {"turn": 9, "reason": "start_turn", "dealt_names": ["CoreA"]},
        {"turn": 10, "reason": "start_turn", "dealt_names": ["CoreA"]},  # entry
        {"turn": 10, "reason": "roll", "dealt_names": ["Other"]},
        {"turn": 11, "reason": "start_turn", "dealt_names": ["CoreB"]},
        {"turn": 12, "reason": "roll", "dealt_names": ["CoreA"]},
    ]
    assert not is_post_assembly_deal(10, entry_turn)
    assert is_post_assembly_deal(11, entry_turn)
    kept = filter_post_assembly_deals(deals, entry_turn)
    assert [d["turn"] for d in kept] == [11, 12]
    assert all(d["turn"] > entry_turn for d in kept)
    # Entry-turn shop that offered the assembling core is excluded
    assert not any(d["turn"] == 10 for d in kept)


def test_catalogue_sync_post_2n_a_clean():
    cat = audit_catalogue_synchronization()
    assert cat["n_core_slots"] > 0
    assert cat.get("n_missing_from_kb", 0) == 0
    assert cat["status_counts"].get("MISSING_FROM_KB", 0) == 0
    assert cat["status_counts"].get("MISSING_OR_INVALID_TIER", 0) == 0
    assert cat["status_counts"].get("IN_EXACT_CATALOGUE", 0) == cat["n_core_slots"]


def test_rule_mismatches_actionable_vs_contextual():
    rules = audit_rule_mismatches()
    # After Phase 2N-A/B/C, actionable pool mismatches should be cleared.
    assert "pool_copies_tier_6" not in rules["phase_2n_actionable_ids"]
    assert "elimination_no_return_to_pool" not in rules["phase_2n_actionable_ids"]
    assert "freeze_no_topup" not in rules["phase_2n_actionable_ids"]
    assert POOL_COPIES[6] == 7
    assert REF_POOL_COPIES[6] == 7
    assert SHOP_SLOTS == audit_pool_contract()["simulator"]["SHOP_SLOTS"]
    assert "shop_slots_vs_spell_era" in rules["contextual_ids"]
    assert "no_tier_7" in rules["contextual_ids"]


def test_pool_deal_hook_is_observational():
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
    assert len(deals) == 8


def test_decision_multiple_mismatches():
    analysis = {
        "headlines": {
            "live_sum_expected_raw": 40.0,
            "live_sum_observed_raw": 38.0,
            "live_raw_ratio_obs_over_exp": 0.95,
            "live_sum_expected_hit_probability": 30.0,
            "live_sum_observed_hit_deals": 29.0,
            "live_hit_ratio_obs_over_exp": 0.97,
            "phase_2l_a1_share": 0.37,
            "phase_2l_a3_share": 0.63,
        },
        "rule_mismatches": {
            "n_demonstrated_mismatches": 5,
            "n_phase_2n_actionable": 3,
            "demonstrated_ids": [
                "pool_copies_tier_6",
                "elimination_no_return_to_pool",
                "freeze_no_topup",
                "shop_slots_vs_spell_era",
                "no_tier_7",
            ],
            "phase_2n_actionable_ids": [
                "pool_copies_tier_6",
                "elimination_no_return_to_pool",
                "freeze_no_topup",
            ],
            "contextual_ids": ["shop_slots_vs_spell_era", "no_tier_7"],
        },
        "live_calibration": {
            "post_assembly_deal_boundary": "turn > entry_turn",
            "primary_deal_level": {
                "n_deal_card_observations": 200,
                "sum_expected_raw": 40.0,
                "sum_observed_raw": 38.0,
                "lobby_clustered": {
                    "raw_obs_minus_exp": {
                        "mean": -0.1, "ci95": [-1.0, 0.8]},
                },
            },
        },
        "catalogue_synchronization": {
            "status_share": {"MISSING_FROM_KB": 0.25},
        },
    }
    d = evaluate_phase_2m_decision(analysis)
    assert d["decision_branch"] == "multiple_substantial_mismatches"
    assert "scoped interventions" in d["recommended_next_step"]
    assert "no_tier_7" in d["contextual_ids"]


def test_decision_live_undershoot_flags_draw():
    analysis = {
        "headlines": {
            "live_sum_expected_raw": 50.0,
            "live_sum_observed_raw": 10.0,
            "live_raw_ratio_obs_over_exp": 0.2,
            "live_sum_expected_hit_probability": 40.0,
            "live_sum_observed_hit_deals": 8.0,
            "phase_2l_a1_share": 0.1,
            "phase_2l_a3_share": 0.5,
        },
        "rule_mismatches": {
            "n_demonstrated_mismatches": 1,
            "n_phase_2n_actionable": 1,
            "demonstrated_ids": ["pool_copies_tier_6"],
            "phase_2n_actionable_ids": ["pool_copies_tier_6"],
            "contextual_ids": [],
        },
        "live_calibration": {
            "primary_deal_level": {
                "n_deal_card_observations": 100,
                "lobby_clustered": {
                    "raw_obs_minus_exp": {
                        "mean": -5.0, "ci95": [-8.0, -2.0]},
                },
            },
        },
        "catalogue_synchronization": {
            "status_share": {"MISSING_FROM_KB": 0.05},
        },
    }
    d = evaluate_phase_2m_decision(analysis)
    assert d["live_surprising"] is True
    assert d["decision_branch"] in (
        "shop_draw_probabilities_rules_mismatch",
        "multiple_substantial_mismatches",
    )
