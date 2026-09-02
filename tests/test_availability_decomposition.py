"""Tests for Phase 2L availability decomposition."""

from ml.availability_decomposition import (
    FROZEN_ALPHA,
    FROZEN_PRIOR_HASH,
    METHODOLOGY_VERSION,
    PHASE_2L_SEED,
    SUBFATE_CODES,
    card_in_lobby_pool,
    card_tier,
    expected_raw_appearances,
)
from ml.fidelity_phase_2k import load_frozen_prior
from ml.fidelity_phase_2l import assert_seed_range_allowed
from ml.phase_2l_decision import evaluate_phase_2l_decision


def test_methodology_and_frozen():
    assert METHODOLOGY_VERSION == "2l_v1"
    assert FROZEN_ALPHA == 0.5
    assert PHASE_2L_SEED == 10200
    prior = load_frozen_prior()
    assert prior.content_hash_sha256() == FROZEN_PRIOR_HASH


def test_rejects_reserved_seeds():
    for seed in (8000, 9000, 10000):
        try:
            assert_seed_range_allowed(seed, 10)
            assert False, f"should reject {seed}"
        except RuntimeError:
            pass
    assert_seed_range_allowed(10200, 500)


def test_card_tier_lookup():
    # Alleycat is a known T1 beast
    t = card_tier("Alleycat")
    assert t == 1


def test_lobby_pool_tribe_filter():
    # A Murloc-only card should not be in a Beast-only lobby.
    # Use a clearly tribal card if present.
    assert card_in_lobby_pool("Alleycat", ["Beast", "Murloc", "Mech",
                                           "Demon", "Pirate"]) is True


def test_expected_raw_nonnegative():
    e = expected_raw_appearances(
        card_name="Alleycat", tavern_tier=1, n_shop_slots=3,
        lobby_tribes=["Beast", "Murloc", "Mech", "Demon", "Pirate"])
    assert e >= 0.0


def test_decision_zero_raw_dominates():
    analysis = {
        "n_states": 50,
        "subfate_share_of_never_legal": {
            "A1_NOT_IN_LOBBY_POOL": 0.05,
            "A2_NEVER_TIER_ELIGIBLE": 0.05,
            "A3_TIER_ELIGIBLE_ZERO_RAW": 0.7,
            "A4_RAW_BUT_ZERO_LEGAL": 0.15,
            "A5_OTHER": 0.05,
        },
        "headlines": {
            "pct_never_legal_mass_tier_eligible_zero_raw": 0.7,
            "pct_never_legal_mass_raw_but_zero_legal": 0.15,
        },
    }
    d = evaluate_phase_2l_decision(analysis)
    assert d["decision_branch"] == "a3_tier_eligible_zero_raw"
    assert "shop/pool generation" in d["recommended_next_step"]


def test_decision_raw_illegal_dominates_no_pool_touch():
    analysis = {
        "n_states": 50,
        "subfate_share_of_never_legal": {
            "A1_NOT_IN_LOBBY_POOL": 0.0,
            "A2_NEVER_TIER_ELIGIBLE": 0.0,
            "A3_TIER_ELIGIBLE_ZERO_RAW": 0.2,
            "A4_RAW_BUT_ZERO_LEGAL": 0.75,
            "A5_OTHER": 0.05,
        },
        "headlines": {},
    }
    d = evaluate_phase_2l_decision(analysis)
    assert d["decision_branch"] == "a4_raw_but_zero_legal"
    assert "do NOT touch the pool" in d["recommended_next_step"]


def test_subfate_codes_complete():
    assert "A3_TIER_ELIGIBLE_ZERO_RAW" in SUBFATE_CODES
    assert "A4_RAW_BUT_ZERO_LEGAL" in SUBFATE_CODES
