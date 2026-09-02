"""Tests for Phase 2L availability decomposition (2l_v2)."""

from ml.availability_decomposition import (
    FROZEN_ALPHA,
    FROZEN_PRIOR_HASH,
    METHODOLOGY_VERSION,
    PHASE_2L_SEED,
    SUBFATE_CODES,
    catalogue_exclusion_reason,
    card_tier,
    exact_catalogue_names,
    expected_raw_one_deal,
    p_zero_one_deal,
    slot_draw_probability,
    tribe_eligible,
)
from hsbg_coach.bg_env import build_pool
from ml.fidelity_phase_2k import load_frozen_prior
from ml.fidelity_phase_2l import assert_seed_range_allowed
from ml.phase_2l_decision import evaluate_phase_2l_decision


def test_methodology_and_frozen():
    assert METHODOLOGY_VERSION == "2l_v2"
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
    t = card_tier("Alleycat")
    assert t == 1


def test_exact_catalogue_matches_build_pool():
    tribes = ["Beast", "Murloc", "Mech", "Demon", "Pirate"]
    names = exact_catalogue_names(tuple(tribes))
    pool_names = {m.name for m in build_pool(lobby_tribes=tribes)}
    assert names == pool_names
    assert "Alleycat" in names


def test_tribe_eligible_filter():
    assert tribe_eligible("Alleycat", ["Beast", "Murloc", "Mech",
                                       "Demon", "Pirate"]) is True


def test_catalogue_exclusion_reasons():
    tribes = ["Beast", "Murloc", "Mech", "Demon", "Pirate"]
    cat = set(exact_catalogue_names(tuple(tribes)))
    assert catalogue_exclusion_reason("Alleycat", tribes, cat) is None
    assert catalogue_exclusion_reason(
        "__not_a_real_card__", tribes, cat) == "MISSING_KB_OR_TIER_OR_STATS"


def test_slot_draw_and_p_zero():
    tribes = ["Beast", "Murloc", "Mech", "Demon", "Pirate"]
    catalogue = list(build_pool(lobby_tribes=tribes))
    p = slot_draw_probability("Alleycat", tavern_tier=1, catalogue=catalogue)
    assert 0.0 < p <= 1.0
    assert expected_raw_one_deal(p, 3) == 3 * p
    pz = p_zero_one_deal(p, 3)
    assert 0.0 <= pz <= 1.0


def test_decision_zero_raw_dominates():
    analysis = {
        "n_states": 50,
        "subfate_share_of_never_legal": {
            "A1_NOT_IN_EXACT_CATALOGUE": 0.05,
            "A2_NEVER_TIER_ELIGIBLE": 0.05,
            "A3_TIER_ELIGIBLE_ZERO_RAW": 0.7,
            "A4_RAW_BUT_ZERO_LEGAL": 0.15,
            "A5_OTHER": 0.05,
        },
        "headlines": {
            "pct_exact_catalogue_tier_eligible_zero_raw": 0.7,
            "pct_raw_but_never_legal": 0.15,
        },
        "sampler_calibration_unconditioned": {
            "observed_zero_offer_rate": 0.8,
            "expected_zero_offer_rate": 0.6,
        },
    }
    d = evaluate_phase_2l_decision(analysis)
    assert d["decision_branch"] == "a3_tier_eligible_zero_raw"
    assert "shop/pool" in d["recommended_next_step"]


def test_decision_catalogue_dominates():
    analysis = {
        "n_states": 50,
        "subfate_share_of_never_legal": {
            "A1_NOT_IN_EXACT_CATALOGUE": 0.7,
            "A2_NEVER_TIER_ELIGIBLE": 0.05,
            "A3_TIER_ELIGIBLE_ZERO_RAW": 0.2,
            "A4_RAW_BUT_ZERO_LEGAL": 0.05,
            "A5_OTHER": 0.0,
        },
        "headlines": {},
        "sampler_calibration_unconditioned": {},
    }
    d = evaluate_phase_2l_decision(analysis)
    assert d["decision_branch"] == "a1_not_in_exact_catalogue"
    assert "catalogue" in d["recommended_next_step"].lower()


def test_decision_raw_illegal_dominates_no_pool_touch():
    analysis = {
        "n_states": 50,
        "subfate_share_of_never_legal": {
            "A1_NOT_IN_EXACT_CATALOGUE": 0.0,
            "A2_NEVER_TIER_ELIGIBLE": 0.0,
            "A3_TIER_ELIGIBLE_ZERO_RAW": 0.2,
            "A4_RAW_BUT_ZERO_LEGAL": 0.75,
            "A5_OTHER": 0.05,
        },
        "headlines": {},
        "sampler_calibration_unconditioned": {},
    }
    d = evaluate_phase_2l_decision(analysis)
    assert d["decision_branch"] == "a4_raw_but_zero_legal"
    assert "do NOT touch the pool" in d["recommended_next_step"]


def test_subfate_codes_complete():
    assert "A1_NOT_IN_EXACT_CATALOGUE" in SUBFATE_CODES
    assert "A3_TIER_ELIGIBLE_ZERO_RAW" in SUBFATE_CODES
    assert "A4_RAW_BUT_ZERO_LEGAL" in SUBFATE_CODES
