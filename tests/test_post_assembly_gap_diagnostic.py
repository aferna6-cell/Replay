"""Tests for Phase 2K post-assembly residual composition-gap diagnostic."""

from hsbg_coach.board_opportunity_policy import policies_for_lobby
from hsbg_coach.build_path import Archetype
from hsbg_coach.persistence_prior import PersistencePrior
from ml.fidelity_phase_2k import (
    assert_seed_range_allowed,
    finals_fingerprint,
    load_frozen_prior,
)
from ml.post_assembly_gap_diagnostic import (
    FORBIDDEN_CONFIRM_SEED,
    FROZEN_ALPHA,
    FROZEN_PRIOR_HASH,
    METHODOLOGY_VERSION,
    PHASE_2K_SEED,
    CoreCardTrace,
    PostAssemblyState,
    _classify_missing_card,
    analyze_post_assembly_gap,
    core_weights,
    weighted_coverage,
)
from ml.phase_2k_decision import evaluate_phase_2k_decision


def test_methodology_and_frozen_policy():
    assert METHODOLOGY_VERSION == "2k_v1"
    assert FROZEN_ALPHA == 0.5
    assert PHASE_2K_SEED == 9000
    prior = load_frozen_prior()
    assert prior.content_hash_sha256() == FROZEN_PRIOR_HASH


def test_rejects_phase_2j_confirmation_seeds():
    try:
        assert_seed_range_allowed(FORBIDDEN_CONFIRM_SEED, 10)
        assert False, "should reject"
    except RuntimeError as e:
        assert "8000" in str(e)
    # Non-overlapping OK
    assert_seed_range_allowed(9000, 500)


def test_tracer_matches_plain_fingerprints():
    prior = load_frozen_prior()
    seed, lobbies = PHASE_2K_SEED + 50, 2
    plain = finals_fingerprint(lobbies, seed, prior, with_tracer=False)
    traced = finals_fingerprint(lobbies, seed, prior, with_tracer=True)
    assert plain == traced


def test_core_weights_sum_to_one():
    arch = Archetype(
        key="t", name="t", tribe=None,
        core={"A": 2.0, "B": 1.0, "C": 1.0}, board_count=10)
    w = core_weights(arch)
    assert abs(sum(w.values()) - 1.0) < 1e-12
    assert abs(w["A"] - 0.5) < 1e-12


def test_missing_mass_assigns_exactly_one_fate():
    card = CoreCardTrace(name="X", weight=0.25, present_at_first_2=False,
                         legally_offered_after=True, purchased_after=False)
    fate = _classify_missing_card(card, target_switched=False)
    assert fate == "B_AVAILABLE_NOT_BOUGHT"
    card2 = CoreCardTrace(name="Y", weight=0.1, present_at_first_2=True,
                          present_final=False)
    assert _classify_missing_card(card2, target_switched=False) == "E_EXISTING_CORE_LOST"


def test_cohort_unique_and_mass_reconciles_synthetic():
    # Minimal synthetic traces: one lobby, winner seat 0, one arch with 2 cores
    from hsbg_coach.build_path import load_archetypes
    arches = [a for a in load_archetypes() if len(a.core) >= 2]
    assert arches
    arch = arches[0]
    cores = list(arch.core.keys())
    c0, c1 = cores[0], cores[1]
    # Build a tiny trace where end-of-recruit turn 5 has both cores
    board2 = [
        {"name": c0, "attack": 1, "health": 1},
        {"name": c1, "attack": 1, "health": 1},
    ]
    traces = {
        "lobbies": 1,
        "seed": 9000,
        "events": [
            {
                "lobby": 0, "seat": 0, "turn": 5, "shop_generation": 0,
                "action": "end", "card": None,
                "pre_shop": [], "legal_buy_slots": [],
                "board_before": board2, "hand_before": [],
                "board_after": board2, "hand_after": [],
                "target_before": {
                    "archetype_key": arch.key, "core_have": 2, "coverage": 0.4},
                "tavern_tier": 5, "lobby_tribes": ["Neutral"] * 5,
            }
        ],
        "turn_summaries": [
            {
                "lobby": 0, "seat": 0, "turn": 5, "tavern_tier": 5,
                "board_after_recruit": board2,
                "target": {
                    "archetype_key": arch.key, "core_have": 2, "coverage": 0.4},
            }
        ],
        "player_finals": [
            {
                "lobby": 0, "seat": 0, "placement": 1,
                "final_board": board2,
                "target": {
                    "archetype_key": arch.key, "core_have": 2, "coverage": 0.4},
                "lobby_tribes": [],
            }
        ],
        "lobby_meta": [{"lobby": 0, "lobby_tribes": []}],
    }
    # Make arch eligible: empty tribes means all eligible in _archetype_eligible?
    from ml.composition_diagnostic import _archetype_eligible
    # Use tribes that include arch.tribe if set
    tribes = [arch.tribe] if arch.tribe else ["Murloc", "Beast", "Mech",
                                              "Demon", "Pirate"]
    traces["lobby_meta"][0]["lobby_tribes"] = tribes
    traces["player_finals"][0]["lobby_tribes"] = tribes

    analysis = analyze_post_assembly_gap(traces)
    keys = [(r["lobby"], r["seat"], r["archetype_key"])
            for r in analysis["state_records"]]
    assert len(keys) == len(set(keys))
    assert analysis["mass_reconciliation"]["within_tolerance"]


def test_decision_tree_availability_dominance():
    analysis = {
        "n_states": 50,
        "missing_coverage_mass_share_by_cause": {
            "A_NEVER_AVAILABLE_POST_ASSEMBLY": 0.62,
            "B_AVAILABLE_NOT_BOUGHT": 0.1,
            "C_BOUGHT_NOT_DEPLOYED": 0.05,
            "D_DEPLOYED_THEN_LOST": 0.05,
            "E_EXISTING_CORE_LOST": 0.05,
            "F_TARGET_SWITCH": 0.05,
            "G_TRANSFORM_TRIPLE_DISCOVER_PATH": 0.03,
            "H_UNRESOLVED": 0.05,
        },
        "weighted_funnel": {},
    }
    d = evaluate_phase_2k_decision(analysis)
    assert d["decision_branch"] == "a_never_available_post_assembly"


def test_fresh_policies_assert_alpha():
    prior = load_frozen_prior()
    pols = policies_for_lobby(FROZEN_ALPHA, prior, 8)
    assert all(p.alpha == 0.5 for p in pols)
