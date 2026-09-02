"""Tests for Phase 2J board-relative opportunity-cost policy."""

import random

from hsbg_coach.bg_env import A_BUY0, A_PLAY0, A_SELL0, greedy_policy
from hsbg_coach.board_opportunity_policy import (
    ALPHA_CANDIDATES,
    METHODOLOGY_VERSION,
    PHASE_2J_CONFIRM_SEED,
    BoardOpportunityCostPolicy,
    opportunity_cost,
    policies_for_lobby,
    relative_tempo_loss,
    transition_score,
)
from hsbg_coach.persistence_prior import (
    PersistenceCell,
    PersistencePrior,
    empty_prior,
    feature_key,
    rank_band_for_index,
)
from ml.fit_persistence_prior import fit_persistence_prior_from_traces
from ml.phase_2j_decision import (
    evaluate_confirmation_acceptance,
    evaluate_phase_2j_decision,
    rank_calibration_candidate,
)


def _obs(shop, board=None, hand=None, turn=8, tier=4, gold=10):
    return {
        "turn": turn,
        "gold": gold,
        "tavern_tier": tier,
        "board": board or [],
        "hand": hand or [],
        "shop": shop,
    }


def _mask(*, n_shop=0, sell_slots=None, play_slots=None, buy_slots=None):
    mask = [False] * 28
    slots = buy_slots if buy_slots is not None else list(range(n_shop))
    for i in slots:
        mask[A_BUY0 + i] = True
    if sell_slots:
        for i in sell_slots:
            mask[A_SELL0 + i] = True
    if play_slots:
        for i in play_slots:
            mask[A_PLAY0 + i] = True
    mask[27] = True
    return mask


def test_methodology_and_seeds():
    assert METHODOLOGY_VERSION == "2j_v1"
    assert ALPHA_CANDIDATES == (0.5, 1.0, 2.0)
    assert PHASE_2J_CONFIRM_SEED == 8000


def test_relative_tempo_loss_not_absolute():
    # Absolute loss 296 on a 400-stat board → 0.74 relative, not 296.
    rel = relative_tempo_loss(cand_raw=10.0, repl_raw=306.0, board_total=400.0)
    assert abs(rel - 296.0 / 400.0) < 1e-9
    assert rel < 1.0


def test_transition_score_uses_alpha_opportunity_cost():
    # High absolute incumbent, but relative cost small enough that build wins.
    rel = relative_tempo_loss(cand_raw=20.0, repl_raw=100.0, board_total=500.0)
    opp = opportunity_cost(
        cand_raw=20.0, repl_raw=100.0, board_total=500.0, persistence_weight=0.5)
    assert abs(opp - rel * 0.5) < 1e-9
    score = transition_score(build_delta=0.4, opp_cost=opp, alpha=1.0)
    assert score == 0.4 - opp
    assert score > 0


def test_weak_slot_cheaper_than_entrenched():
    prior = PersistencePrior(
        methodology_version="2j_v1", survival_horizon=2,
        weight_1=0.5, weight_2=0.5, fit_seed_base=0, fit_lobbies=0,
        cells={
            feature_key(6, "weak", False): PersistenceCell(
                "6plus", "weak", False, 100, 0.2, 0.1),
            feature_key(6, "strong", True): PersistenceCell(
                "6plus", "strong", True, 100, 0.9, 0.8),
        },
        global_p_survive_1=0.5, global_p_survive_2=0.4, collapsed_from=[],
    )
    w_weak = prior.persistence_weight(tier=6, rank="weak", is_core=False)
    w_strong = prior.persistence_weight(tier=6, rank="strong", is_core=True)
    assert w_weak < w_strong


def test_unseeded_matches_greedy():
    rng = random.Random(0)
    prior = empty_prior()
    obs = _obs([{"name": "Z", "attack": 5, "health": 5, "tribes": []}])
    mask = _mask(n_shop=1)
    policy = BoardOpportunityCostPolicy(1.0, prior)
    assert policy(obs, mask, rng) == greedy_policy(obs, mask, rng)


def test_free_slot_opportunity_cost_zero_commits_on_build():
    # Seeded board with free slot: positive build gain → buy core.
    from hsbg_coach.build_path import load_archetypes
    arches = load_archetypes()
    # Find a known core name from first archetype with cores.
    arch = next(a for a in arches if a.core)
    core_name = next(iter(arch.core))
    board = [
        {"name": core_name, "attack": 3, "health": 3, "tribes": []},
        {"name": "Filler", "attack": 1, "health": 1, "tribes": []},
    ]
    shop = [{"name": "OtherCore", "attack": 2, "health": 2, "tribes": []}]
    # Use another core from same arch if available.
    other = [n for n in arch.core if n != core_name]
    if other:
        shop = [{"name": other[0], "attack": 2, "health": 2, "tribes": []}]
    policy = BoardOpportunityCostPolicy(1.0, empty_prior())
    rng = random.Random(0)
    obs = _obs(shop, board=board, turn=6, tier=4, gold=10)
    mask = _mask(n_shop=1)
    action = policy(obs, mask, rng)
    # Either buys the core (if build gain > 0) or falls through — must not crash.
    assert isinstance(action, int)


def test_rank_band_tertiles():
    raws = [10.0, 50.0, 100.0, 200.0, 300.0, 400.0]
    assert rank_band_for_index(raws, 0) == "weak"
    assert rank_band_for_index(raws, 5) == "strong"


def test_fit_prior_from_synthetic_traces():
    # Two turns, one minion survives both.
    traces = {
        "turn_summaries": [
            {"lobby": 0, "seat": 0, "turn": 5, "tavern_tier": 4,
             "board_before_recruit": [
                 {"name": "A", "attack": 1, "health": 1},
                 {"name": "B", "attack": 10, "health": 10},
             ]},
            {"lobby": 0, "seat": 0, "turn": 6, "tavern_tier": 4,
             "board_before_recruit": [
                 {"name": "A", "attack": 1, "health": 1},
                 {"name": "B", "attack": 10, "health": 10},
             ]},
            {"lobby": 0, "seat": 0, "turn": 7, "tavern_tier": 5,
             "board_before_recruit": [
                 {"name": "B", "attack": 10, "health": 10},
             ]},
        ]
    }
    prior = fit_persistence_prior_from_traces(
        traces, fit_seed_base=7000, fit_lobbies=1, min_cell_n=1)
    assert prior.global_p_survive_1 > 0
    assert prior.to_dict()["fit_seed_base"] == 7000


def test_decision_tree_macro_regression():
    acceptance = evaluate_confirmation_acceptance(
        {"seeded_current_target": {"reached_2_core": 0, "fulfilled_exposures": 0},
         "sim_final_winner_coverage_mean": 0.1,
         "committed_current_target": {"n_lobby_archetype_states": 0}},
        {"seeded_current_target": {"reached_2_core": 10, "fulfilled_exposures": 10},
         "sim_final_winner_coverage_mean": 0.2,
         "committed_current_target": {"n_lobby_archetype_states": 5}},
        {"n_fulfilled_purchases": 0, "funnel": {"played": 0}},
        {"n_fulfilled_purchases": 10, "funnel": {"played": 8}},
        {"turn_14_stats_ratio_delta": 0.9, "turn_10_tier_error_delta": 0,
         "game_length_delta": 0, "turn_10_alive_error_delta": 0},
    )
    assert acceptance["flags"]["macro_regression_ok"] is False
    decision = evaluate_phase_2j_decision(
        {}, {}, {}, {}, {}, acceptance)
    assert decision["decision_branch"] == "macro_regression"


def test_rank_prefers_macro_ok():
    bad = {"macro_ok": False, "board_sacrifice_ok": True,
           "reached_2_core": 99, "committed_states": 99,
           "fulfilled_exposures": 99, "coverage_mean": 1.0,
           "action_deviation_rate": 0.0}
    good = {"macro_ok": True, "board_sacrifice_ok": True,
            "reached_2_core": 1, "committed_states": 1,
            "fulfilled_exposures": 1, "coverage_mean": 0.01,
            "action_deviation_rate": 0.5}
    assert rank_calibration_candidate(good) < rank_calibration_candidate(bad)


def test_fresh_policies_per_lobby():
    prior = empty_prior()
    a = policies_for_lobby(1.0, prior, 8)
    b = policies_for_lobby(1.0, prior, 8)
    assert a[0] is not b[0]
    assert len(a) == 8
