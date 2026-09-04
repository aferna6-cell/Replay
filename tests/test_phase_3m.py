"""Phase 3M earliest same-seat HP divergence — observational locks."""

import random

from hsbg_coach.bg_env import (
    PHASE_2Q_RECRUIT_VALUE_STATS,
    PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING,
    BGEnv,
    EnvMinion,
    greedy_policy,
)
from hsbg_coach.sim import Combatant, simulate_once
from ml.phase_2s_prereg import (
    GATE_GAME_LENGTH_DELTA_FLOOR,
    GATE_MEAN_COMBAT_LOSS_MAX,
    GATE_REPLACE_RATE_MIN,
    GATE_T10_POST_SCALE_DELTA_FLOOR,
    GATE_T10_POST_SCALE_MIN,
)
from ml.phase_3e_prereg import PHASE_3D_BOARD_POOL_MAGNITUDE as A1_3E
from ml.phase_3m_prereg import (
    BYE_TOKEN,
    CANDIDATE_CHOICE_IDENTITY,
    CHAIN_COMPONENTS,
    CHAIN_HP_RECONCILE_IDENTITY,
    CHAIN_RECONCILE_IDENTITY,
    EARLY_TURNS,
    ELIGIBILITY_TIMING_IDENTITY,
    ELIMINATION_IDENTITY,
    FEATURE_TOGGLE,
    FEATURE_TOGGLE_DEFAULT,
    FIRST_DIVERGENCE_COMPONENTS,
    FIRST_DIVERGENCE_RECONCILE_IDENTITY,
    FORBIDDEN_RANGES,
    FROZEN_ALPHA,
    GHOST_TOKEN,
    HISTORY_LINK_IDENTITY,
    HOLD_PRS,
    HP_FLOW_IDENTITY,
    HP_GAP_COMPONENTS,
    HP_GAP_RECONCILE_IDENTITY,
    HP_WALK_FROM_TURN,
    IMPACT_ATTACK_IDENTITY,
    INSTRUMENT_TURNS,
    LATE_TURNS,
    LEFTOVER_RECONCILE_IDENTITY,
    LINEAGE_IDENTITY,
    LOW_TIERS,
    LOW_WINNER_START_TIERS,
    MATCHMAKING_RECONCILE_IDENTITY,
    METHODOLOGY_VERSION,
    PAIRED_SEAT_IDENTITY,
    PAIRING_IDENTITY,
    PAIRING_TURNS,
    PHASE_3D_BOARD_POOL_MAGNITUDE,
    PHASE_3E_CARRY_DELTA,
    PHASE_3E_CARRY_SHARE_OF_A1,
    PHASE_3E_PUNCH_DELTA_CARRY,
    PHASE_3F_SELECTION_SHARE,
    PHASE_3F_UNCOND_SHARE,
    PHASE_3G_MIXTURE,
    PHASE_3G_MIXTURE_SHARE,
    PHASE_3G_MIX_ROLE_SHARE,
    PHASE_3G_N_CONTROL,
    PHASE_3G_N_TREATMENT,
    PHASE_3G_WITHIN_SHARE,
    PHASE_3H_COLLAPSE,
    PHASE_3H_LATE_CONTROL,
    PHASE_3H_LATE_TREATMENT,
    PHASE_3H_LEFTOVER,
    PHASE_3H_SHARE_LEFTOVER,
    PHASE_3I_DIFFERENT_OPPONENT,
    PHASE_3I_KIND_MISMATCH,
    PHASE_3I_OUTCOME_FLIP,
    PHASE_3I_PAIRING_SCHEDULE,
    PHASE_3I_RESIDUAL,
    PHASE_3I_SHARE_PAIRING_SCHEDULE,
    PHASE_3I_SURVIVOR_SUBSTITUTION,
    PHASE_3J_ELIGIBILITY,
    PHASE_3J_ELIG_DIFFERENT_OPPONENT,
    PHASE_3J_ELIG_KIND_MISMATCH,
    PHASE_3J_HISTORY_LEGAL,
    PHASE_3J_RNG_ORDER,
    PHASE_3J_SHARE_ELIGIBILITY,
    PHASE_3K_CONTROL_OPPONENT,
    PHASE_3K_NAMED,
    PHASE_3K_PRIOR_HP,
    PHASE_3K_SHARE_PRIOR_HP,
    PHASE_3K_SHARE_THIRD_PARTY,
    PHASE_3K_THIRD_PARTY,
    PHASE_3K_TREATMENT_EARLIER,
    PHASE_3L_PRIOR_HP,
    PHASE_3L_SAME_SEAT_EARLIER,
    PHASE_3L_SHARE_EARLIER,
    PHASE_3L_SHARE_PRIOR_HP,
    PHASE_3M_LOBBIES,
    PHASE_3M_SEED,
    POOL_FLOW_IDENTITY,
    REUSED_SEED_HI,
    REUSED_SEED_LO,
    ROW_ELIM_HP_IDENTITY,
    ROW_HISTORY_DIVERGENCE_IDENTITY,
    SHARE_DOMINANT,
    TRACE_FROM_TURN,
    VERY_LATE_TURNS,
    WEIGHT_RECONCILIATION_IDENTITY,
    assert_seed_range_allowed,
    classify_first_divergence,
    diagnose_phase_3m,
    share_of_class1,
)
from ml.board_retention_diagnostic import compare_retention
from ml.carry_divergence_diagnostic import (
    build_seat_trajectories,
    reconcile_history_links,
)
from ml.elimination_chain_diagnostic import (
    attribute_elimination_chain,
    compare_chain,
    reconcile_eliminations,
    reconcile_hp_flow,
)
from ml.elimination_timing_diagnostic import (
    EliminationTimingTracer,
    attribute_elimination_timing,
    compare_elimination,
)
from ml.hp_divergence_diagnostic import (
    attribute_first_divergence,
    compare_first_divergence,
    find_first_hp_divergence,
)
from ml.matchmaking_divergence_diagnostic import (
    ghost_bye_eligibility,
    legal_candidates_for_seat,
    chosen_opponent_for_seat,
    choice_in_candidates,
)
from ml.pairing_who_wins_diagnostic import (
    attribute_leftover_pairing,
    compare_pairing,
)
from ml.punch_selection_diagnostic import (
    collect_punch_sample_rows,
    decompose_punch_selection,
)
from ml.survivor_tier_damage_diagnostic import rules_faithful_hero_damage


def test_methodology_is_3m_v1_default_off():
    assert METHODOLOGY_VERSION == "3m_v1"
    assert FEATURE_TOGGLE == "PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING"
    assert FEATURE_TOGGLE_DEFAULT is False
    assert PHASE_2Q_RECRUIT_VALUE_STATS is False
    assert PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING is False
    assert SHARE_DOMINANT == 0.70
    assert LOW_TIERS == (1, 2, 3)
    assert LOW_WINNER_START_TIERS == (1, 2, 3)
    assert EARLY_TURNS == (7, 8, 9)
    assert LATE_TURNS == (10, 11, 12, 13, 14)
    assert VERY_LATE_TURNS == (12, 13, 14)
    assert PAIRING_TURNS == LATE_TURNS
    assert TRACE_FROM_TURN == 7
    assert HP_WALK_FROM_TURN == 1
    assert CHAIN_COMPONENTS == (
        "same_seat_earlier_elimination",
        "different_seat_alive_set_cascade",
        "same_fight_outcome_flip",
        "same_outcome_damage_threshold",
        "unreconciled",
    )
    assert HP_GAP_COMPONENTS == (
        "accumulated_prior_hp",
        "current_fight_hit",
        "current_fight_damage_magnitude",
        "hp_unreconciled",
    )
    assert FIRST_DIVERGENCE_COMPONENTS == (
        "prior_alive_set_or_pairing",
        "same_pairing_outcome_flip",
        "same_outcome_damage",
        "inherited_hp_carry",
        "unreconciled",
    )
    assert GHOST_TOKEN == "ghost"
    assert BYE_TOKEN == "bye"
    assert "post_hp = pre_hp - applied_to_seat" in HP_FLOW_IDENTITY
    assert "elimination_turn is the first" in ELIMINATION_IDENTITY
    assert "treatment_eliminated_earlier +" in ELIGIBILITY_TIMING_IDENTITY
    assert "same_seat_earlier_elimination +" in CHAIN_RECONCILE_IDENTITY
    assert "accumulated_prior_hp +" in CHAIN_HP_RECONCILE_IDENTITY
    assert "every 3K third-party row maps" in ROW_ELIM_HP_IDENTITY
    assert "prior_alive_set_or_pairing +" in FIRST_DIVERGENCE_RECONCILE_IDENTITY
    assert "every class-(1) row maps" in ROW_HISTORY_DIVERGENCE_IDENTITY
    assert "accumulated_prior_hp +" in HP_GAP_RECONCILE_IDENTITY
    assert "chosen opponent is an element" in CANDIDATE_CHOICE_IDENTITY
    assert "eligibility + history_legal" in MATCHMAKING_RECONCILE_IDENTITY
    assert HISTORY_LINK_IDENTITY == (
        "punch.opp_carry = seat_history[turn].attack_pool_recruit_start"
    )
    assert "opponent_seat" in PAIRING_IDENTITY
    assert "pairing_schedule + outcome_flip" in LEFTOVER_RECONCILE_IDENTITY
    assert LINEAGE_IDENTITY == (
        "t1t3_end = t1t3_start + t1t3_added - t1t3_removed"
    )
    assert "paired (seed, seat)" in PAIRED_SEAT_IDENTITY
    assert POOL_FLOW_IDENTITY == (
        "post = pre + add - represented_loss_or_transfer"
    )
    assert IMPACT_ATTACK_IDENTITY == (
        "impact_attack = start_recruit + start_pool_share + combat_delta"
    )
    assert "sum_k n_arm(k) = N_arm" in WEIGHT_RECONCILIATION_IDENTITY


def test_reuses_2s_dev_and_forbids_confirm():
    assert PHASE_3M_SEED == 14200
    assert PHASE_3M_LOBBIES == 500
    assert REUSED_SEED_LO == 14200
    assert REUSED_SEED_HI == 14699
    assert INSTRUMENT_TURNS == tuple(range(7, 15))
    assert (11500, 11699) in FORBIDDEN_RANGES
    assert (13700, 14199) in FORBIDDEN_RANGES
    assert_seed_range_allowed(14200, 500)
    assert_seed_range_allowed(14200, 8)
    try:
        assert_seed_range_allowed(11500, 10)
        raise AssertionError("expected confirm overlap to fail")
    except ValueError:
        pass
    try:
        assert_seed_range_allowed(14700, 1)
        raise AssertionError("expected new seeds to fail")
    except ValueError:
        pass


def test_hold_stack_includes_3l_and_frozen_alpha():
    assert HOLD_PRS == (
        29, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
        50, 51, 52, 53, 54, 55, 56, 57, 58,
    )
    assert FROZEN_ALPHA == 0.5
    assert abs(PHASE_3D_BOARD_POOL_MAGNITUDE - 0.4216721428553852) < 1e-9
    assert abs(A1_3E - PHASE_3D_BOARD_POOL_MAGNITUDE) < 1e-12
    assert abs(PHASE_3E_CARRY_DELTA - 0.30513688784757187) < 1e-9
    assert abs(PHASE_3E_CARRY_SHARE_OF_A1 - 0.7236353954551374) < 1e-9
    assert abs(PHASE_3E_PUNCH_DELTA_CARRY - (-196.33317557443002)) < 1e-9
    assert abs(PHASE_3F_UNCOND_SHARE - 0.09084015396406948) < 1e-9
    assert abs(PHASE_3F_SELECTION_SHARE - 0.9091598460359305) < 1e-9
    assert abs(PHASE_3G_MIXTURE - (-196.52943934946725)) < 1e-9
    assert abs(PHASE_3G_MIXTURE_SHARE - 1.0009996465165045) < 1e-9
    assert abs(PHASE_3G_WITHIN_SHARE - (-0.0009996465165047867)) < 1e-9
    assert abs(PHASE_3G_MIX_ROLE_SHARE - 0.8158532590211308) < 1e-9
    assert PHASE_3G_N_CONTROL == 54223
    assert PHASE_3G_N_TREATMENT == 50116
    assert PHASE_3H_LATE_CONTROL == 17924
    assert PHASE_3H_LATE_TREATMENT == 4273
    assert PHASE_3H_COLLAPSE == 13651
    assert PHASE_3H_LEFTOVER == 7155
    assert abs(PHASE_3H_SHARE_LEFTOVER - 0.5241374258296095) < 1e-9
    assert PHASE_3I_PAIRING_SCHEDULE == 5952
    assert PHASE_3I_OUTCOME_FLIP == 668
    assert PHASE_3I_SURVIVOR_SUBSTITUTION == 292
    assert PHASE_3I_RESIDUAL == 243
    assert PHASE_3I_DIFFERENT_OPPONENT == 5009
    assert PHASE_3I_KIND_MISMATCH == 943
    assert abs(PHASE_3I_SHARE_PAIRING_SCHEDULE - 0.8318658280922432) < 1e-9
    assert PHASE_3J_ELIGIBILITY == 5648
    assert PHASE_3J_HISTORY_LEGAL == 0
    assert PHASE_3J_RNG_ORDER == 304
    assert PHASE_3J_ELIG_DIFFERENT_OPPONENT == 4771
    assert PHASE_3J_ELIG_KIND_MISMATCH == 877
    assert abs(PHASE_3J_SHARE_ELIGIBILITY - 0.9489247311827957) < 1e-9
    assert PHASE_3K_THIRD_PARTY == 3701
    assert PHASE_3K_TREATMENT_EARLIER == 1108
    assert PHASE_3K_CONTROL_OPPONENT == 839
    assert PHASE_3K_NAMED == 1947
    assert PHASE_3K_PRIOR_HP == 1818
    assert abs(PHASE_3K_SHARE_THIRD_PARTY - 0.6552762039660056) < 1e-9
    assert abs(PHASE_3K_SHARE_PRIOR_HP - 0.9337442218798151) < 1e-9
    assert PHASE_3L_SAME_SEAT_EARLIER == 2082
    assert PHASE_3L_PRIOR_HP == 1858
    assert abs(PHASE_3L_SHARE_EARLIER - 0.5625506619832478) < 1e-9
    assert abs(PHASE_3L_SHARE_PRIOR_HP - 0.8924111431316043) < 1e-9
    d = diagnose_phase_3m()
    assert d["no_merge"] is True
    assert d["no_hero_damage_retune"] is True
    assert d["no_behavior_change"] is True
    assert d["no_2q_rewrite"] is True
    assert d["no_scaling_constant_change"] is True
    assert d["keep_hold_prs"][-1] == 58
    assert d["history_filters_applied"] is False
    assert d["candidate_choice_identity"] == CANDIDATE_CHOICE_IDENTITY
    assert d["chain_reconcile_identity"] == CHAIN_RECONCILE_IDENTITY
    assert d["first_divergence_reconcile_identity"] == FIRST_DIVERGENCE_RECONCILE_IDENTITY
    assert d["row_history_divergence_identity"] == ROW_HISTORY_DIVERGENCE_IDENTITY


def test_2s_gates_unchanged():
    assert GATE_REPLACE_RATE_MIN == 0.10
    assert GATE_T10_POST_SCALE_MIN == 0.85
    assert GATE_T10_POST_SCALE_DELTA_FLOOR == -0.10
    assert GATE_GAME_LENGTH_DELTA_FLOOR == -0.50
    assert GATE_MEAN_COMBAT_LOSS_MAX == 20.0


def test_hero_damage_formula_unchanged():
    board = [
        EnvMinion("id-a", "a", 5, 10, 10, [], []),
        EnvMinion("id-b", "b", 3, 8, 8, [], []),
        EnvMinion("id-c", "c", 4, 6, 6, [], []),
    ]
    raw = 4 + 3
    survivors = max(1, abs(raw) - 4)
    avg = (5 + 3 + 4) / 3
    assert BGEnv._hero_damage(raw, 4, board) == 4 + max(1, round(survivors * avg))


def test_share_and_classify_first_divergence():
    assert abs(share_of_class1(70.0, denom=100.0) - 0.70) < 1e-9
    assert share_of_class1(None, denom=100.0) is None
    base = dict(
        found_event=True, control_obs=True, treatment_obs=True,
        pre_hp_equal=True, pairing_equal=True,
        outcomes_equal=True, applied_equal=True,
    )
    assert classify_first_divergence(
        **{**base, "pre_hp_equal": False}
    ) == "inherited_hp_carry"
    assert classify_first_divergence(
        **{**base, "pairing_equal": False}
    ) == "prior_alive_set_or_pairing"
    assert classify_first_divergence(
        **{**base, "outcomes_equal": False}
    ) == "same_pairing_outcome_flip"
    assert classify_first_divergence(
        **{**base, "applied_equal": False}
    ) == "same_outcome_damage"
    assert classify_first_divergence(**base) == "unreconciled"
    assert classify_first_divergence(found_event=False) == "unreconciled"
    assert classify_first_divergence(
        **{**base, "control_obs": False}
    ) == "unreconciled"
    # Inherited pre-HP beats pairing / outcome / damage.
    assert classify_first_divergence(
        **{**base, "pre_hp_equal": False, "pairing_equal": False,
           "outcomes_equal": False, "applied_equal": False}
    ) == "inherited_hp_carry"
    # Pairing beats outcome / damage when pre-HP matches.
    assert classify_first_divergence(
        **{**base, "pairing_equal": False, "outcomes_equal": False}
    ) == "prior_alive_set_or_pairing"
    # No paired fight at the first observed split is inherited carry.
    assert classify_first_divergence(
        **{**base, "paired_fights_present": False}
    ) == "inherited_hp_carry"


def test_diagnose_routes():
    pairing = diagnose_phase_3m({
        "attribution": {
            "share_prior_alive_set_or_pairing": 0.80,
            "share_same_pairing_outcome_flip": 0.10,
            "share_same_outcome_damage": 0.06,
            "share_inherited_hp_carry": 0.03,
            "share_unreconciled": 0.01,
        }
    })
    assert pairing["primary_finding"] == "prior_alive_set_or_pairing_dominates"
    assert "matchmaking" in pairing["recommended_next_step"]
    flip = diagnose_phase_3m({
        "attribution": {
            "share_prior_alive_set_or_pairing": 0.10,
            "share_same_pairing_outcome_flip": 0.80,
            "share_same_outcome_damage": 0.06,
            "share_inherited_hp_carry": 0.03,
            "share_unreconciled": 0.01,
        }
    })
    assert flip["primary_finding"] == "same_pairing_outcome_dominates"
    assert "combat-outcome" in flip["recommended_next_step"]
    damage = diagnose_phase_3m({
        "attribution": {
            "share_prior_alive_set_or_pairing": 0.10,
            "share_same_pairing_outcome_flip": 0.08,
            "share_same_outcome_damage": 0.76,
            "share_inherited_hp_carry": 0.05,
            "share_unreconciled": 0.01,
        }
    })
    assert damage["primary_finding"] == "same_outcome_damage_dominates"
    assert "matched-state" in damage["recommended_next_step"]
    inherited = diagnose_phase_3m({
        "attribution": {
            "share_prior_alive_set_or_pairing": 0.10,
            "share_same_pairing_outcome_flip": 0.08,
            "share_same_outcome_damage": 0.05,
            "share_inherited_hp_carry": 0.76,
            "share_unreconciled": 0.01,
        }
    })
    assert inherited["primary_finding"] == "inherited_hp_carry_dominates"
    assert "earliest originating fight" in inherited["recommended_next_step"]
    assert "_hero_damage" in inherited["recommended_next_step"]
    mixed = diagnose_phase_3m({
        "attribution": {
            "share_prior_alive_set_or_pairing": 0.40,
            "share_same_pairing_outcome_flip": 0.25,
            "share_same_outcome_damage": 0.20,
            "share_inherited_hp_carry": 0.10,
            "share_unreconciled": 0.05,
        }
    })
    assert mixed["primary_finding"] == "mixed_route_to_larger"
    leftover_d = diagnose_phase_3m({
        "attribution": {
            "share_prior_alive_set_or_pairing": 0.10,
            "share_same_pairing_outcome_flip": 0.08,
            "share_same_outcome_damage": 0.05,
            "share_inherited_hp_carry": 0.04,
            "share_unreconciled": 0.73,
        }
    })
    assert leftover_d["primary_finding"] == "ranked_residual_needs_next_observable"
    smoke = diagnose_phase_3m(mixed, non_evaluative=True)
    assert smoke["primary_finding"] == "measurement_smoke_non_evaluative"
    assert smoke["evaluative"] is False


def _play(seed, with_hook):
    env = BGEnv(seed=seed)
    tracer = None
    if with_hook:
        tracer = EliminationTimingTracer(0, seed, "obs")
        tracer.attach_to_env(env)
    recs = env.play_scripted(
        [greedy_policy] * env.n_players,
        recruit_tracer=tracer,
    )
    return {
        "env": env,
        "length": max((r["turn"] for r in recs), default=0),
        "placements": [p.placement for p in env.players],
        "hp": [p.hp for p in env.players],
        "n_fights": len(tracer.fights) if tracer else 0,
        "tracer": tracer,
        "rng_state": env.rng.getstate(),
    }


def test_combat_pairing_hooks_are_observational_same_seed():
    """Hooked vs unhooked: placements, HP, length, outcome, and RNG state match."""
    plain = _play(14200, False)
    hooked = _play(14200, True)
    assert hooked["length"] == plain["length"]
    assert hooked["placements"] == plain["placements"]
    assert hooked["hp"] == plain["hp"]
    assert hooked["rng_state"] == plain["rng_state"]
    assert hooked["n_fights"] > 0
    tracer = hooked["tracer"]
    assert tracer.turn_rows
    assert tracer.board_snapshots
    assert tracer.pairing_decisions
    assert tracer.hp_rows
    assert tracer.eliminations
    late = [d for d in tracer.pairing_decisions if int(d["turn"]) in LATE_TURNS]
    assert late
    for rec in late:
        assert rec["history_filters_applied"] is False
        assert rec["rng_state_digest_pre"]
        assert rec["hp_at_pair"]
        for seat, view in rec["per_seat"].items():
            assert view["choice_in_candidates"] is True
    traj = build_seat_trajectories(tracer.turn_rows, tracer.fights)
    assert traj
    hist = reconcile_history_links(tracer.fights, tracer.turn_rows)
    if hist["n_punch_rows"] > 0:
        assert hist["n_carry_mismatch"] == 0
        assert hist["n_missing_turn_row"] == 0
        assert hist["p_ok"] == 1.0
    rows = collect_punch_sample_rows(tracer.fights)
    assert rows
    hp = reconcile_hp_flow(tracer.fights)
    assert hp["ok"] is True
    elim = reconcile_eliminations(
        tracer.fights, tracer.eliminations, n_lobbies=1,
    )
    assert elim["census_ok"] is True
    assert elim["n_combat_eliminations"] + elim["n_survived"] == 8
    hits = [
        f for f in tracer.fights
        if int(f.get("applied_hp_loss") or 0) > 0
        and int(f.get("turn") or 0) in INSTRUMENT_TURNS
    ]
    assert hits
    for f in hits:
        assert f["counterfactual_damage"] == rules_faithful_hero_damage(
            f["winner_tavern_tier"],
            [s["tier"] for s in f["actual_survivors"]],
        )


def test_unhooked_simulate_once_unchanged():
    r1 = random.Random(42)
    r2 = random.Random(42)
    a = [Combatant(8, 5, name="a", tier=3, recruit_attack=3)]
    b = [Combatant(4, 8, name="b", tier=2, recruit_attack=4)]
    raw1 = simulate_once(a, b, r1, tier_a=3, tier_b=2)
    trace = {}
    raw2 = simulate_once(a, b, r2, tier_a=3, tier_b=2, trace=trace)
    assert raw1 == raw2
    assert r1.getstate() == r2.getstate()


def _punch(seed, seat, turn, tier=1, carry=800.0, loser=1):
    return {
        "seed": seed,
        "turn": turn,
        "winner_seat": seat,
        "loser_seat": loser,
        "winner_start_tier": tier,
        "carry": carry,
    }


def _turn(seed, seat, turn, t1t3, *, alive=True):
    return {
        "seed": seed,
        "seat": seat,
        "turn": turn,
        "t1t3_count_combat_start": t1t3,
        "t1t3_count_recruit_start": t1t3,
        "alive_at_combat": alive,
        "alive_at_recruit": alive,
        "n_replacements": 0,
        "shop_t1t3_offers_recruit_start": 1,
        "attack_pool_recruit_start": 80.0,
        "scale_add_attack": 0.0,
        "flow_ok": True,
        "n_alive": 6,
    }


def _fight(seed, turn, winner, loser, *, kind="live", start_tier=1, raw=4,
           applied=4, t1t3_a=3, t1t3_b=2, pre_a=30, pre_b=30,
           post_a=None, post_b=None):
    sa, sb = winner, loser
    if post_a is None:
        post_a = pre_a
    if post_b is None:
        post_b = pre_b - applied if raw > 0 else pre_b
    start = [{
        "tier": start_tier, "n_damaging_hits": 1, "n_hits": 1,
        "survived": start_tier <= 3, "attacked": True,
        "recruit_raw": 4, "synthetic_share": 2, "combat_raw": 6,
        "opp_carry_attack_pool": 80.0,
        "opp_scale_add_attack": 0.0,
        "opp_attack_pool_recruit_start": 80.0,
        "opp_n_alive": 6, "opp_alive": True,
    }]
    return {
        "seed": seed, "lobby": 0, "turn": turn, "kind": kind,
        "ghost": kind == "ghost", "seat_a": sa, "seat_b": None if kind != "live" else sb,
        "winner_seat": winner if raw > 0 else (None if raw == 0 else loser),
        "loser_seat": loser if raw > 0 else (None if raw == 0 else winner),
        "applied_hp_loss": applied, "winner_tavern_tier": 4,
        "fight_outcome": "a_win" if raw > 0 else ("tie" if raw == 0 else "b_win"),
        "outcome": "win" if raw > 0 else ("tie" if raw == 0 else "loss"),
        "raw": raw, "combat_margin_raw": raw,
        "pre_hp_a": pre_a, "pre_hp_b": pre_b, "post_hp_a": post_a, "post_hp_b": post_b,
        "actual_survivor_count": 2, "actual_survivor_tier_sum": 8,
        "start_minions": start,
        "pairing": {
            "seat_a": sa,
            "seat_b": None if kind != "live" else sb,
            "kind": kind,
            "winner_seat": winner if raw > 0 else None,
            "fight_outcome": "a_win" if raw > 0 else "tie",
            "combat_margin_raw": raw,
            "survivor_count": 2,
            "survivor_tier_sum": 8,
            "low_tier_attacked": start_tier <= 3,
            "low_tier_survived": start_tier <= 3,
            "a": {
                "t1t3_count": t1t3_a, "t1t3_share": 0.5, "tavern_tier": 4,
                "recruit_raw": 40.0, "abstract_pool_raw": 8.0, "combat_raw": 48.0,
                "pre_fight_hp": pre_a, "board_size": 6, "alive": True,
            },
            "b": {
                "t1t3_count": t1t3_b, "t1t3_share": 0.4, "tavern_tier": 5,
                "recruit_raw": 50.0, "abstract_pool_raw": 12.0, "combat_raw": 62.0,
                "pre_fight_hp": pre_b, "board_size": 6, "alive": True,
            },
            "alive_next": {str(sa): True, str(sb): post_b > 0} if kind == "live" else {str(sa): True},
        },
    }


def _decision(seed, turn, alive, pairs, *, dead=None, chosen_map=None,
              digest="aa", index=3):
    dead = list(dead or [])
    elig = ghost_bye_eligibility(alive, dead)
    per_seat = {}
    for seat in alive:
        legal = legal_candidates_for_seat(
            seat, alive_seats=alive,
            ghost_eligible=elig["ghost_eligible"],
            bye_eligible=elig["bye_eligible"],
        )
        chosen = (chosen_map or {}).get(seat)
        if chosen is None:
            chosen = chosen_opponent_for_seat(
                seat, pairs,
                ghost_eligible=elig["ghost_eligible"],
                bye_eligible=elig["bye_eligible"],
            )
        per_seat[str(int(seat))] = {
            "legal_candidates": legal,
            "prior_opponents": [],
            "history_constrained_candidates": legal,
            "chosen": chosen,
            "choice_in_candidates": choice_in_candidates(chosen, legal),
            "pairing_index": list(alive).index(seat) if seat in alive else None,
        }
    return {
        "seed": seed, "turn": turn, "arm": "obs",
        "alive_seats": list(alive),
        "dead_with_board_seats": dead,
        **elig,
        "rng_state_digest_pre": digest,
        "rng_index_pre": index,
        "shuffled_order": [a for pair in pairs for a in pair if a is not None],
        "pairs": [list(p) for p in pairs],
        "per_seat": per_seat,
    }


def _arm_bundle(seed, *, leftover=0, opp=1, alive=None, pairs=None, dead=None,
                fights=None, elims=None, t1t3=3, hp_rows=None):
    alive = list(alive or [0, 1, 2, 3])
    pairs = list(pairs or [(0, 1), (2, 3)])
    turns = list(INSTRUMENT_TURNS)
    return {
        "turn_rows": [_turn(seed, leftover, t, t1t3) for t in turns],
        "fights": fights if fights is not None else [
            _fight(seed, t, leftover, opp) for t in LATE_TURNS
        ],
        "pairing_decisions": [
            _decision(seed, t, alive, pairs, dead=dead) for t in turns
        ],
        "last_t1t3_losses": [],
        "t1t3_events": [],
        "eliminations": list(elims or []),
        "hp_rows": list(hp_rows or []),
        "replacement_events": [],
        "board_snapshots": [],
        "game_lengths": [14],
        "n_lobbies": 1,
    }


def _ghost_fights(seed, leftover):
    fights = [
        _fight(seed, t, leftover, None, kind="ghost", raw=0, applied=0, pre_b=None)
        for t in LATE_TURNS
    ]
    for f in fights:
        f["seat_b"] = None
        f["loser_seat"] = None
        f["winner_seat"] = None
        f["pre_hp_b"] = None
        f["post_hp_b"] = None
        f["pairing"]["seat_b"] = None
        f["pairing"]["kind"] = "ghost"
        f["pairing"]["b"] = {}
    return fights


def _class1_base(*, origin="inherited"):
    leftover = [_punch(14200, 0, t, loser=1) for t in LATE_TURNS] * 4
    c_alive = [0, 1, 2, 3]
    t_alive_late = [0, 1, 3]
    c_decs = [_decision(14200, t, c_alive, [(0, 1), (2, 3)]) for t in INSTRUMENT_TURNS]
    t_decs = (
        [_decision(14200, t, c_alive, [(0, 1), (2, 3)]) for t in (7, 8)]
        + [_decision(14200, t, t_alive_late, [(0, 1), (3, None)], dead=[2, 4, 5, 6, 7])
           for t in (9, 10, 11, 12, 13, 14)]
    )
    c_fights = [_fight(14200, t, 0, 1) for t in LATE_TURNS]
    t_fights = _ghost_fights(14200, 0)
    if origin == "inherited":
        c_fights.append(_fight(14200, 8, 0, 2, pre_b=12, applied=4, post_b=8))
        c_fights.append(_fight(14200, 11, 0, 2, pre_b=8, applied=10, post_b=-2))
        t_fights.append(_fight(14200, 8, 3, 2, pre_a=20, pre_b=3, applied=4, post_b=-1))
        c_hp = [
            {"seed": 14200, "seat": 2, "turn": 7, "hp": 12, "alive": True},
            {"seed": 14200, "seat": 2, "turn": 8, "hp": 8, "alive": True},
            {"seed": 14200, "seat": 2, "turn": 11, "hp": 0, "alive": False},
        ]
        t_hp = [
            {"seed": 14200, "seat": 2, "turn": 7, "hp": 3, "alive": True},
            {"seed": 14200, "seat": 2, "turn": 8, "hp": 0, "alive": False},
        ]
    elif origin == "pairing":
        c_fights.append(_fight(14200, 7, 0, 2, pre_b=20, applied=4, post_b=16))
        t_fights.append(_fight(14200, 7, 3, 2, pre_a=20, pre_b=20, applied=8, post_b=12))
        c_fights.append(_fight(14200, 8, 0, 2, pre_b=16, applied=4, post_b=12))
        t_fights.append(_fight(14200, 8, 3, 2, pre_a=20, pre_b=12, applied=12, post_b=0))
        c_hp = [
            {"seed": 14200, "seat": 2, "turn": 7, "hp": 16, "alive": True},
            {"seed": 14200, "seat": 2, "turn": 8, "hp": 12, "alive": True},
        ]
        t_hp = [
            {"seed": 14200, "seat": 2, "turn": 7, "hp": 12, "alive": True},
            {"seed": 14200, "seat": 2, "turn": 8, "hp": 0, "alive": False},
        ]
    elif origin == "flip":
        # T7 same pairing is the first HP split; T8 pairing differs so 3L
        # still labels the elimination as class-(1) same-seat earlier.
        c_fights.append(_fight(14200, 7, 2, 3, pre_a=20, pre_b=20, applied=0, post_a=20, post_b=20))
        t_fights.append(_fight(14200, 7, 3, 2, pre_a=20, pre_b=20, applied=8, post_a=20, post_b=12))
        c_fights.append(_fight(14200, 8, 0, 2, pre_b=20, applied=4, post_b=16))
        t_fights.append(_fight(14200, 8, 3, 2, pre_a=20, pre_b=12, applied=12, post_b=0))
        c_hp = [
            {"seed": 14200, "seat": 2, "turn": 7, "hp": 20, "alive": True},
            {"seed": 14200, "seat": 2, "turn": 8, "hp": 20, "alive": True},
        ]
        t_hp = [
            {"seed": 14200, "seat": 2, "turn": 7, "hp": 12, "alive": True},
            {"seed": 14200, "seat": 2, "turn": 8, "hp": 0, "alive": False},
        ]
    else:  # damage
        c_fights.append(_fight(14200, 7, 3, 2, pre_a=20, pre_b=20, applied=4, post_b=16))
        t_fights.append(_fight(14200, 7, 3, 2, pre_a=20, pre_b=20, applied=8, post_b=12))
        c_fights.append(_fight(14200, 8, 0, 2, pre_b=16, applied=4, post_b=12))
        t_fights.append(_fight(14200, 8, 3, 2, pre_a=20, pre_b=12, applied=12, post_b=0))
        c_hp = [
            {"seed": 14200, "seat": 2, "turn": 7, "hp": 16, "alive": True},
            {"seed": 14200, "seat": 2, "turn": 8, "hp": 12, "alive": True},
        ]
        t_hp = [
            {"seed": 14200, "seat": 2, "turn": 7, "hp": 12, "alive": True},
            {"seed": 14200, "seat": 2, "turn": 8, "hp": 0, "alive": False},
        ]
    control = _arm_bundle(
        14200, fights=c_fights,
        elims=[{"seed": 14200, "seat": 2, "turn": 11 if origin == "inherited" else 14,
                "hp": 0 if origin == "inherited" else 12, "survived": origin != "inherited"}],
        hp_rows=c_hp,
    )
    if origin != "inherited":
        control["eliminations"] = []
    control["pairing_decisions"] = c_decs
    treatment = _arm_bundle(
        14200, alive=t_alive_late, pairs=[(0, 1), (3, None)],
        dead=[2, 4, 5, 6, 7], fights=t_fights,
        elims=[{"seed": 14200, "seat": 2, "turn": 8, "hp": 0, "survived": False}],
        hp_rows=t_hp,
    )
    treatment["pairing_decisions"] = t_decs
    return leftover, control, treatment


def test_attribute_routes_inherited_hp_carry():
    leftover, control, treatment = _class1_base(origin="inherited")
    timing = attribute_elimination_timing(
        control, treatment, leftover_rows=leftover, treatment_punch=[],
    )
    assert (timing["share_ghost_bye_third_party"] or 0.0) > 0.70
    chain = attribute_elimination_chain(
        control, treatment, leftover_rows=leftover, treatment_punch=[],
    )
    assert chain["reconciliation_ok"] is True
    assert (chain["share_same_seat_earlier_elimination"] or 0.0) > 0.70
    attr = attribute_first_divergence(
        control, treatment, leftover_rows=leftover, treatment_punch=[],
    )
    assert attr["reconciliation_ok"] is True
    assert attr["row_divergence_ok"] is True
    assert attr["row_history_ok"] is True
    assert (attr["share_inherited_hp_carry"] or 0.0) > 0.70
    examples = attr["examples"]["inherited_hp_carry"]
    assert examples
    rec = examples[0]
    assert rec["causal_seat"] == 2
    assert rec["found_event"] is True
    assert rec["first_divergence_turn"] <= rec["earlier_elimination_turn"]
    assert rec["control_post_hp"] != rec["treatment_post_hp"]
    assert rec["class"] == "inherited_hp_carry"
    decision = diagnose_phase_3m({"attribution": attr})
    assert decision["primary_finding"] == "inherited_hp_carry_dominates"


def test_attribute_routes_prior_alive_set_or_pairing():
    leftover, control, treatment = _class1_base(origin="pairing")
    chain = attribute_elimination_chain(
        control, treatment, leftover_rows=leftover, treatment_punch=[],
    )
    assert (chain["share_same_seat_earlier_elimination"] or 0.0) > 0.70
    attr = attribute_first_divergence(
        control, treatment, leftover_rows=leftover, treatment_punch=[],
    )
    assert attr["reconciliation_ok"] is True
    assert attr["row_divergence_ok"] is True
    assert (attr["share_prior_alive_set_or_pairing"] or 0.0) > 0.70
    examples = attr["examples"]["prior_alive_set_or_pairing"]
    assert examples
    rec = examples[0]
    assert rec["causal_seat"] == 2
    assert rec["first_divergence_turn"] == 7
    assert rec["pairing_equal"] is False
    assert rec["control_pre_hp"] == rec["treatment_pre_hp"] == 20
    assert rec["control_opponent"] != rec["treatment_opponent"]
    assert rec["control_decisive"]["tavern_tier"] == 5
    assert rec["treatment_decisive"]["board_recruit_raw"] == 50.0
    decision = diagnose_phase_3m({"attribution": attr})
    assert decision["primary_finding"] == "prior_alive_set_or_pairing_dominates"


def test_attribute_routes_same_pairing_outcome_flip():
    leftover, control, treatment = _class1_base(origin="flip")
    chain = attribute_elimination_chain(
        control, treatment, leftover_rows=leftover, treatment_punch=[],
    )
    assert (chain["share_same_seat_earlier_elimination"] or 0.0) > 0.70
    attr = attribute_first_divergence(
        control, treatment, leftover_rows=leftover, treatment_punch=[],
    )
    assert attr["reconciliation_ok"] is True
    assert (attr["share_same_pairing_outcome_flip"] or 0.0) > 0.70
    examples = attr["examples"]["same_pairing_outcome_flip"]
    assert examples
    rec = examples[0]
    assert rec["causal_seat"] == 2
    assert rec["first_divergence_turn"] == 7
    assert rec["same_fight_pairing"] is True
    assert rec["control_outcome"] == "win"
    assert rec["treatment_outcome"] == "loss"
    assert rec["control_pre_hp"] == rec["treatment_pre_hp"] == 20
    assert rec["treatment_applied"] == 8
    assert rec["treatment_decisive"]["survivor_count"] == 2
    decision = diagnose_phase_3m({"attribution": attr})
    assert decision["primary_finding"] == "same_pairing_outcome_dominates"


def test_attribute_routes_same_outcome_damage():
    leftover, control, treatment = _class1_base(origin="damage")
    chain = attribute_elimination_chain(
        control, treatment, leftover_rows=leftover, treatment_punch=[],
    )
    assert (chain["share_same_seat_earlier_elimination"] or 0.0) > 0.70
    attr = attribute_first_divergence(
        control, treatment, leftover_rows=leftover, treatment_punch=[],
    )
    assert attr["reconciliation_ok"] is True
    assert (attr["share_same_outcome_damage"] or 0.0) > 0.70
    examples = attr["examples"]["same_outcome_damage"]
    assert examples
    rec = examples[0]
    assert rec["causal_seat"] == 2
    assert rec["first_divergence_turn"] == 7
    assert rec["same_fight_pairing"] is True
    assert rec["control_outcome"] == rec["treatment_outcome"] == "loss"
    assert rec["control_applied"] == 4
    assert rec["treatment_applied"] == 8
    assert rec["control_pre_hp"] == rec["treatment_pre_hp"] == 20
    assert rec["treatment_decisive"]["abstract_pool_raw"] == 12.0
    assert rec["treatment_decisive"]["total_combat_raw"] == 62.0
    decision = diagnose_phase_3m({"attribution": attr})
    assert decision["primary_finding"] == "same_outcome_damage_dominates"


def test_find_first_divergence_records_hp_flow():
    leftover, control, treatment = _class1_base(origin="damage")
    chain = attribute_elimination_chain(
        control, treatment, leftover_rows=leftover, treatment_punch=[],
    )
    rec = chain["examples"]["same_seat_earlier_elimination"][0]
    from ml.elimination_chain_diagnostic import _hp_row_index
    from ml.matchmaking_divergence_diagnostic import _index_decisions
    from ml.pairing_who_wins_diagnostic import _index_seat_fights
    event = find_first_hp_divergence(
        rec,
        _index_seat_fights(control["fights"]),
        _index_seat_fights(treatment["fights"]),
        _hp_row_index(control["hp_rows"]),
        _hp_row_index(treatment["hp_rows"]),
        _index_decisions(control["pairing_decisions"]),
        _index_decisions(treatment["pairing_decisions"]),
    )
    assert event["found_event"] is True
    assert event["hp_flow_ok"] is True
    assert event["control_post_hp"] == event["control_pre_hp"] - event["control_applied"]
    assert event["treatment_post_hp"] == event["treatment_pre_hp"] - event["treatment_applied"]


def test_hp_flow_and_elimination_reconciliation():
    fights = [
        _fight(14200, 9, 0, 1, pre_a=30, pre_b=8, applied=8, post_a=30, post_b=0),
        _fight(14200, 10, 0, 2, pre_a=30, pre_b=20, applied=4, post_a=30, post_b=16),
    ]
    rec = reconcile_hp_flow(fights)
    assert rec["ok"] is True
    assert rec["n_fights"] == 2
    elims = [
        {"seed": 14200, "seat": 1, "turn": 9, "hp": 0, "survived": False},
        {"seed": 14200, "seat": 0, "turn": 14, "hp": 12, "survived": True},
    ]
    for s in range(2, 8):
        elims.append({"seed": 14200, "seat": s, "turn": 14, "hp": 10, "survived": True})
    elim = reconcile_eliminations(fights, elims, n_lobbies=1)
    assert elim["census_ok"] is True
    assert elim["n_combat_eliminations"] == 1
    assert elim["link_ok"] is True


def test_3g_mixture_lock_on_weight_shift_still_holds():
    def _row(turn, tier, carry):
        return {
            "turn": turn, "winner_start_tier": tier, "tier": tier,
            "carry": float(carry), "n_alive": 6, "opp_n_alive": 6,
            "opp_alive": True, "winner_tavern_tier": 2,
            "alive_bin": "alive_6_plus", "role_bin": "winner_tavern_low",
            "damaging": True, "opp_carry_attack_pool": float(carry),
            "attack_pool_recruit_start": float(carry),
        }

    control = (
        [_row(7, 1, 100.0) for _ in range(20)]
        + [_row(12, 5, 900.0) for _ in range(80)]
    )
    treat = (
        [_row(7, 1, 100.0) for _ in range(80)]
        + [_row(12, 5, 900.0) for _ in range(20)]
    )
    decomp = decompose_punch_selection(control, treat, unpaired_punch=-196.0)
    assert (decomp["share_mixture_turn_winner_tier"] or 0.0) > 0.70
    assert abs(decomp["share_within_cell_opponent_carry"] or 0.0) < 0.05


def test_compare_first_divergence_reproduces_3g_on_synthetic():
    def _seat_turn(seed, seat, turn, carry, *, t1t3=3, alive=True):
        return {
            "seed": seed, "seat": seat, "turn": turn, "arm": "obs",
            "attack_pool_recruit_start": carry,
            "scale_add_attack": 0.0,
            "alive_at_recruit": alive,
            "alive_at_combat": alive,
            "tier_at_recruit": 3,
            "board_size_post_scale": 7,
            "mean_tier_recruit_start": 2.0,
            "n_alive": 6,
            "flow_ok": True,
            "flow_residual": 0.0,
            "t1t3_count_recruit_start": t1t3,
            "t1t3_count_combat_start": t1t3,
            "t1t3_share_recruit_start": t1t3 / 7.0,
            "gold_recruit_start": 5,
            "shop_tiers_recruit_start": [3, 4],
            "shop_t1t3_offers_recruit_start": 1,
            "n_replacements": 0,
            "n_sells": 0,
            "tier_hist_recruit_start": {"1": 1, "2": 1, "3": 1, "4": 2, "5": 2, "6": 0},
        }

    def _fight_row(seed, turn, winner, loser, *, start_tier=1, carry=100.0):
        rec = _fight(seed, turn, winner, loser, start_tier=start_tier)
        rec["start_minions"][0]["opp_carry_attack_pool"] = carry
        rec["start_minions"][0]["opp_attack_pool_recruit_start"] = carry
        return rec

    control = {
        "turn_rows": [
            _seat_turn(14200, 0, t, 80.0 if t >= 10 else 40.0, t1t3=3)
            for t in INSTRUMENT_TURNS
        ] + [
            _seat_turn(14200, 1, t, 800.0 if t >= 10 else 100.0, t1t3=2)
            for t in INSTRUMENT_TURNS
        ],
        "fights": [_fight_row(14200, 12, 0, 1, start_tier=1, carry=800.0)],
        "pairing_decisions": [
            _decision(14200, 12, [0, 1, 2, 3], [(0, 1), (2, 3)])
        ],
        "last_t1t3_losses": [],
        "t1t3_events": [],
        "replacement_events": [],
        "board_snapshots": [],
        "game_lengths": [14],
        "eliminations": [],
        "hp_rows": [],
        "n_lobbies": 1,
    }
    treatment = {
        "turn_rows": [
            _seat_turn(14200, 0, t, 40.0 if t >= 10 else 40.0, t1t3=2)
            for t in INSTRUMENT_TURNS
        ] + [
            _seat_turn(14200, 1, t, 50.0 if t >= 10 else 100.0, t1t3=2)
            for t in INSTRUMENT_TURNS
        ],
        "fights": [_fight_row(14200, 12, 1, 0, start_tier=5, carry=50.0)],
        "pairing_decisions": [
            _decision(14200, 12, [0, 1, 2, 3], [(0, 1), (2, 3)])
        ],
        "last_t1t3_losses": [],
        "t1t3_events": [],
        "replacement_events": [],
        "board_snapshots": [],
        "game_lengths": [14],
        "eliminations": [],
        "hp_rows": [],
        "n_lobbies": 1,
    }
    timing = compare_elimination(control, treatment)
    chain = compare_chain(control, treatment, timing=timing)
    cmp = compare_first_divergence(control, treatment, timing=timing, chain=chain)
    decomp = cmp["decomposition_3g"]
    assert decomp["n_control"] == 1
    assert decomp["n_treatment"] == 1
    assert cmp["reconciliation"]["history_link_control"]["n_ok"] == 1
    late = cmp["pairing_3i"]
    assert late["n_leftover"] == 1
    assert late["reconciliation_ok"] is True
    assert late["attributed"]["outcome_flip"] == 1.0
    assert cmp["attribution"]["n_pairing_schedule"] == 0
    assert cmp["attribution"]["n_third_party"] == 0
    assert cmp["attribution"]["n_same_seat_earlier"] == 0
    assert cmp["attribution"]["reconciliation_ok"] is True
    ret = compare_retention(control, treatment)
    assert ret["attribution"]["leftover"] == 1.0
    pair = compare_pairing(control, treatment)
    assert pair["attribution"]["n_leftover"] == 1
    assert attribute_leftover_pairing(
        control, treatment, leftover_rows=[_punch(14200, 0, 12)],
        treatment_punch=[],
    )["reconciliation_ok"] is True
