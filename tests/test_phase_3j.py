"""Phase 3J matchmaking divergence — observational locks."""

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
from ml.phase_3j_prereg import (
    BYE_TOKEN,
    CANDIDATE_CHOICE_IDENTITY,
    EARLY_TURNS,
    FEATURE_TOGGLE,
    FEATURE_TOGGLE_DEFAULT,
    FORBIDDEN_RANGES,
    FROZEN_ALPHA,
    GHOST_TOKEN,
    HISTORY_LINK_IDENTITY,
    HOLD_PRS,
    IMPACT_ATTACK_IDENTITY,
    INSTRUMENT_TURNS,
    LATE_TURNS,
    LEFTOVER_RECONCILE_IDENTITY,
    LINEAGE_IDENTITY,
    LOW_TIERS,
    LOW_WINNER_START_TIERS,
    MATCHMAKING_COMPONENTS,
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
    PHASE_3J_LOBBIES,
    PHASE_3J_SEED,
    POOL_FLOW_IDENTITY,
    REUSED_SEED_HI,
    REUSED_SEED_LO,
    SHARE_DOMINANT,
    VERY_LATE_TURNS,
    WEIGHT_RECONCILIATION_IDENTITY,
    assert_seed_range_allowed,
    classify_matchmaking_gap,
    diagnose_phase_3j,
    share_of_schedule,
)
from ml.board_retention_diagnostic import compare_retention
from ml.carry_divergence_diagnostic import (
    build_seat_trajectories,
    reconcile_history_links,
)
from ml.matchmaking_divergence_diagnostic import (
    MatchmakingDivergenceTracer,
    attribute_matchmaking,
    choice_in_candidates,
    chosen_opponent_for_seat,
    classify_matchmaking_gap as _cls_mm_imported,
    compare_matchmaking,
    ghost_bye_eligibility,
    history_constrained_candidates,
    iter_pairing_schedule_rows,
    legal_candidates_for_seat,
    rng_state_meta,
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


def test_methodology_is_3j_v1_default_off():
    assert METHODOLOGY_VERSION == "3j_v1"
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
    assert MATCHMAKING_COMPONENTS == (
        "eligibility",
        "history_legal",
        "rng_order",
        "unreconciled",
    )
    assert GHOST_TOKEN == "ghost"
    assert BYE_TOKEN == "bye"
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
    assert PHASE_3J_SEED == 14200
    assert PHASE_3J_LOBBIES == 500
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


def test_hold_stack_includes_3i_and_frozen_alpha():
    assert HOLD_PRS == (
        29, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
        50, 51, 52, 53, 54, 55,
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
    d = diagnose_phase_3j()
    assert d["no_merge"] is True
    assert d["no_hero_damage_retune"] is True
    assert d["no_behavior_change"] is True
    assert d["no_2q_rewrite"] is True
    assert d["no_scaling_constant_change"] is True
    assert d["keep_hold_prs"][-1] == 55
    assert d["history_filters_applied"] is False
    assert d["candidate_choice_identity"] == CANDIDATE_CHOICE_IDENTITY
    assert d["matchmaking_reconcile_identity"] == MATCHMAKING_RECONCILE_IDENTITY
    assert _cls_mm_imported is classify_matchmaking_gap


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


def test_legal_candidates_and_choice_reconciliation():
    even = legal_candidates_for_seat(
        0, alive_seats=[0, 1, 2, 3], ghost_eligible=False, bye_eligible=False,
    )
    assert even == [1, 2, 3]
    odd_ghost = legal_candidates_for_seat(
        0, alive_seats=[0, 1, 2], ghost_eligible=True, bye_eligible=False,
    )
    assert odd_ghost == [1, 2, GHOST_TOKEN]
    odd_bye = legal_candidates_for_seat(
        2, alive_seats=[0, 1, 2], ghost_eligible=False, bye_eligible=True,
    )
    assert odd_bye == [0, 1, BYE_TOKEN]
    elig = ghost_bye_eligibility([0, 1, 2], [4])
    assert elig["odd_alive"] is True
    assert elig["ghost_eligible"] is True
    assert elig["bye_eligible"] is False
    assert elig["history_filters_applied"] is False
    chosen = chosen_opponent_for_seat(
        0, [(0, 1), (2, None)], ghost_eligible=True, bye_eligible=False,
    )
    assert chosen == 1
    ghosted = chosen_opponent_for_seat(
        2, [(0, 1), (2, None)], ghost_eligible=True, bye_eligible=False,
    )
    assert ghosted == GHOST_TOKEN
    assert choice_in_candidates(1, even) is True
    assert choice_in_candidates(GHOST_TOKEN, odd_ghost) is True
    assert choice_in_candidates(4, even) is False
    hist = history_constrained_candidates(
        0, alive_seats=[0, 1, 2, 3], prior_opponents=[1],
        ghost_eligible=False, bye_eligible=False,
    )
    assert hist == [2, 3]


def test_share_of_schedule_and_classify():
    assert abs(share_of_schedule(70.0, denom=100.0) - 0.70) < 1e-9
    assert share_of_schedule(None, denom=100.0) is None
    assert share_of_schedule(1.0, denom=0.0) is None
    base = dict(
        control_present=True, treatment_present=True,
        leftover_alive_control=True, leftover_alive_treatment=True,
        choice_in_candidates_control=True, choice_in_candidates_treatment=True,
        alive_sets_equal=True, ghost_bye_eligible_equal=True,
        legal_candidates_equal=True, chosen_equal=False,
    )
    assert classify_matchmaking_gap(**base) == "rng_order"
    assert classify_matchmaking_gap(**{**base, "alive_sets_equal": False}) == (
        "eligibility"
    )
    assert classify_matchmaking_gap(
        **{**base, "ghost_bye_eligible_equal": False}
    ) == "eligibility"
    assert classify_matchmaking_gap(
        **{**base, "legal_candidates_equal": False}
    ) == "history_legal"
    assert classify_matchmaking_gap(**{**base, "chosen_equal": True}) == (
        "unreconciled"
    )
    assert classify_matchmaking_gap(
        **{**base, "control_present": False}
    ) == "unreconciled"
    assert classify_matchmaking_gap(
        **{**base, "choice_in_candidates_control": False}
    ) == "unreconciled"


def test_diagnose_routes():
    elig = diagnose_phase_3j({
        "attribution": {
            "share_eligibility": 0.80,
            "share_history_legal": 0.10,
            "share_rng_order": 0.08,
            "share_unreconciled": 0.02,
        }
    })
    assert elig["primary_finding"] == "eligibility_dominates"
    assert "elimination timing" in elig["recommended_next_step"]
    hist = diagnose_phase_3j({
        "attribution": {
            "share_eligibility": 0.10,
            "share_history_legal": 0.75,
            "share_rng_order": 0.10,
            "share_unreconciled": 0.05,
        }
    })
    assert hist["primary_finding"] == "history_legal_dominates"
    assert "no-repeat" in hist["recommended_next_step"]
    rng = diagnose_phase_3j({
        "attribution": {
            "share_eligibility": 0.10,
            "share_history_legal": 0.05,
            "share_rng_order": 0.80,
            "share_unreconciled": 0.05,
        }
    })
    assert rng["primary_finding"] == "rng_order_dominates"
    assert "RNG coupling" in rng["recommended_next_step"]
    mixed = diagnose_phase_3j({
        "attribution": {
            "share_eligibility": 0.40,
            "share_history_legal": 0.25,
            "share_rng_order": 0.20,
            "share_unreconciled": 0.15,
        }
    })
    assert mixed["primary_finding"] == "mixed_route_to_larger"
    leftover_d = diagnose_phase_3j({
        "attribution": {
            "share_eligibility": 0.10,
            "share_history_legal": 0.08,
            "share_rng_order": 0.05,
            "share_unreconciled": 0.77,
        }
    })
    assert leftover_d["primary_finding"] == "ranked_residual_needs_next_observable"
    smoke = diagnose_phase_3j(mixed, non_evaluative=True)
    assert smoke["primary_finding"] == "measurement_smoke_non_evaluative"
    assert smoke["evaluative"] is False


def _play(seed, with_hook):
    env = BGEnv(seed=seed)
    tracer = None
    if with_hook:
        tracer = MatchmakingDivergenceTracer(0, seed, "obs")
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


def test_combat_and_pairing_hooks_are_observational_same_seed():
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
    late = [d for d in tracer.pairing_decisions if int(d["turn"]) in LATE_TURNS]
    assert late
    for rec in late:
        assert rec["history_filters_applied"] is False
        assert rec["rng_state_digest_pre"]
        assert rec["rng_state_head_pre"] or rec["rng_index_pre"] is not None
        assert rec["shuffled_order"]
        assert rec["alive_seats"]
        for seat, view in rec["per_seat"].items():
            assert view["choice_in_candidates"] is True
            assert view["chosen"] in view["legal_candidates"] or (
                view["chosen"] in (GHOST_TOKEN, BYE_TOKEN)
                and view["chosen"] in view["legal_candidates"]
            )
    traj = build_seat_trajectories(tracer.turn_rows, tracer.fights)
    assert traj
    hist = reconcile_history_links(tracer.fights, tracer.turn_rows)
    if hist["n_punch_rows"] > 0:
        assert hist["n_carry_mismatch"] == 0
        assert hist["n_missing_turn_row"] == 0
        assert hist["p_ok"] == 1.0
    rows = collect_punch_sample_rows(tracer.fights)
    assert rows
    stamped = [f for f in tracer.fights if f.get("pairing")]
    assert stamped
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
    meta = rng_state_meta(r1.getstate())
    assert len(meta["rng_state_digest"]) == 64
    assert meta["rng_state_head"] or meta["rng_index"] is not None


def test_pairing_hook_does_not_consume_extra_rng():
    seen = []

    def _hook(env, rec):
        seen.append(rec)
        # Hook must not touch env.rng.
        assert rec["rng_state_post"] == env.rng.getstate()

    env = BGEnv(seed=14201)
    env.pairing_audit_hook = _hook
    recs = env.play_scripted([greedy_policy] * env.n_players)
    assert seen
    assert recs
    plain = BGEnv(seed=14201)
    plain.play_scripted([greedy_policy] * plain.n_players)
    assert [p.placement for p in env.players] == [p.placement for p in plain.players]
    assert [p.hp for p in env.players] == [p.hp for p in plain.players]
    assert env.rng.getstate() == plain.rng.getstate()


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
           applied=4, t1t3_a=3, t1t3_b=2):
    sa, sb = winner, loser
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
        "pre_hp_a": 30, "pre_hp_b": 30, "post_hp_a": 30, "post_hp_b": 26,
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
                "pre_fight_hp": 30, "board_size": 6, "alive": True,
            },
            "b": {
                "t1t3_count": t1t3_b, "t1t3_share": 0.4, "tavern_tier": 5,
                "recruit_raw": 50.0, "abstract_pool_raw": 12.0, "combat_raw": 62.0,
                "pre_fight_hp": 30, "board_size": 6, "alive": True,
            },
            "alive_next": {str(sa): True, str(sb): True} if kind == "live" else {str(sa): True},
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


def test_attribute_routes_eligibility():
    leftover = [_punch(14200, 0, t, loser=1) for t in LATE_TURNS] * 4
    control = {
        "turn_rows": [_turn(14200, 0, t, 3) for t in LATE_TURNS],
        "fights": [_fight(14200, t, 0, 1) for t in LATE_TURNS],
        "pairing_decisions": [
            _decision(14200, t, [0, 1, 2, 3], [(0, 1), (2, 3)])
            for t in LATE_TURNS
        ],
        "last_t1t3_losses": [],
        "t1t3_events": [],
    }
    treatment = {
        "turn_rows": [_turn(14200, 0, t, 2) for t in LATE_TURNS],
        "fights": [_fight(14200, t, 0, 2) for t in LATE_TURNS],
        "pairing_decisions": [
            _decision(14200, t, [0, 2, 3], [(0, 2), (3, None)], dead=[1, 4, 5, 6, 7])
            for t in LATE_TURNS
        ],
        "last_t1t3_losses": [],
        "t1t3_events": [],
    }
    attr = attribute_matchmaking(
        control, treatment, leftover_rows=leftover, treatment_punch=[],
    )
    assert attr["n_pairing_schedule"] == len(leftover)
    assert attr["reconciliation_ok"] is True
    assert attr["candidate_choice_ok"] is True
    assert (attr["share_eligibility"] or 0.0) > 0.70
    decision = diagnose_phase_3j({"attribution": attr})
    assert decision["primary_finding"] == "eligibility_dominates"


def test_attribute_routes_rng_order():
    leftover = [_punch(14200, 0, t, loser=1) for t in LATE_TURNS] * 4
    control = {
        "turn_rows": [_turn(14200, 0, t, 3) for t in LATE_TURNS],
        "fights": [_fight(14200, t, 0, 1) for t in LATE_TURNS],
        "pairing_decisions": [
            _decision(14200, t, [0, 1, 2, 3], [(0, 1), (2, 3)], digest="c")
            for t in LATE_TURNS
        ],
        "last_t1t3_losses": [],
        "t1t3_events": [],
    }
    treatment = {
        "turn_rows": [_turn(14200, 0, t, 2) for t in LATE_TURNS],
        "fights": [_fight(14200, t, 0, 2) for t in LATE_TURNS],
        "pairing_decisions": [
            _decision(14200, t, [0, 1, 2, 3], [(0, 2), (1, 3)], digest="t")
            for t in LATE_TURNS
        ],
        "last_t1t3_losses": [],
        "t1t3_events": [],
    }
    attr = attribute_matchmaking(
        control, treatment, leftover_rows=leftover, treatment_punch=[],
    )
    assert attr["reconciliation_ok"] is True
    assert (attr["share_rng_order"] or 0.0) > 0.70
    assert diagnose_phase_3j({"attribution": attr})["primary_finding"] == (
        "rng_order_dominates"
    )


def test_attribute_routes_history_legal_and_unreconciled():
    leftover = [_punch(14200, 0, 12, loser=1) for _ in range(10)]
    # Same alive / eligibility, but legal sets differ (synthetic history filter).
    c_dec = _decision(14200, 12, [0, 1, 2, 3], [(0, 1), (2, 3)])
    t_dec = _decision(14200, 12, [0, 1, 2, 3], [(0, 2), (1, 3)])
    t_dec["per_seat"]["0"]["legal_candidates"] = [2, 3]
    t_dec["per_seat"]["0"]["chosen"] = 2
    t_dec["per_seat"]["0"]["choice_in_candidates"] = True
    control = {
        "turn_rows": [_turn(14200, 0, 12, 5)],
        "fights": [_fight(14200, 12, 0, 1)],
        "pairing_decisions": [c_dec],
        "last_t1t3_losses": [],
        "t1t3_events": [],
    }
    treatment = {
        "turn_rows": [_turn(14200, 0, 12, 2)],
        "fights": [_fight(14200, 12, 0, 2)],
        "pairing_decisions": [t_dec],
        "last_t1t3_losses": [],
        "t1t3_events": [],
    }
    attr = attribute_matchmaking(
        control, treatment, leftover_rows=leftover, treatment_punch=[],
    )
    assert attr["reconciliation_ok"] is True
    assert (attr["share_history_legal"] or 0.0) > 0.70
    assert diagnose_phase_3j({"attribution": attr})["primary_finding"] == (
        "history_legal_dominates"
    )

    missing = attribute_matchmaking(
        {"fights": [_fight(14200, 12, 0, 1)], "pairing_decisions": [],
         "turn_rows": [], "last_t1t3_losses": [], "t1t3_events": []},
        {"fights": [_fight(14200, 12, 0, 2)], "pairing_decisions": [],
         "turn_rows": [], "last_t1t3_losses": [], "t1t3_events": []},
        leftover_rows=leftover, treatment_punch=[],
    )
    assert (missing["share_unreconciled"] or 0.0) > 0.70


def test_iter_pairing_schedule_matches_3i_classifier():
    leftover = [_punch(14200, 0, t, loser=1) for t in LATE_TURNS]
    control = {
        "turn_rows": [_turn(14200, 0, t, 3) for t in LATE_TURNS],
        "fights": [_fight(14200, t, 0, 1) for t in LATE_TURNS],
        "last_t1t3_losses": [],
        "t1t3_events": [],
    }
    treatment = {
        "turn_rows": [_turn(14200, 0, t, 2) for t in LATE_TURNS],
        "fights": [_fight(14200, t, 0, 3) for t in LATE_TURNS],
        "last_t1t3_losses": [],
        "t1t3_events": [],
    }
    attr = attribute_leftover_pairing(
        control, treatment, leftover_rows=leftover, treatment_punch=[],
    )
    schedule = iter_pairing_schedule_rows(
        leftover, control, treatment, treatment_punch=[],
    )
    assert len(schedule) == int(attr["attributed"]["pairing_schedule"])
    assert all(r["class"] == "pairing_schedule" for r in schedule)


def test_compare_matchmaking_reproduces_3g_on_synthetic():
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
    }
    cmp = compare_matchmaking(control, treatment)
    decomp = cmp["decomposition_3g"]
    assert decomp["n_control"] == 1
    assert decomp["n_treatment"] == 1
    assert cmp["reconciliation"]["history_link_control"]["n_ok"] == 1
    late = cmp["pairing_3i"]
    assert late["n_leftover"] == 1
    assert late["reconciliation_ok"] is True
    assert late["attributed"]["outcome_flip"] == 1.0
    # Same pairing is not in the 3I pairing-schedule leftover.
    assert cmp["attribution"]["n_pairing_schedule"] == 0
    assert cmp["attribution"]["reconciliation_ok"] is True
    ret = compare_retention(control, treatment)
    assert ret["attribution"]["leftover"] == 1.0
    pair = compare_pairing(control, treatment)
    assert pair["attribution"]["n_leftover"] == 1


def test_3g_mixture_lock_on_weight_shift_still_holds():
    """3J must still recover the 3G mixture-vs-within split on synthetic rows."""
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
