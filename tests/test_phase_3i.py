"""Phase 3I T1–T3 pairing / who-wins — observational locks."""

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
from ml.phase_3i_prereg import (
    EARLY_TURNS,
    FEATURE_TOGGLE,
    FEATURE_TOGGLE_DEFAULT,
    FORBIDDEN_RANGES,
    FROZEN_ALPHA,
    HISTORY_LINK_IDENTITY,
    HOLD_PRS,
    IMPACT_ATTACK_IDENTITY,
    INSTRUMENT_TURNS,
    LATE_TURNS,
    LEFTOVER_RECONCILE_IDENTITY,
    LINEAGE_IDENTITY,
    LOW_TIERS,
    LOW_WINNER_START_TIERS,
    METHODOLOGY_VERSION,
    PAIRED_SEAT_IDENTITY,
    PAIRING_COMPONENTS,
    PAIRING_IDENTITY,
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
    PHASE_3I_LOBBIES,
    PHASE_3I_SEED,
    POOL_FLOW_IDENTITY,
    REUSED_SEED_HI,
    REUSED_SEED_LO,
    SHARE_DOMINANT,
    VERY_LATE_TURNS,
    WEIGHT_RECONCILIATION_IDENTITY,
    assert_seed_range_allowed,
    classify_pairing_gap,
    diagnose_phase_3i,
    share_of_leftover,
)
from ml.board_retention_diagnostic import (
    collect_3h_leftover_rows,
    compare_retention,
)
from ml.carry_divergence_diagnostic import (
    build_seat_trajectories,
    reconcile_history_links,
)
from ml.pairing_who_wins_diagnostic import (
    PairingWhoWinsTracer,
    attribute_leftover_pairing,
    compare_pairing,
    same_pairing,
    treatment_won,
)
from ml.punch_selection_diagnostic import (
    collect_punch_sample_rows,
    decompose_punch_selection,
)
from ml.survivor_tier_damage_diagnostic import rules_faithful_hero_damage


def test_methodology_is_3i_v1_default_off():
    assert METHODOLOGY_VERSION == "3i_v1"
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
    assert PAIRING_COMPONENTS == (
        "pairing_schedule",
        "outcome_flip",
        "survivor_substitution",
        "residual",
    )
    assert LINEAGE_IDENTITY == (
        "t1t3_end = t1t3_start + t1t3_added - t1t3_removed"
    )
    assert "paired (seed, seat)" in PAIRED_SEAT_IDENTITY
    assert "opponent_seat" in PAIRING_IDENTITY
    assert "pairing_schedule + outcome_flip" in LEFTOVER_RECONCILE_IDENTITY
    assert HISTORY_LINK_IDENTITY == (
        "punch.opp_carry = seat_history[turn].attack_pool_recruit_start"
    )
    assert "sum_k n_arm(k) = N_arm" in WEIGHT_RECONCILIATION_IDENTITY
    assert POOL_FLOW_IDENTITY == (
        "post = pre + add - represented_loss_or_transfer"
    )
    assert IMPACT_ATTACK_IDENTITY == (
        "impact_attack = start_recruit + start_pool_share + combat_delta"
    )


def test_reuses_2s_dev_and_forbids_confirm():
    assert PHASE_3I_SEED == 14200
    assert PHASE_3I_LOBBIES == 500
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


def test_hold_stack_includes_3h_and_frozen_alpha():
    assert HOLD_PRS == (
        29, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
        50, 51, 52, 53, 54,
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
    d = diagnose_phase_3i()
    assert d["no_merge"] is True
    assert d["no_hero_damage_retune"] is True
    assert d["no_behavior_change"] is True
    assert d["no_2q_rewrite"] is True
    assert d["no_scaling_constant_change"] is True
    assert d["keep_hold_prs"][-1] == 54
    assert d["pairing_identity"] == PAIRING_IDENTITY
    assert d["leftover_reconcile_identity"] == LEFTOVER_RECONCILE_IDENTITY


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


def test_share_of_leftover_and_classify():
    assert abs(share_of_leftover(70.0, denom=100.0) - 0.70) < 1e-9
    assert share_of_leftover(None, denom=100.0) is None
    assert share_of_leftover(1.0, denom=0.0) is None
    assert classify_pairing_gap(same_pairing=False) == "pairing_schedule"
    assert classify_pairing_gap(
        same_pairing=True, treatment_wins=False, treatment_tie_or_loss=True,
    ) == "outcome_flip"
    assert classify_pairing_gap(
        same_pairing=True, treatment_wins=True,
        treatment_t1t3_punches=0, uncovered=True,
    ) == "survivor_substitution"
    assert classify_pairing_gap(
        same_pairing=True, treatment_wins=True,
        treatment_t1t3_punches=3, uncovered=False,
    ) == "residual"


def test_same_pairing_and_treatment_won_helpers():
    live = {"kind": "live", "seat_a": 0, "seat_b": 1, "winner_seat": 0}
    other = {"kind": "live", "seat_a": 0, "seat_b": 2, "winner_seat": 2}
    ghost = {"kind": "ghost", "seat_a": 0, "seat_b": None, "winner_seat": 0}
    assert same_pairing(live, dict(live), 0) is True
    assert same_pairing(live, other, 0) is False
    assert same_pairing(live, ghost, 0) is False
    assert same_pairing(live, None, 0) is False
    assert treatment_won(live, 0) is True
    assert treatment_won(live, 1) is False
    assert treatment_won({"winner_seat": None}, 0) is False


def _play(seed, with_hook):
    env = BGEnv(seed=seed)
    tracer = None
    if with_hook:
        tracer = PairingWhoWinsTracer(0, seed, "obs")
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


def test_combat_and_scaling_hooks_are_observational_same_seed():
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
    for rec in stamped:
        pairing = rec["pairing"]
        assert "kind" in pairing
        assert "a" in pairing or pairing.get("seat_a") is None
        assert "alive_next" in pairing
        assert "low_tier_attacked" in pairing
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
           applied=4, t1t3_a=3, t1t3_b=2, winner_punches=None):
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
    rec = {
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
    return rec


def test_attribute_routes_pairing_schedule():
    leftover = [_punch(14200, 0, t, loser=1) for t in LATE_TURNS] * 4
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
    assert attr["n_leftover"] == len(leftover)
    assert attr["reconciliation_ok"] is True
    assert (attr["share_pairing_schedule"] or 0.0) > 0.70
    decision = diagnose_phase_3i({"attribution": attr})
    assert decision["primary_finding"] == "opponent_schedule_dominates"
    assert "pairing fidelity" in decision["recommended_next_step"]


def test_attribute_routes_outcome_flip():
    leftover = [_punch(14200, 0, t, loser=1) for t in LATE_TURNS] * 4
    control = {
        "turn_rows": [_turn(14200, 0, t, 3) for t in LATE_TURNS],
        "fights": [_fight(14200, t, 0, 1, raw=5) for t in LATE_TURNS],
        "last_t1t3_losses": [],
        "t1t3_events": [],
    }
    treatment = {
        "turn_rows": [_turn(14200, 0, t, 2) for t in LATE_TURNS],
        "fights": [_fight(14200, t, 1, 0, raw=4) for t in LATE_TURNS],
        "last_t1t3_losses": [],
        "t1t3_events": [],
    }
    # treatment fight: seat_a=1 wins, seat_b=0 loses. leftover seat is 0.
    # Reconstruct so leftover seat 0 faces the same opponent 1 and loses.
    treatment["fights"] = [
        _fight(14200, t, 1, 0, raw=4) for t in LATE_TURNS
    ]
    # _fight(winner=1, loser=0) sets seat_a=1, seat_b=0. Opponent of seat 0 is 1. Same pairing.
    attr = attribute_leftover_pairing(
        control, treatment, leftover_rows=leftover, treatment_punch=[],
    )
    assert attr["reconciliation_ok"] is True
    assert (attr["share_outcome_flip"] or 0.0) > 0.70
    assert diagnose_phase_3i({"attribution": attr})["primary_finding"] == (
        "same_pairing_outcome_flip_dominates"
    )


def test_attribute_routes_survivor_substitution_and_residual():
    leftover = [_punch(14200, 0, 12, loser=1) for _ in range(10)]
    control = {
        "turn_rows": [_turn(14200, 0, 12, 5)],
        "fights": [_fight(14200, 12, 0, 1, start_tier=1)],
        "last_t1t3_losses": [],
        "t1t3_events": [],
    }
    treatment = {
        "turn_rows": [_turn(14200, 0, 12, 2)],
        "fights": [_fight(14200, 12, 0, 1, start_tier=5)],
        "last_t1t3_losses": [],
        "t1t3_events": [],
    }
    # Treatment wins same pairing but produces 0 T1–T3 punches.
    attr = attribute_leftover_pairing(
        control, treatment, leftover_rows=leftover, treatment_punch=[],
    )
    assert attr["reconciliation_ok"] is True
    assert (attr["share_survivor_substitution"] or 0.0) > 0.70
    assert diagnose_phase_3i({"attribution": attr})["primary_finding"] == (
        "survivor_substitution_dominates"
    )

    t_punch = [_punch(14200, 0, 12, loser=1) for _ in range(10)]
    covered = attribute_leftover_pairing(
        control, treatment, leftover_rows=leftover, treatment_punch=t_punch,
    )
    assert (covered["share_residual"] or 0.0) > 0.70
    mixed = diagnose_phase_3i({
        "attribution": {
            "share_pairing_schedule": 0.40,
            "share_outcome_flip": 0.25,
            "share_survivor_substitution": 0.20,
            "share_residual": 0.15,
        }
    })
    assert mixed["primary_finding"] == "mixed_route_to_larger"
    leftover_d = diagnose_phase_3i({
        "attribution": {
            "share_pairing_schedule": 0.10,
            "share_outcome_flip": 0.08,
            "share_survivor_substitution": 0.05,
            "share_residual": 0.77,
        }
    })
    assert leftover_d["primary_finding"] == "ranked_residual_needs_next_observable"
    smoke = diagnose_phase_3i(mixed, non_evaluative=True)
    assert smoke["primary_finding"] == "measurement_smoke_non_evaluative"
    assert smoke["evaluative"] is False


def test_leftover_collect_matches_3h_still_fields():
    def _p(seed, seat, turn):
        return {
            "seed": seed, "turn": turn, "winner_seat": seat,
            "winner_start_tier": 1, "carry": 900.0,
        }

    c_rows = [_turn(14200, 0, t, 3) for t in LATE_TURNS]
    t_rows = [_turn(14200, 0, t, 2) for t in LATE_TURNS]
    punches = [_p(14200, 0, t) for t in LATE_TURNS]
    leftover = collect_3h_leftover_rows(
        {"turn_rows": c_rows, "last_t1t3_losses": []},
        {"turn_rows": t_rows, "last_t1t3_losses": []},
        control_punch=punches,
    )
    assert len(leftover) == len(punches)
    dead = collect_3h_leftover_rows(
        {"turn_rows": c_rows, "last_t1t3_losses": []},
        {"turn_rows": [], "last_t1t3_losses": [{
            "seed": 14200, "seat": 0, "class": "alive_elimination",
            "first_loss_turn": None,
        }]},
        control_punch=punches,
    )
    assert dead == []


def test_compare_pairing_reproduces_3g_on_synthetic():
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
        "last_t1t3_losses": [],
        "t1t3_events": [],
        "replacement_events": [],
        "board_snapshots": [],
        "game_lengths": [14],
    }
    # Same pairing, treatment seat 0 loses (outcome flip of the leftover punch).
    treatment["fights"] = [_fight_row(14200, 12, 1, 0, start_tier=5, carry=50.0)]
    cmp = compare_pairing(control, treatment)
    decomp = cmp["decomposition_3g"]
    assert decomp["n_control"] == 1
    assert decomp["n_treatment"] == 1
    assert cmp["reconciliation"]["history_link_control"]["n_ok"] == 1
    late = cmp["attribution"]
    assert late["n_leftover"] == 1
    assert late["reconciliation_ok"] is True
    assert late["attributed"]["outcome_flip"] == 1.0
    ret = compare_retention(control, treatment)
    assert ret["attribution"]["leftover"] == 1.0


def test_3g_mixture_lock_on_weight_shift_still_holds():
    """3I must still recover the 3G mixture-vs-within split on synthetic rows."""
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
