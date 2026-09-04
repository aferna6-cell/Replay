"""Phase 3H low-tier board-retention lifecycle — observational locks."""

import random

from hsbg_coach.bg_env import (
    A_BUY0,
    A_PLAY0,
    A_SELL0,
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
from ml.phase_3h_prereg import (
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
    LIFECYCLE_COMPONENTS,
    LINEAGE_IDENTITY,
    LOW_TIERS,
    LOW_WINNER_START_TIERS,
    METHODOLOGY_VERSION,
    PAIRED_SEAT_IDENTITY,
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
    PHASE_3H_LOBBIES,
    PHASE_3H_SEED,
    POOL_FLOW_IDENTITY,
    REUSED_SEED_HI,
    REUSED_SEED_LO,
    SHARE_DOMINANT,
    VERY_LATE_TURNS,
    WEIGHT_RECONCILIATION_IDENTITY,
    assert_seed_range_allowed,
    classify_t1t3_exit,
    diagnose_phase_3h,
    share_of_collapse,
)
from ml.board_retention_diagnostic import (
    BoardRetentionTracer,
    _t1t3_count,
    _t1t3_share,
    attribute_late_t1t3_collapse,
    compare_retention,
)
from ml.carry_divergence_diagnostic import (
    build_seat_trajectories,
    reconcile_history_links,
)
from ml.punch_selection_diagnostic import (
    collect_punch_sample_rows,
    decompose_punch_selection,
)
from ml.survivor_tier_damage_diagnostic import rules_faithful_hero_damage


def test_methodology_is_3h_v1_default_off():
    assert METHODOLOGY_VERSION == "3h_v1"
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
    assert LIFECYCLE_COMPONENTS == (
        "full_board_2q_replacement",
        "open_slot_fill",
        "tavern_offer_shift",
        "generated_transform_triple",
        "alive_elimination",
    )
    assert LINEAGE_IDENTITY == (
        "t1t3_end = t1t3_start + t1t3_added - t1t3_removed"
    )
    assert "paired (seed, seat)" in PAIRED_SEAT_IDENTITY
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
    assert PHASE_3H_SEED == 14200
    assert PHASE_3H_LOBBIES == 500
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


def test_hold_stack_includes_3g_and_frozen_alpha():
    assert HOLD_PRS == (
        29, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
        50, 51, 52, 53,
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
    d = diagnose_phase_3h()
    assert d["no_merge"] is True
    assert d["no_hero_damage_retune"] is True
    assert d["no_behavior_change"] is True
    assert d["no_2q_rewrite"] is True
    assert d["no_scaling_constant_change"] is True
    assert d["keep_hold_prs"][-1] == 53
    assert d["lineage_identity"] == LINEAGE_IDENTITY


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


def test_share_of_collapse_and_classify():
    assert abs(share_of_collapse(70.0, denom=100.0) - 0.70) < 1e-9
    assert share_of_collapse(None, denom=100.0) is None
    assert share_of_collapse(1.0, denom=0.0) is None
    assert classify_t1t3_exit(
        sold=True, board_full=True, replacement_completed=True,
        shop_t1t3_offers=2,
    ) == "full_board_2q_replacement"
    assert classify_t1t3_exit(
        sold=True, board_full=True, replacement_completed=True,
        shop_t1t3_offers=0,
    ) == "tavern_offer_shift"
    assert classify_t1t3_exit(
        open_slot_higher_tier_play=True,
    ) == "open_slot_fill"
    assert classify_t1t3_exit(triple=True) == "generated_transform_triple"
    assert classify_t1t3_exit(
        seat_died=True, had_t1t3_at_death=True,
    ) == "alive_elimination"


def test_t1t3_count_share_helpers():
    board = [
        {"name": "a", "tier": 1},
        {"name": "b", "tier": 5},
        {"name": "c", "tier": 2},
    ]
    assert _t1t3_count(board) == 2
    assert abs((_t1t3_share(board) or 0.0) - (2 / 3)) < 1e-9
    assert _t1t3_count([]) == 0
    assert _t1t3_share([]) is None


def _play(seed, with_hook):
    env = BGEnv(seed=seed)
    tracer = None
    if with_hook:
        tracer = BoardRetentionTracer(0, seed, "obs")
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
    for r in rows:
        assert r["turn"] in INSTRUMENT_TURNS
        assert r["winner_start_tier"] >= 1
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
    for row in tracer.turn_rows:
        start = row.get("t1t3_count_recruit_start")
        if start is None:
            continue
        assert int(start) >= 0
        assert "gold_recruit_start" in row


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


def test_tracer_records_replacement_context_on_scripted_sell_buy_play():
    """Drive a full-board T1 sell → T5 buy → play and record lineage fields."""
    tracer = BoardRetentionTracer(0, 14200, "obs")
    t1 = EnvMinion("id-t1", "Tiny", 1, 2, 2, [], [])
    t5 = EnvMinion("id-t5", "Big", 5, 8, 8, [], [])
    board_views = [t1.view() for _ in range(7)]
    shop_views = [t5.view()]
    hand_after_buy = [t5.view()]
    player = type("P", (), {})()
    player.board = [EnvMinion("id-t1", "Tiny", 1, 2, 2, [], []) for _ in range(6)]
    player.board.append(EnvMinion("id-t5", "Big", 5, 8, 8, [], []))
    player.hand = []
    player.shop = []
    player.gold = 3
    player.tier = 5
    player.alive = True

    obs_sell = {
        "board": board_views,
        "shop": shop_views,
        "hand": [],
        "gold": 5,
        "tavern_tier": 5,
    }
    tracer.before_action(0, 10, 0, obs_sell, [])
    tracer.after_action(0, 10, 0, A_SELL0, False, player=player)
    assert 0 in tracer._pending
    assert tracer._pending[0].get("sold_view") is not None

    obs_buy = {
        "board": board_views[:6],
        "shop": shop_views,
        "hand": [],
        "gold": 5,
        "tavern_tier": 5,
    }
    # Parent sell already popped pending if we also ended; keep it for buy.
    if 0 not in tracer._pending:
        tracer._pending[0] = {
            "sold_view": board_views[0],
            "candidate": None,
            "source": None,
            "sold_name": "Tiny",
            "pre_attack_pool": 0.0,
            "pre_stats_pool": 0.0,
            "sold_attack_pool": 0.0,
            "sold_stats_pool": 0.0,
        }
    tracer.before_action(0, 10, 0, obs_buy, [])
    tracer.after_action(0, 10, 0, A_BUY0, False, player=player)

    obs_play = {
        "board": board_views[:6],
        "shop": [],
        "hand": hand_after_buy,
        "gold": 2,
        "tavern_tier": 5,
    }
    if 0 not in tracer._pending:
        tracer._pending[0] = {
            "sold_view": board_views[0],
            "candidate": t5.view(),
            "source": "shop",
            "sold_name": "Tiny",
            "pre_attack_pool": 0.0,
            "pre_stats_pool": 0.0,
            "sold_attack_pool": 0.0,
            "sold_stats_pool": 0.0,
        }
    tracer.before_action(0, 10, 0, obs_play, [])
    tracer.after_action(0, 10, 0, A_PLAY0, False, player=player)
    assert tracer.t1t3_events
    last = tracer.t1t3_events[-1]
    assert last["incumbent_tier"] in (1, None) or last.get("sold")
    assert "shop_offer_tiers" in last
    assert "gold" in last
    assert "player_tavern_tier" in last
    assert "replacement_flag" in last


def test_attribute_routes_replacement_collapse():
    """Control late T1 punch rows vanish in treatment via 2Q replacement."""
    def _punch(seed, seat, turn, tier=1, carry=800.0):
        return {
            "seed": seed,
            "turn": turn,
            "winner_seat": seat,
            "winner_start_tier": tier,
            "carry": carry,
        }

    def _turn(seed, seat, turn, t1t3, *, alive=True, repl=0, shop=2):
        return {
            "seed": seed,
            "seat": seat,
            "turn": turn,
            "t1t3_count_combat_start": t1t3,
            "t1t3_count_recruit_start": t1t3,
            "alive_at_combat": alive,
            "alive_at_recruit": alive,
            "n_replacements": repl,
            "shop_t1t3_offers_recruit_start": shop,
        }

    control = {
        "turn_rows": [_turn(14200, 0, t, 3) for t in LATE_TURNS],
        "fights": [],
        "last_t1t3_losses": [],
        "t1t3_events": [],
    }
    treatment = {
        "turn_rows": [_turn(14200, 0, t, 0, repl=2, shop=2) for t in LATE_TURNS],
        "fights": [],
        "last_t1t3_losses": [{
            "seed": 14200, "seat": 0, "turn": 9,
            "class": "full_board_2q_replacement",
            "first_loss_turn": 9,
        }],
        "t1t3_events": [],
    }
    c_punch = [_punch(14200, 0, t) for t in LATE_TURNS] * 10
    t_punch = []
    attr = attribute_late_t1t3_collapse(
        control, treatment, control_punch=c_punch, treatment_punch=t_punch,
    )
    assert attr["collapse"] == float(len(c_punch))
    assert attr["reconciliation_ok"] is True
    assert (attr["share_full_board_2q_replacement"] or 0.0) > 0.70
    decision = diagnose_phase_3h({"attribution": attr})
    assert decision["primary_finding"] == "full_board_2q_replacement_dominates"
    assert "board reference" in decision["recommended_next_step"]


def test_attribute_routes_elimination_and_offer():
    def _punch(seed, seat, turn):
        return {
            "seed": seed, "turn": turn, "winner_seat": seat,
            "winner_start_tier": 1, "carry": 900.0,
        }

    c_rows = [{
        "seed": 14200, "seat": 1, "turn": t,
        "t1t3_count_combat_start": 2,
        "alive_at_combat": True, "alive_at_recruit": True,
        "n_replacements": 0, "shop_t1t3_offers_recruit_start": 0,
    } for t in LATE_TURNS]
    # Treatment seat is gone on late turns.
    t_raw = {
        "turn_rows": [],
        "fights": [],
        "last_t1t3_losses": [{
            "seed": 14200, "seat": 1, "class": "alive_elimination",
            "first_loss_turn": None, "had_t1t3_at_death": True,
        }],
        "t1t3_events": [],
    }
    c_raw = {"turn_rows": c_rows, "fights": [], "last_t1t3_losses": [], "t1t3_events": []}
    punches = [_punch(14200, 1, t) for t in LATE_TURNS]
    elim = attribute_late_t1t3_collapse(
        c_raw, t_raw, control_punch=punches, treatment_punch=[],
    )
    assert (elim["share_alive_elimination"] or 0.0) > 0.70
    assert diagnose_phase_3h({"attribution": elim})["primary_finding"] == (
        "alive_elimination_selection_dominates"
    )

    t_offer = {
        "turn_rows": [{
            "seed": 14200, "seat": 1, "turn": t,
            "t1t3_count_combat_start": 0,
            "alive_at_combat": True, "alive_at_recruit": True,
            "n_replacements": 1, "shop_t1t3_offers_recruit_start": 0,
        } for t in LATE_TURNS],
        "fights": [],
        "last_t1t3_losses": [{
            "seed": 14200, "seat": 1, "class": "tavern_offer_shift",
            "first_loss_turn": 9,
        }],
        "t1t3_events": [],
    }
    offer = attribute_late_t1t3_collapse(
        c_raw, t_offer, control_punch=punches, treatment_punch=[],
    )
    assert (offer["share_tavern_offer_shift"] or 0.0) > 0.70
    assert diagnose_phase_3h({"attribution": offer})["primary_finding"] == (
        "tavern_offer_availability_dominates"
    )


def test_diagnose_routes_mixed_and_smoke():
    mixed = diagnose_phase_3h({
        "attribution": {
            "share_full_board_2q_replacement": 0.40,
            "share_open_slot_fill": 0.25,
            "share_tavern_offer_shift": 0.15,
            "share_generated_transform_triple": 0.05,
            "share_alive_elimination": 0.10,
            "share_leftover": 0.05,
        }
    })
    assert mixed["primary_finding"] == "mixed_route_to_larger"
    leftover = diagnose_phase_3h({
        "attribution": {
            "share_full_board_2q_replacement": 0.10,
            "share_open_slot_fill": 0.08,
            "share_tavern_offer_shift": 0.05,
            "share_generated_transform_triple": 0.04,
            "share_alive_elimination": 0.06,
            "share_leftover": 0.67,
        }
    })
    assert leftover["primary_finding"] == "ranked_residual_needs_next_observable"
    smoke = diagnose_phase_3h(mixed, non_evaluative=True)
    assert smoke["primary_finding"] == "measurement_smoke_non_evaluative"
    assert smoke["evaluative"] is False


def test_compare_retention_reproduces_3g_on_synthetic():
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

    def _fight(seed, turn, winner, loser, *, start_tier=1, carry=100.0):
        return {
            "seed": seed, "lobby": 0, "turn": turn, "kind": "live",
            "ghost": False, "seat_a": winner, "seat_b": loser,
            "winner_seat": winner, "loser_seat": loser,
            "applied_hp_loss": 4, "winner_tavern_tier": 4,
            "fight_outcome": "a_win", "outcome": "win",
            "start_minions": [{
                "tier": start_tier, "n_damaging_hits": 1, "n_hits": 1,
                "survived": False, "recruit_raw": 4, "synthetic_share": 2,
                "opp_carry_attack_pool": carry,
                "opp_scale_add_attack": 0.0,
                "opp_attack_pool_recruit_start": carry,
                "opp_n_alive": 6, "opp_alive": True,
            }],
        }

    control = {
        "turn_rows": [
            _seat_turn(14200, 0, t, 80.0 if t >= 10 else 40.0, t1t3=3)
            for t in INSTRUMENT_TURNS
        ] + [
            _seat_turn(14200, 1, t, 800.0 if t >= 10 else 100.0, t1t3=2)
            for t in INSTRUMENT_TURNS
        ],
        "fights": [
            _fight(14200, 12, winner=0, loser=1, start_tier=1, carry=800.0)
        ],
        "last_t1t3_losses": [],
        "t1t3_events": [],
        "replacement_events": [],
        "board_snapshots": [],
        "game_lengths": [14],
    }
    treatment = {
        "turn_rows": [
            _seat_turn(
                14200, 0, t,
                40.0 if t >= 10 else 40.0,
                t1t3=0 if t >= 10 else 3,
            )
            for t in INSTRUMENT_TURNS
        ] + [
            _seat_turn(14200, 1, t, 50.0 if t >= 10 else 100.0, t1t3=2)
            for t in INSTRUMENT_TURNS
        ],
        "fights": [
            _fight(14200, 12, winner=0, loser=1, start_tier=5, carry=50.0)
        ],
        "last_t1t3_losses": [{
            "seed": 14200, "seat": 0, "turn": 10,
            "class": "full_board_2q_replacement",
            "first_loss_turn": 10,
        }],
        "t1t3_events": [],
        "replacement_events": [],
        "board_snapshots": [],
        "game_lengths": [14],
    }
    cmp = compare_retention(control, treatment)
    decomp = cmp["decomposition_3g"]
    assert decomp["n_control"] == 1
    assert decomp["n_treatment"] == 1
    assert cmp["reconciliation"]["history_link_control"]["n_ok"] == 1
    assert cmp["reconciliation"]["history_link_treatment"]["n_ok"] == 1
    assert cmp["paired_seats"]["n_paired_seats"] >= 1
    late = cmp["attribution"]
    assert late["n_control_late_t1t3_punch"] == 1
    assert late["n_treatment_late_t1t3_punch"] == 0
    assert late["reconciliation_ok"] is True
    assert late["attributed"]["full_board_2q_replacement"] == 1.0


def test_3g_mixture_lock_on_weight_shift_still_holds():
    """3H must still recover the 3G mixture-vs-within split on synthetic rows."""
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
