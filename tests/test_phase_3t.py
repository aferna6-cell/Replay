"""Phase 3T earliest T5 incumbent-synth divergence — observational locks."""

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
from ml.phase_3t_prereg import (
    BODY_EVENT_POOL_FLOW_IDENTITY,
    DIVERGENCE_COMPONENTS,
    EXCLUSIVE_T5_FIRST_DIFF_IDENTITY,
    FEATURE_TOGGLE,
    FEATURE_TOGGLE_DEFAULT,
    FORBIDDEN_RANGES,
    FROZEN_ALPHA,
    HOLD_PRS,
    INSTRUMENT_TURNS,
    LIFECYCLE_PROPAGATION_IDENTITY,
    MEMBERSHIP_EVENT_KINDS,
    METHODOLOGY_VERSION,
    NESTED_DIVERGENCE_IDENTITY,
    PHASE_3D_BOARD_POOL_MAGNITUDE,
    PHASE_3N_CLASS3,
    PHASE_3N_CLASS3_T5,
    PHASE_3N_CLASS3_T6,
    PHASE_3N_WITHIN_TIER_B,
    PHASE_3O_PRIMARY_N,
    PHASE_3O_SHARE_START_STATS,
    PHASE_3O_SHARE_SYNTH,
    PHASE_3O_T5T6_B,
    PHASE_3P_PRIMARY_N_FIGHTS,
    PHASE_3P_PRIMARY_N_PAIRS,
    PHASE_3P_SHARE_POOL,
    PHASE_3P_SHARE_ROUNDING,
    PHASE_3P_SHARE_TIMING,
    PHASE_3P_SHARE_WEIGHT,
    PHASE_3Q_PRIMARY_N_FIGHTS,
    PHASE_3Q_PRIMARY_N_PAIRS,
    PHASE_3Q_SHARE_LIFECYCLE,
    PHASE_3Q_SHARE_RESIDUAL,
    PHASE_3Q_SHARE_SAME_STATE,
    PHASE_3Q_SHARE_SCALING,
    PHASE_3Q_T1_SYNTH_CONTROL,
    PHASE_3Q_T1_SYNTH_TREATMENT,
    PHASE_3Q_T3_SYNTH_CONTROL,
    PHASE_3Q_T3_SYNTH_TREATMENT,
    PHASE_3R_PRIMARY_N_FIGHTS,
    PHASE_3R_PRIMARY_N_PAIRS,
    PHASE_3R_SHARE_INPUT,
    PHASE_3R_SHARE_MEMBERSHIP,
    PHASE_3R_SHARE_ROUNDING,
    PHASE_3R_SHARE_TIMING,
    PHASE_3R_T1_SYNTH_CONTROL,
    PHASE_3R_T1_SYNTH_TREATMENT,
    PHASE_3R_T3_SYNTH_CONTROL,
    PHASE_3R_T3_SYNTH_TREATMENT,
    PHASE_3S_LIFECYCLE_ABS_MASS,
    PHASE_3S_MODAL_EARLIEST,
    PHASE_3S_PRIMARY_N_FIGHTS,
    PHASE_3S_PRIMARY_N_PAIRS,
    PHASE_3S_SAME_PRE_PLAY_IDENTITY_RATE,
    PHASE_3S_SAME_PRE_PLAY_STATE_RATE,
    PHASE_3S_SHARE_INCOMING,
    PHASE_3S_SHARE_MEMBERSHIP_PROP,
    PHASE_3S_SHARE_OPENING,
    PHASE_3S_SHARE_ORDER,
    PHASE_3S_SHARE_PRE_PLAY,
    PHASE_3S_SHARE_RESIDUAL,
    PHASE_3T_LOBBIES,
    PHASE_3T_SEED,
    PRIMARY_TURNS,
    REUSED_SEED_HI,
    REUSED_SEED_LO,
    SHARE_DOMINANT,
    WALK_TURN,
    assert_seed_range_allowed,
    diagnose_phase_3t,
)
from ml.t5_incumbent_synth_diagnostic import (
    T5IncumbentSynthTracer,
    decompose_t5_synth_pair,
    first_synth_component,
    in_t5_walk_window,
    incumbent_snapshot,
    sticky_after_membership,
    tier_mass_divergence,
    tier_mass_membership_prop,
)
from ml.survivor_tier_damage_diagnostic import rules_faithful_hero_damage


def test_methodology_is_3t_v1_default_off():
    assert METHODOLOGY_VERSION == "3t_v1"
    assert FEATURE_TOGGLE == "PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING"
    assert FEATURE_TOGGLE_DEFAULT is False
    assert PHASE_2Q_RECRUIT_VALUE_STATS is False
    assert PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING is False
    assert SHARE_DOMINANT == 0.70
    assert PRIMARY_TURNS == (5, 6)
    assert WALK_TURN == 5
    assert DIVERGENCE_COMPONENTS == (
        "carry_in",
        "earlier_t5_membership",
        "paint_repaint",
        "scale_sync",
        "residual",
    )
    assert "sell" in MEMBERSHIP_EVENT_KINDS
    assert "triple" in MEMBERSHIP_EVENT_KINDS
    assert "carry_in +" in NESTED_DIVERGENCE_IDENTITY
    assert "T5 start board" in BODY_EVENT_POOL_FLOW_IDENTITY
    assert "first T5 synth-state" in EXCLUSIVE_T5_FIRST_DIFF_IDENTITY
    assert "replacement_lifecycle" in LIFECYCLE_PROPAGATION_IDENTITY


def test_reuses_2s_dev_and_forbids_confirm():
    assert PHASE_3T_SEED == 14200
    assert PHASE_3T_LOBBIES == 500
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


def test_hold_stack_includes_3s_and_frozen_alpha():
    assert HOLD_PRS == (
        29, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
        50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
    )
    assert FROZEN_ALPHA == 0.5
    assert abs(PHASE_3D_BOARD_POOL_MAGNITUDE - A1_3E) < 1e-12
    assert PHASE_3N_CLASS3 == 1059
    assert PHASE_3N_CLASS3_T5 == 884
    assert PHASE_3N_CLASS3_T6 == 149
    assert PHASE_3O_PRIMARY_N == 1033
    assert PHASE_3P_PRIMARY_N_PAIRS == 4132
    assert PHASE_3P_PRIMARY_N_FIGHTS == 1033
    assert PHASE_3Q_PRIMARY_N_PAIRS == 4132
    assert PHASE_3Q_PRIMARY_N_FIGHTS == 1033
    assert PHASE_3R_PRIMARY_N_PAIRS == 4132
    assert PHASE_3R_PRIMARY_N_FIGHTS == 1033
    assert PHASE_3S_PRIMARY_N_PAIRS == 4132
    assert PHASE_3S_PRIMARY_N_FIGHTS == 1033
    assert abs(PHASE_3N_WITHIN_TIER_B - 0.6883852691218131) < 1e-12
    assert abs(PHASE_3O_T5T6_B - 0.6166505324298197) < 1e-12
    assert abs(PHASE_3O_SHARE_START_STATS - 1.2098231585111623) < 1e-12
    assert abs(PHASE_3O_SHARE_SYNTH - 1.1847200887085545) < 1e-12
    assert abs(PHASE_3P_SHARE_TIMING - 0.9857690083784691) < 1e-12
    assert abs(PHASE_3P_SHARE_POOL - 0.006333231838752983) < 1e-12
    assert PHASE_3P_SHARE_WEIGHT == 0.0
    assert abs(PHASE_3P_SHARE_ROUNDING - 0.007897759782777772) < 1e-12
    assert abs(PHASE_3Q_SHARE_SAME_STATE - 0.07190788470337045) < 1e-12
    assert abs(PHASE_3Q_SHARE_LIFECYCLE - 0.4481470184535611) < 1e-12
    assert abs(PHASE_3Q_SHARE_SCALING - 0.47994509684306846) < 1e-12
    assert PHASE_3Q_SHARE_RESIDUAL == 0.0
    assert abs(PHASE_3R_SHARE_MEMBERSHIP - 0.969763531093672) < 1e-12
    assert abs(PHASE_3R_SHARE_INPUT - 0.011512383221278037) < 1e-12
    assert PHASE_3R_SHARE_TIMING == 0.0
    assert abs(PHASE_3R_SHARE_ROUNDING - 0.018724085685049907) < 1e-12
    assert PHASE_3S_SHARE_PRE_PLAY == 1.0
    assert PHASE_3S_SHARE_INCOMING == 0.0
    assert PHASE_3S_SHARE_OPENING == 0.0
    assert PHASE_3S_SHARE_ORDER == 0.0
    assert PHASE_3S_SHARE_RESIDUAL == 0.0
    assert PHASE_3S_SHARE_MEMBERSHIP_PROP == 1.0
    assert PHASE_3S_SAME_PRE_PLAY_IDENTITY_RATE == 1.0
    assert PHASE_3S_SAME_PRE_PLAY_STATE_RATE == 0.0
    assert PHASE_3S_MODAL_EARLIEST == 5
    assert abs(PHASE_3S_LIFECYCLE_ABS_MASS - 11754.0) < 1e-12
    assert abs(PHASE_3Q_T1_SYNTH_CONTROL - 22.23794950267789) < 1e-12
    assert abs(PHASE_3Q_T1_SYNTH_TREATMENT - 14.631981637337415) < 1e-12
    assert abs(PHASE_3Q_T3_SYNTH_CONTROL - 7.066298342541437) < 1e-12
    assert abs(PHASE_3Q_T3_SYNTH_TREATMENT - 19.185082872928177) < 1e-12
    assert abs(PHASE_3R_T1_SYNTH_CONTROL - PHASE_3Q_T1_SYNTH_CONTROL) < 1e-12
    assert abs(PHASE_3R_T3_SYNTH_TREATMENT - PHASE_3Q_T3_SYNTH_TREATMENT) < 1e-12
    d = diagnose_phase_3t()
    assert d["no_merge"] is True
    assert d["no_hero_damage_retune"] is True
    assert d["no_behavior_change"] is True
    assert d["no_2q_rewrite"] is True
    assert d["no_recruit_change"] is True
    assert d["keep_hold_prs"][-1] == 65
    assert d["history_filters_applied"] is False
    assert d["primary_turns"] == [5, 6]
    assert d["walk_turn"] == 5


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


def _slot(card, tier, raw, synth, slot=0):
    return {
        "slot": slot,
        "card_id": card,
        "name": card,
        "tier": tier,
        "recruit_raw": raw,
        "synthetic_share": synth,
        "obj_id": hash((card, slot)),
    }


def test_incumbent_snapshot_records_identity_tier_raw_synth():
    rows = incumbent_snapshot([
        _slot("a", 1, 4, 20, 0),
        _slot("b", 3, 8, 6, 1),
    ])
    assert rows[0]["card_id"] == "a"
    assert rows[0]["tier"] == 1
    assert rows[0]["recruit_raw"] == 4
    assert rows[0]["synthetic_share"] == 20
    assert rows[1]["tier"] == 3


def test_sticky_after_membership_keeps_surviving_synth():
    pre = [_slot("a", 1, 4, 20, 0), _slot("b", 3, 8, 6, 1)]
    post = [_slot("a", 1, 4, 99, 0)]
    post[0]["obj_id"] = pre[0]["obj_id"]
    sticky = sticky_after_membership(pre, post)
    assert len(sticky) == 1
    assert sticky[0]["synthetic_share"] == 20


def test_in_t5_walk_window_excludes_t5_scale_sync_before_t5_last_play():
    last = {"turn": 5, "action_seq": 2, "seq": 0}
    assert in_t5_walk_window({"kind": "turn_start", "turn": 5, "seq": -2}, last)
    assert in_t5_walk_window({"kind": "play", "turn": 5, "seq": 1}, last)
    assert not in_t5_walk_window({"kind": "play", "turn": 5, "seq": 2}, last)
    assert not in_t5_walk_window({"kind": "scale_sync", "turn": 5, "seq": 0}, last)
    t6_last = {"turn": 6, "action_seq": 0, "seq": 0}
    assert in_t5_walk_window({"kind": "scale_sync", "turn": 5, "seq": 0}, t6_last)
    assert not in_t5_walk_window({"kind": "play", "turn": 6, "seq": 0}, t6_last)


def test_first_synth_component_exclusive_rank():
    start = {"kind": "turn_start", "slots": [_slot("a", 1, 4, 10)]}
    same = {"kind": "turn_start", "slots": [_slot("a", 1, 4, 10)]}
    assert first_synth_component([start], [same]) == "residual"

    carry = {"kind": "turn_start", "slots": [_slot("a", 1, 4, 22)]}
    assert first_synth_component([start], [carry]) == "carry_in"

    mem_c = [
        start,
        {"kind": "play", "slots": [_slot("a", 1, 4, 10), _slot("b", 2, 6, 0)]},
    ]
    mem_t = [
        same,
        {"kind": "play", "slots": [_slot("a", 1, 4, 10), _slot("c", 3, 8, 0)]},
    ]
    assert first_synth_component(mem_c, mem_t) == "earlier_t5_membership"

    paint_c = [
        start,
        {"kind": "play", "slots": [_slot("a", 1, 4, 10)]},
        {"kind": "paint", "slots": [_slot("a", 1, 4, 10)]},
    ]
    paint_t = [
        same,
        {"kind": "play", "slots": [_slot("a", 1, 4, 10)]},
        {"kind": "paint", "slots": [_slot("a", 1, 4, 18)]},
    ]
    assert first_synth_component(paint_c, paint_t) == "paint_repaint"

    sync_c = [
        start,
        {"kind": "play", "slots": [_slot("a", 1, 4, 10)]},
        {"kind": "paint", "slots": [_slot("a", 1, 4, 10)]},
        {"kind": "scale_sync", "slots": [_slot("a", 1, 4, 14)]},
    ]
    sync_t = [
        same,
        {"kind": "play", "slots": [_slot("a", 1, 4, 10)]},
        {"kind": "paint", "slots": [_slot("a", 1, 4, 10)]},
        {"kind": "scale_sync", "slots": [_slot("a", 1, 4, 19)]},
    ]
    assert first_synth_component(sync_c, sync_t) == "scale_sync"


def _play_pair(sticky_c=10, sticky_t=18):
    control_start = {
        "synthetic_share": sticky_c + 4, "board_slot": 0, "name": "A",
        "card_id": "a", "recruit_raw": 4,
    }
    treatment_start = {
        "synthetic_share": sticky_t + 4, "board_slot": 0, "name": "A",
        "card_id": "a", "recruit_raw": 4,
    }
    control_play = {
        "play_subtype": "open_slot",
        "pre_play": [_slot("a", 1, 4, sticky_c)],
        "incoming": {"card_id": "c", "tier": 2, "recruit_raw": 6},
        "post_play_pre_realloc": [
            {"slot": 0, "synthetic_share": sticky_c, "name": "A",
             "card_id": "a", "recruit_raw": 4},
        ],
        "cf_a_paint": [
            {"slot": 0, "synthetic_share": sticky_c, "name": "A",
             "card_id": "a", "recruit_raw": 4},
        ],
    }
    treatment_play = {
        "play_subtype": "open_slot",
        "pre_play": [_slot("a", 1, 4, sticky_t)],
        "incoming": {"card_id": "c", "tier": 2, "recruit_raw": 6},
        "post_realloc": [
            {"slot": 0, "synthetic_share": sticky_t, "name": "A",
             "card_id": "a", "recruit_raw": 4},
        ],
        "cf_b_sticky": [
            {"slot": 0, "synthetic_share": sticky_t, "name": "A",
             "card_id": "a", "recruit_raw": 4},
        ],
    }
    return control_start, treatment_start, control_play, treatment_play


def test_decompose_carry_in_only():
    c0, t0, cp, tp = _play_pair(10, 18)
    c_ev = [{"kind": "turn_start", "slots": [_slot("a", 1, 4, 10)]}]
    t_ev = [{"kind": "turn_start", "slots": [_slot("a", 1, 4, 18)]}]
    parts = decompose_t5_synth_pair(c0, t0, cp, tp, c_ev, t_ev)
    explained = sum(parts[n] for n in DIVERGENCE_COMPONENTS)
    assert abs(explained - parts["replacement_lifecycle"]) < 1e-12
    assert abs(parts["replacement_lifecycle"] - 8.0) < 1e-12
    assert abs(parts["carry_in"] - 8.0) < 1e-12
    assert abs(parts["earlier_t5_membership"]) < 1e-12
    assert abs(parts["paint_repaint"]) < 1e-12
    assert abs(parts["scale_sync"]) < 1e-12
    assert abs(parts["residual"]) < 1e-12
    assert parts["divergence_component"] == "carry_in"


def test_decompose_earlier_t5_membership_only():
    c0, t0, cp, tp = _play_pair(10, 18)
    start = [_slot("a", 1, 4, 10)]
    c_ev = [
        {"kind": "turn_start", "slots": start},
        {"kind": "sell", "slots": []},
    ]
    t_ev = [
        {"kind": "turn_start", "slots": [_slot("a", 1, 4, 10)]},
        {"kind": "play", "slots": [_slot("b", 3, 8, 18)]},
    ]
    parts = decompose_t5_synth_pair(c0, t0, cp, tp, c_ev, t_ev)
    assert parts["divergence_component"] == "earlier_t5_membership"
    assert abs(parts["earlier_t5_membership"] - parts["replacement_lifecycle"]) < 1e-12
    assert abs(parts["carry_in"]) < 1e-12


def test_decompose_paint_repaint_only():
    c0, t0, cp, tp = _play_pair(10, 18)
    start = [_slot("a", 1, 4, 10)]
    c_ev = [
        {"kind": "turn_start", "slots": start},
        {"kind": "play", "slots": start},
        {"kind": "paint", "slots": [_slot("a", 1, 4, 10)]},
    ]
    t_ev = [
        {"kind": "turn_start", "slots": [_slot("a", 1, 4, 10)]},
        {"kind": "play", "slots": [_slot("a", 1, 4, 10)]},
        {"kind": "paint", "slots": [_slot("a", 1, 4, 18)]},
    ]
    parts = decompose_t5_synth_pair(c0, t0, cp, tp, c_ev, t_ev)
    assert parts["divergence_component"] == "paint_repaint"
    assert abs(parts["paint_repaint"] - parts["replacement_lifecycle"]) < 1e-12
    assert abs(parts["carry_in"]) < 1e-12
    assert abs(parts["earlier_t5_membership"]) < 1e-12


def test_decompose_scale_sync_only():
    c0, t0, cp, tp = _play_pair(10, 18)
    start = [_slot("a", 1, 4, 10)]
    c_ev = [
        {"kind": "turn_start", "slots": start},
        {"kind": "paint", "slots": start},
        {"kind": "scale_sync", "slots": [_slot("a", 1, 4, 12)]},
    ]
    t_ev = [
        {"kind": "turn_start", "slots": [_slot("a", 1, 4, 10)]},
        {"kind": "paint", "slots": [_slot("a", 1, 4, 10)]},
        {"kind": "scale_sync", "slots": [_slot("a", 1, 4, 20)]},
    ]
    parts = decompose_t5_synth_pair(c0, t0, cp, tp, c_ev, t_ev)
    assert parts["divergence_component"] == "scale_sync"
    assert abs(parts["scale_sync"] - parts["replacement_lifecycle"]) < 1e-12
    assert abs(parts["paint_repaint"]) < 1e-12


def test_tier_mass_divergence_does_not_cancel_t1_t3():
    per_tier = {
        "1": {
            "n_pairs": 1307,
            "replacement_lifecycle": -3.33,
            "carry_in": -3.33,
            "earlier_t5_membership": 0.0,
            "paint_repaint": 0.0,
            "scale_sync": 0.0,
            "residual": 0.0,
            "membership_allocation": -3.61,
            "membership_carry_in": -3.61,
            "membership_earlier_t5_membership": 0.0,
            "membership_paint_repaint": 0.0,
            "membership_scale_sync": 0.0,
            "membership_residual": 0.0,
        },
        "3": {
            "n_pairs": 1086,
            "replacement_lifecycle": 5.41,
            "carry_in": 5.41,
            "earlier_t5_membership": 0.0,
            "paint_repaint": 0.0,
            "scale_sync": 0.0,
            "residual": 0.0,
            "membership_allocation": 5.82,
            "membership_carry_in": 5.82,
            "membership_earlier_t5_membership": 0.0,
            "membership_paint_repaint": 0.0,
            "membership_scale_sync": 0.0,
            "membership_residual": 0.0,
        },
    }
    primary = tier_mass_divergence(per_tier)
    assert primary["share_of_delta_carry_in"] > 0.70
    assert (primary["share_of_delta_paint_repaint"] or 0.0) < 0.10
    prop = tier_mass_membership_prop(per_tier)
    assert prop["share_of_membership_carry_in"] > 0.70
    routed = diagnose_phase_3t({
        "primary": primary,
        "attribution": {"modal_first_event_kind": "turn_start"},
    })
    assert routed["primary_finding"] == "carry_in_dominates"
    assert "T4" in routed["recommended_next_step"]


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


def _play(seed, with_hook):
    env = BGEnv(seed=seed)
    tracer = None
    if with_hook:
        tracer = T5IncumbentSynthTracer(0, seed, "obs")
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
    assert tracer.pairing_decisions
    assert tracer.hp_rows
    assert tracer.eliminations
    assert tracer.play_events
    assert tracer.scale_syncs
    assert tracer.recruit_ops
    assert tracer.turn_starts
    t5_starts = [s for s in tracer.turn_starts if int(s.get("turn") or 0) == 5]
    assert t5_starts
    for start in t5_starts:
        assert start.get("kind") == "turn_start"
        assert "incumbents" in start
        for row in start["incumbents"]:
            assert "card_id" in row
            assert "tier" in row
            assert "recruit_raw" in row
            assert "synthetic_share" in row
    hits = [
        f for f in tracer.fights
        if int(f.get("applied_hp_loss") or 0) > 0
    ]
    assert hits
    for ev in tracer.scale_syncs:
        assert ev.get("kind") == "scale_sync"
        assert abs(float(ev.get("board_flow_gap") or 0.0)) < 1e-6
    for op in tracer.recruit_ops:
        if op.get("kind") in ("sell", "play"):
            assert "pre_slots" in op
            assert "post_slots" in op
            assert "incumbents_post" in op
    for f in hits:
        rows = f.get("start_minions") or []
        if not rows:
            continue
        assert f.get("shares_sum_to_painted_pool") is True
        assert f["counterfactual_damage"] == rules_faithful_hero_damage(
            f["winner_tavern_tier"],
            [s["tier"] for s in f["actual_survivors"]],
        )


def test_diagnose_routes():
    def _bag(carry, membership, paint, scale=0.0, residual=None):
        if residual is None:
            residual = 1.0 - (carry + membership + paint + scale)
        return {
            "primary": {
                "share_of_delta_carry_in": carry,
                "share_of_delta_earlier_t5_membership": membership,
                "share_of_delta_paint_repaint": paint,
                "share_of_delta_scale_sync": scale,
                "share_of_delta_residual": residual,
            },
        }

    carry = diagnose_phase_3t(_bag(0.80, 0.05, 0.05))
    assert carry["primary_finding"] == "carry_in_dominates"
    assert "T4" in carry["recommended_next_step"]
    memb = diagnose_phase_3t(_bag(0.05, 0.80, 0.05))
    assert memb["primary_finding"] == "earlier_t5_membership_dominates"
    assert "subtype" in memb["recommended_next_step"]
    paint = diagnose_phase_3t(_bag(0.05, 0.05, 0.80))
    assert paint["primary_finding"] == "paint_repaint_dominates"
    assert "sticky-vs-repaint" in paint["recommended_next_step"]
    assert "before any implementation" in paint["recommended_next_step"]
    scale = diagnose_phase_3t(_bag(0.05, 0.05, 0.05, 0.80))
    assert scale["primary_finding"] == "scale_sync_dominates"
    assert "first differing input" in scale["recommended_next_step"]
    leftover = diagnose_phase_3t(_bag(0.20, 0.20, 0.20, 0.20))
    assert leftover["primary_finding"] == "ranked_residual_needs_next_observable"
    named = diagnose_phase_3t({
        **_bag(0.05, 0.80, 0.05),
        "attribution": {"modal_first_event_kind": "triple"},
    })
    assert "triple" in named["recommended_next_step"]
    smoke = diagnose_phase_3t(carry, non_evaluative=True)
    assert smoke["primary_finding"] == "measurement_smoke_non_evaluative"
    assert smoke["evaluative"] is False
