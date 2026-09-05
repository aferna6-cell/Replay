"""Phase 3S open-slot board-formation attribution — observational locks."""

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
from ml.phase_3s_prereg import (
    EXCLUSIVE_FIRST_DIFF_IDENTITY,
    FEATURE_TOGGLE,
    FEATURE_TOGGLE_DEFAULT,
    FORBIDDEN_RANGES,
    FORMATION_COMPONENTS,
    FORMATION_FLOW_RECONCILE_IDENTITY,
    FROZEN_ALPHA,
    HOLD_PRS,
    INSTRUMENT_TURNS,
    MEMBERSHIP_PROPAGATION_IDENTITY,
    METHODOLOGY_VERSION,
    NESTED_FORMATION_IDENTITY,
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
    PHASE_3S_LOBBIES,
    PHASE_3S_SEED,
    PRIMARY_TURNS,
    REUSED_SEED_HI,
    REUSED_SEED_LO,
    SHARE_DOMINANT,
    SLOT_OPENING_CAUSES,
    assert_seed_range_allowed,
    diagnose_phase_3s,
)
from ml.open_slot_formation_diagnostic import (
    OpenSlotFormationTracer,
    board_composition_key,
    board_membership_key,
    classify_slot_opening_cause,
    decompose_formation_pair,
    earliest_membership_diverge_turn,
    first_formation_component,
    incoming_identity_key,
    slot_opening_key,
    stamp_play_opening,
    tier_mass_formation,
    tier_mass_membership_prop,
)
from ml.survivor_tier_damage_diagnostic import rules_faithful_hero_damage


def test_methodology_is_3s_v1_default_off():
    assert METHODOLOGY_VERSION == "3s_v1"
    assert FEATURE_TOGGLE == "PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING"
    assert FEATURE_TOGGLE_DEFAULT is False
    assert PHASE_2Q_RECRUIT_VALUE_STATS is False
    assert PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING is False
    assert SHARE_DOMINANT == 0.70
    assert PRIMARY_TURNS == (5, 6)
    assert FORMATION_COMPONENTS == (
        "pre_play_membership",
        "incoming_identity",
        "slot_opening_cause",
        "buy_play_order",
        "residual",
    )
    assert SLOT_OPENING_CAUSES == (
        "normal_underfill",
        "prior_sell",
        "death_cleanup",
        "triple_transform",
    )
    assert "pre_play_membership +" in NESTED_FORMATION_IDENTITY
    assert "sticky occupant" in FORMATION_FLOW_RECONCILE_IDENTITY
    assert "first differing" in EXCLUSIVE_FIRST_DIFF_IDENTITY
    assert "membership_allocation" in MEMBERSHIP_PROPAGATION_IDENTITY


def test_reuses_2s_dev_and_forbids_confirm():
    assert PHASE_3S_SEED == 14200
    assert PHASE_3S_LOBBIES == 500
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


def test_hold_stack_includes_3r_and_frozen_alpha():
    assert HOLD_PRS == (
        29, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
        50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
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
    assert abs(PHASE_3Q_T1_SYNTH_CONTROL - 22.23794950267789) < 1e-12
    assert abs(PHASE_3Q_T1_SYNTH_TREATMENT - 14.631981637337415) < 1e-12
    assert abs(PHASE_3Q_T3_SYNTH_CONTROL - 7.066298342541437) < 1e-12
    assert abs(PHASE_3Q_T3_SYNTH_TREATMENT - 19.185082872928177) < 1e-12
    assert abs(PHASE_3R_T1_SYNTH_CONTROL - PHASE_3Q_T1_SYNTH_CONTROL) < 1e-12
    assert abs(PHASE_3R_T3_SYNTH_TREATMENT - PHASE_3Q_T3_SYNTH_TREATMENT) < 1e-12
    d = diagnose_phase_3s()
    assert d["no_merge"] is True
    assert d["no_hero_damage_retune"] is True
    assert d["no_behavior_change"] is True
    assert d["no_2q_rewrite"] is True
    assert d["no_recruit_change"] is True
    assert d["keep_hold_prs"][-1] == 64
    assert d["history_filters_applied"] is False
    assert d["primary_turns"] == [5, 6]


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


def test_board_and_incoming_keys_ignore_synth():
    a = [
        {"slot": 0, "card_id": "a", "tier": 1, "recruit_raw": 4, "synthetic_share": 20},
        {"slot": 1, "card_id": "b", "tier": 3, "recruit_raw": 8, "synthetic_share": 6},
    ]
    b = [
        {"slot": 1, "card_id": "b", "tier": 3, "recruit_raw": 8, "synthetic_share": 99},
        {"slot": 0, "card_id": "a", "tier": 1, "recruit_raw": 4, "synthetic_share": 1},
    ]
    assert board_membership_key(a) == board_membership_key(b)
    assert board_composition_key(a) == board_composition_key(b)
    swapped = [
        {"slot": 0, "card_id": "b", "tier": 3, "recruit_raw": 8},
        {"slot": 1, "card_id": "a", "tier": 1, "recruit_raw": 4},
    ]
    assert board_membership_key(a) == board_membership_key(swapped)
    assert board_composition_key(a) != board_composition_key(swapped)
    assert incoming_identity_key(
        {"card_id": "c", "tier": 2, "recruit_raw": 6, "synthetic_share": 0}
    ) == incoming_identity_key(
        {"card_id": "c", "tier": 2, "recruit_raw": 6, "synthetic_share": 12}
    )


def test_classify_slot_opening_underfill_sell_death_triple():
    play = {"turn": 5, "seq": 2, "board_len_before": 4}
    under = classify_slot_opening_cause(play, [])
    assert under["slot_opening_cause"] == "normal_underfill"
    assert under["underfill"] is True

    sell = classify_slot_opening_cause(play, [
        {"turn": 4, "seq": 0, "kind": "sell", "vacancy_kind": "prior_sell",
         "board_len_before": 5, "board_len_after": 4},
    ])
    assert sell["slot_opening_cause"] == "prior_sell"
    assert sell["slot_opening_turn"] == 4
    assert sell["turns_open"] == 1

    refilled = classify_slot_opening_cause(play, [
        {"turn": 3, "seq": 0, "kind": "sell", "vacancy_kind": "prior_sell",
         "board_len_before": 5, "board_len_after": 4},
        {"turn": 3, "seq": 1, "kind": "play", "board_len_before": 4,
         "board_len_after": 5},
    ])
    assert refilled["slot_opening_cause"] == "normal_underfill"

    death = classify_slot_opening_cause(play, [
        {"turn": 5, "seq": -1, "kind": "death_cleanup",
         "vacancy_kind": "death_cleanup",
         "board_len_before": 6, "board_len_after": 4},
    ])
    assert death["slot_opening_cause"] == "death_cleanup"

    triple = classify_slot_opening_cause(play, [
        {"turn": 4, "seq": 1, "kind": "play", "vacancy_kind": "triple_transform",
         "board_len_before": 6, "board_len_after": 4},
    ])
    assert triple["slot_opening_cause"] == "triple_transform"


def test_stamp_play_opening_attaches_cause():
    play = {
        "turn": 6, "seq": 0, "board_len_before": 3, "play_subtype": "open_slot",
    }
    stamped = stamp_play_opening(play, [
        {"turn": 5, "seq": 0, "vacancy_kind": "prior_sell",
         "board_len_before": 4, "board_len_after": 3},
    ])
    assert stamped["slot_opening_cause"] == "prior_sell"
    assert stamped["turns_open"] == 1
    assert slot_opening_key(stamped)[0] == "prior_sell"


def test_first_formation_component_exclusive_rank():
    base_pre = [
        {"slot": 0, "card_id": "a", "tier": 1, "recruit_raw": 4},
    ]
    incoming = {"card_id": "c", "tier": 2, "recruit_raw": 6}
    control = {
        "pre_play": base_pre,
        "board_len_before": 1,
        "incoming": incoming,
        "incoming_tier": 2,
        "incoming_recruit_raw": 6,
        "slot_opening_cause": "normal_underfill",
        "slot_opening_turn": None,
        "turn": 5,
        "turns_open": None,
        "buy_play_order": ["buy", "play"],
        "gold": 3,
        "could_afford_buy": False,
    }
    same = dict(control)
    same["pre_play"] = [dict(base_pre[0])]
    same["incoming"] = dict(incoming)
    assert first_formation_component(control, same) == "residual"

    other_board = dict(same)
    other_board["pre_play"] = [
        {"slot": 0, "card_id": "z", "tier": 3, "recruit_raw": 10},
    ]
    assert first_formation_component(control, other_board) == "pre_play_membership"

    other_in = dict(same)
    other_in["incoming"] = {"card_id": "d", "tier": 4, "recruit_raw": 12}
    assert first_formation_component(control, other_in) == "incoming_identity"

    other_open = dict(same)
    other_open["slot_opening_cause"] = "prior_sell"
    other_open["slot_opening_turn"] = 4
    other_open["turns_open"] = 1
    assert first_formation_component(control, other_open) == "slot_opening_cause"

    other_order = dict(same)
    other_order["buy_play_order"] = ["play"]
    other_order["gold"] = 6
    assert first_formation_component(control, other_order) == "buy_play_order"


def _play_pair(life_c, life_t, sticky_c=20, sticky_t=20, start_c=None, start_t=None):
    if start_c is None:
        start_c = sticky_c + 4
    if start_t is None:
        start_t = sticky_t + 4
    control_start = {"synthetic_share": start_c, "board_slot": 0, "name": "A",
                     "card_id": "a", "recruit_raw": 4}
    treatment_start = {"synthetic_share": start_t, "board_slot": 0, "name": "A",
                       "card_id": "a", "recruit_raw": 4}
    control_play = {
        "play_subtype": "open_slot",
        "pre_play": [
            {"slot": 0, "card_id": "a", "tier": 1, "recruit_raw": 4,
             "synthetic_share": sticky_c},
        ],
        "incoming": {"card_id": "c", "tier": 2, "recruit_raw": 6},
        "board_len_before": 1,
        "slot_opening_cause": "normal_underfill",
        "slot_opening_turn": None,
        "turn": 5,
        "turns_open": None,
        "buy_play_order": ["buy", "play"],
        "gold": 3,
        "could_afford_buy": False,
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
        "pre_play": [
            {"slot": 0, "card_id": "a", "tier": 1, "recruit_raw": 4,
             "synthetic_share": sticky_t},
        ],
        "incoming": {"card_id": "c", "tier": 2, "recruit_raw": 6},
        "board_len_before": 1,
        "slot_opening_cause": "normal_underfill",
        "slot_opening_turn": None,
        "turn": 5,
        "turns_open": None,
        "buy_play_order": ["buy", "play"],
        "gold": 3,
        "could_afford_buy": False,
        "post_realloc": [
            {"slot": 0, "synthetic_share": sticky_t, "name": "A",
             "card_id": "a", "recruit_raw": 4},
        ],
        "cf_b_sticky": [
            {"slot": 0, "synthetic_share": sticky_t, "name": "A",
             "card_id": "a", "recruit_raw": 4},
        ],
    }
    del life_c, life_t
    return control_start, treatment_start, control_play, treatment_play


def test_decompose_pre_play_membership_only():
    c0, t0, cp, tp = _play_pair(10, 18, sticky_c=10, sticky_t=18)
    tp["pre_play"] = [
        {"slot": 0, "card_id": "b", "tier": 3, "recruit_raw": 8,
         "synthetic_share": 18},
    ]
    parts = decompose_formation_pair(c0, t0, cp, tp)
    explained = sum(parts[n] for n in FORMATION_COMPONENTS)
    assert abs(explained - parts["replacement_lifecycle"]) < 1e-12
    assert abs(parts["replacement_lifecycle"] - 8.0) < 1e-12
    assert abs(parts["pre_play_membership"] - 8.0) < 1e-12
    assert abs(parts["incoming_identity"]) < 1e-12
    assert abs(parts["slot_opening_cause"]) < 1e-12
    assert abs(parts["buy_play_order"]) < 1e-12
    assert abs(parts["residual"]) < 1e-12
    assert parts["formation_component"] == "pre_play_membership"


def test_decompose_incoming_identity_only():
    c0, t0, cp, tp = _play_pair(10, 18, sticky_c=10, sticky_t=18)
    # same pre-play membership; incoming differs. Lifecycle is still the
    # sticky occupant synth gap (the paired slot is the incumbent here).
    tp["incoming"] = {"card_id": "d", "tier": 4, "recruit_raw": 14}
    parts = decompose_formation_pair(c0, t0, cp, tp)
    assert parts["formation_component"] == "incoming_identity"
    assert abs(parts["incoming_identity"] - parts["replacement_lifecycle"]) < 1e-12
    assert abs(parts["pre_play_membership"]) < 1e-12
    assert abs(parts["residual"]) < 1e-12


def test_decompose_slot_opening_only():
    c0, t0, cp, tp = _play_pair(10, 14, sticky_c=10, sticky_t=14)
    tp["slot_opening_cause"] = "death_cleanup"
    tp["slot_opening_turn"] = 4
    tp["turns_open"] = 1
    parts = decompose_formation_pair(c0, t0, cp, tp)
    assert parts["formation_component"] == "slot_opening_cause"
    assert abs(parts["slot_opening_cause"] - 4.0) < 1e-12
    assert abs(parts["pre_play_membership"]) < 1e-12
    assert abs(parts["incoming_identity"]) < 1e-12


def test_decompose_buy_play_order_only():
    c0, t0, cp, tp = _play_pair(12, 16, sticky_c=12, sticky_t=16)
    tp["buy_play_order"] = ["sell", "buy", "play"]
    tp["gold"] = 1
    parts = decompose_formation_pair(c0, t0, cp, tp)
    assert parts["formation_component"] == "buy_play_order"
    assert abs(parts["buy_play_order"] - 4.0) < 1e-12
    assert abs(parts["slot_opening_cause"]) < 1e-12


def test_earliest_membership_diverge_is_first_turn():
    c_plays = [
        {"turn": 3, "pre_play": [{"card_id": "a", "tier": 1, "recruit_raw": 4}],
         "incoming": {"card_id": "x", "tier": 1, "recruit_raw": 3}},
        {"turn": 5, "pre_play": [{"card_id": "b", "tier": 2, "recruit_raw": 6}],
         "incoming": {"card_id": "y", "tier": 2, "recruit_raw": 5}},
    ]
    t_plays = [
        {"turn": 3, "pre_play": [{"card_id": "a", "tier": 1, "recruit_raw": 4}],
         "incoming": {"card_id": "x", "tier": 1, "recruit_raw": 3}},
        {"turn": 5, "pre_play": [{"card_id": "z", "tier": 4, "recruit_raw": 12}],
         "incoming": {"card_id": "y", "tier": 2, "recruit_raw": 5}},
    ]
    assert earliest_membership_diverge_turn(c_plays, t_plays, 6) == 5


def test_tier_mass_formation_does_not_cancel_t1_t3():
    per_tier = {
        "1": {
            "n_pairs": 1307,
            "replacement_lifecycle": -3.33,
            "pre_play_membership": -3.33,
            "incoming_identity": 0.0,
            "slot_opening_cause": 0.0,
            "buy_play_order": 0.0,
            "residual": 0.0,
            "membership_allocation": -3.61,
            "membership_pre_play_membership": -3.61,
            "membership_incoming_identity": 0.0,
            "membership_slot_opening_cause": 0.0,
            "membership_buy_play_order": 0.0,
            "membership_residual": 0.0,
        },
        "3": {
            "n_pairs": 1086,
            "replacement_lifecycle": 5.41,
            "pre_play_membership": 5.41,
            "incoming_identity": 0.0,
            "slot_opening_cause": 0.0,
            "buy_play_order": 0.0,
            "residual": 0.0,
            "membership_allocation": 5.82,
            "membership_pre_play_membership": 5.82,
            "membership_incoming_identity": 0.0,
            "membership_slot_opening_cause": 0.0,
            "membership_buy_play_order": 0.0,
            "membership_residual": 0.0,
        },
    }
    primary = tier_mass_formation(per_tier)
    assert primary["share_of_delta_pre_play_membership"] > 0.70
    assert (primary["share_of_delta_incoming_identity"] or 0.0) < 0.10
    prop = tier_mass_membership_prop(per_tier)
    assert prop["share_of_membership_pre_play_membership"] > 0.70
    routed = diagnose_phase_3s({
        "primary": primary,
        "attribution": {"modal_earliest_membership_diverge_turn": 3},
    })
    assert routed["primary_finding"] == "pre_play_membership_dominates"
    assert "T3" in routed["recommended_next_step"] or "composition" in routed["recommended_next_step"]


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
        tracer = OpenSlotFormationTracer(0, seed, "obs")
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
    hits = [
        f for f in tracer.fights
        if int(f.get("applied_hp_loss") or 0) > 0
    ]
    assert hits
    saw_play = False
    saw_open = False
    for ev in tracer.play_events:
        assert ev.get("play_subtype") in (
            "open_slot", "sell_buy_play", "sell_play", "triple",
        )
        assert "gold" in ev
        assert "shop" in ev
        assert "buy_play_order" in ev
        if ev.get("event") == "play":
            saw_play = True
        if ev.get("play_subtype") == "open_slot":
            saw_open = True
            stamped = stamp_play_opening(ev, list(tracer.combat_shrinks) + [
                o for o in tracer.recruit_ops
                if int(o.get("seat") or -1) == int(ev.get("seat") or -2)
            ])
            assert stamped["slot_opening_cause"] in SLOT_OPENING_CAUSES
    assert saw_play
    assert saw_open
    for ev in tracer.scale_syncs:
        assert ev.get("kind") == "scale_sync"
        assert abs(float(ev.get("board_flow_gap") or 0.0)) < 1e-6
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
    def _bag(memb, incoming, opening, order=0.0, residual=None):
        if residual is None:
            residual = 1.0 - (memb + incoming + opening + order)
        return {
            "primary": {
                "share_of_delta_pre_play_membership": memb,
                "share_of_delta_incoming_identity": incoming,
                "share_of_delta_slot_opening_cause": opening,
                "share_of_delta_buy_play_order": order,
                "share_of_delta_residual": residual,
            },
        }

    memb = diagnose_phase_3s(_bag(0.80, 0.05, 0.05))
    assert memb["primary_finding"] == "pre_play_membership_dominates"
    assert "composition" in memb["recommended_next_step"]
    incoming = diagnose_phase_3s(_bag(0.05, 0.80, 0.05))
    assert incoming["primary_finding"] == "incoming_identity_dominates"
    assert "shop" in incoming["recommended_next_step"] or "2Q" in incoming["recommended_next_step"]
    opening = diagnose_phase_3s(_bag(0.05, 0.05, 0.80))
    assert opening["primary_finding"] == "slot_opening_cause_dominates"
    assert "lifecycle" in opening["recommended_next_step"]
    order = diagnose_phase_3s(_bag(0.05, 0.05, 0.05, 0.80))
    assert order["primary_finding"] == "buy_play_order_dominates"
    leftover = diagnose_phase_3s(_bag(0.20, 0.20, 0.20, 0.20))
    assert leftover["primary_finding"] == "ranked_residual_needs_next_observable"
    named = diagnose_phase_3s({
        **_bag(0.80, 0.05, 0.05),
        "attribution": {"modal_earliest_membership_diverge_turn": 4},
    })
    assert "T4" in named["recommended_next_step"]
    smoke = diagnose_phase_3s(memb, non_evaluative=True)
    assert smoke["primary_finding"] == "measurement_smoke_non_evaluative"
    assert smoke["evaluative"] is False
