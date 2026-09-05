"""Phase 3Q play-lifecycle sticky-vs-repaint causal audit — observational locks."""

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
from ml.phase_3q_prereg import (
    FEATURE_TOGGLE,
    FEATURE_TOGGLE_DEFAULT,
    FORBIDDEN_RANGES,
    FROZEN_ALPHA,
    HOLD_PRS,
    INSTRUMENT_TURNS,
    LIFECYCLE_COMPONENTS,
    METHODOLOGY_VERSION,
    NESTED_LIFECYCLE_IDENTITY,
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
    PHASE_3P_T1_SYNTH_CONTROL,
    PHASE_3P_T1_SYNTH_TREATMENT,
    PHASE_3P_T3_SYNTH_CONTROL,
    PHASE_3P_T3_SYNTH_TREATMENT,
    PHASE_3Q_LOBBIES,
    PHASE_3Q_SEED,
    PLAY_POOL_RECONCILE_IDENTITY,
    PLAY_SUBTYPES,
    PRIMARY_TURNS,
    REUSED_SEED_HI,
    REUSED_SEED_LO,
    SAME_STATE_IDENTITY,
    SHARE_DOMINANT,
    assert_seed_range_allowed,
    diagnose_phase_3q,
)
from ml.play_lifecycle_diagnostic import (
    PlayLifecycleTracer,
    classify_play_subtype,
    decompose_play_pair,
    paint_same_state,
    reconstruct_post_play_sticky,
    sticky_same_state,
    tier_mass_lifecycle,
)
from ml.survivor_tier_damage_diagnostic import rules_faithful_hero_damage


def test_methodology_is_3q_v1_default_off():
    assert METHODOLOGY_VERSION == "3q_v1"
    assert FEATURE_TOGGLE == "PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING"
    assert FEATURE_TOGGLE_DEFAULT is False
    assert PHASE_2Q_RECRUIT_VALUE_STATS is False
    assert PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING is False
    assert SHARE_DOMINANT == 0.70
    assert PRIMARY_TURNS == (5, 6)
    assert LIFECYCLE_COMPONENTS == (
        "same_state_repaint",
        "replacement_lifecycle",
        "subsequent_scaling",
        "residual",
    )
    assert PLAY_SUBTYPES == (
        "open_slot",
        "sell_buy_play",
        "sell_play",
        "triple",
    )
    assert "would-be 2S" in SAME_STATE_IDENTITY or "largest-remainder" in SAME_STATE_IDENTITY
    assert "same_state_repaint +" in NESTED_LIFECYCLE_IDENTITY
    assert "painted_pool" in PLAY_POOL_RECONCILE_IDENTITY


def test_reuses_2s_dev_and_forbids_confirm():
    assert PHASE_3Q_SEED == 14200
    assert PHASE_3Q_LOBBIES == 500
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


def test_hold_stack_includes_3p_and_frozen_alpha():
    assert HOLD_PRS == (
        29, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
        50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
    )
    assert FROZEN_ALPHA == 0.5
    assert abs(PHASE_3D_BOARD_POOL_MAGNITUDE - A1_3E) < 1e-12
    assert PHASE_3N_CLASS3 == 1059
    assert PHASE_3N_CLASS3_T5 == 884
    assert PHASE_3N_CLASS3_T6 == 149
    assert PHASE_3O_PRIMARY_N == 1033
    assert PHASE_3P_PRIMARY_N_PAIRS == 4132
    assert PHASE_3P_PRIMARY_N_FIGHTS == 1033
    assert abs(PHASE_3N_WITHIN_TIER_B - 0.6883852691218131) < 1e-12
    assert abs(PHASE_3O_T5T6_B - 0.6166505324298197) < 1e-12
    assert abs(PHASE_3O_SHARE_START_STATS - 1.2098231585111623) < 1e-12
    assert abs(PHASE_3O_SHARE_SYNTH - 1.1847200887085545) < 1e-12
    assert abs(PHASE_3P_SHARE_TIMING - 0.9857690083784691) < 1e-12
    assert abs(PHASE_3P_SHARE_POOL - 0.006333231838752983) < 1e-12
    assert PHASE_3P_SHARE_WEIGHT == 0.0
    assert abs(PHASE_3P_SHARE_ROUNDING - 0.007897759782777772) < 1e-12
    assert abs(PHASE_3P_T1_SYNTH_CONTROL - 22.23794950267789) < 1e-12
    assert abs(PHASE_3P_T1_SYNTH_TREATMENT - 14.631981637337415) < 1e-12
    assert abs(PHASE_3P_T3_SYNTH_CONTROL - 7.066298342541437) < 1e-12
    assert abs(PHASE_3P_T3_SYNTH_TREATMENT - 19.185082872928177) < 1e-12
    d = diagnose_phase_3q()
    assert d["no_merge"] is True
    assert d["no_hero_damage_retune"] is True
    assert d["no_behavior_change"] is True
    assert d["no_2q_rewrite"] is True
    assert d["no_recruit_change"] is True
    assert d["keep_hold_prs"][-1] == 62
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


def test_play_subtype_open_slot_vs_replacement():
    assert classify_play_subtype(sold=False, bought=False) == "open_slot"
    assert classify_play_subtype(sold=True, bought=True) == "sell_buy_play"
    assert classify_play_subtype(sold=True, bought=False) == "sell_play"
    assert classify_play_subtype(sold=True, bought=True, triple=True) == "triple"
    assert classify_play_subtype(
        sold=True, bought=True, board_full_at_sell=True,
    ) == "sell_buy_play"


def test_reconstruct_sticky_appends_incoming_without_repaint():
    pre = [
        {"slot": 0, "obj_id": 1, "name": "A", "card_id": "a",
         "tier": 1, "recruit_raw": 4, "synthetic_share": 20},
        {"slot": 1, "obj_id": 2, "name": "B", "card_id": "b",
         "tier": 3, "recruit_raw": 10, "synthetic_share": 6},
    ]
    incoming = {
        "slot": 0, "obj_id": 9, "name": "C", "card_id": "c",
        "tier": 2, "recruit_raw": 6, "synthetic_share": 0,
    }
    sticky = reconstruct_post_play_sticky(pre, incoming, [1, 2], [1, 2, 9])
    assert [s["obj_id"] for s in sticky] == [1, 2, 9]
    assert sticky[0]["synthetic_share"] == 20
    assert sticky[1]["synthetic_share"] == 6
    assert sticky[2]["synthetic_share"] == 0
    assert sticky[2]["slot"] == 2
    painted = paint_same_state(sticky, 26.0)
    assert painted["painted_pool"] == 26
    assert painted["shares_sum_to_painted_pool"] is True
    assert sum(int(r["largest_remainder_share"]) for r in painted["rows"]) == 26
    hold = sticky_same_state(sticky)
    assert hold["synthetic_shares_sum"] == 26
    assert [r["synthetic_share"] for r in hold["rows"]] == [20, 6, 0]


def test_decompose_play_identity_sums_to_delta():
    control_start = {"synthetic_share": 22, "board_slot": 0, "name": "A",
                     "card_id": "a", "recruit_raw": 4}
    treatment_start = {"synthetic_share": 14, "board_slot": 0, "name": "A",
                       "card_id": "a", "recruit_raw": 4}
    control_play = {
        "play_subtype": "open_slot",
        "post_play_pre_realloc": [
            {"slot": 0, "synthetic_share": 20, "name": "A",
             "card_id": "a", "recruit_raw": 4},
        ],
        "cf_a_paint": [
            {"slot": 0, "synthetic_share": 13, "name": "A",
             "card_id": "a", "recruit_raw": 4},
        ],
    }
    treatment_play = {
        "play_subtype": "open_slot",
        "post_realloc": [
            {"slot": 0, "synthetic_share": 12, "name": "A",
             "card_id": "a", "recruit_raw": 4},
        ],
        "cf_b_sticky": [
            {"slot": 0, "synthetic_share": 20, "name": "A",
             "card_id": "a", "recruit_raw": 4},
        ],
    }
    parts = decompose_play_pair(
        control_start, treatment_start, control_play, treatment_play,
    )
    explained = (
        parts["same_state_repaint"] + parts["replacement_lifecycle"]
        + parts["subsequent_scaling"] + parts["residual"]
    )
    assert abs(explained - parts["delta_synth"]) < 1e-12
    assert abs(parts["residual"]) < 1e-12
    assert parts["delta_synth"] == -8
    # same_state: 12-20 = -8; lifecycle: 20-20 = 0
    # scale: (14-12) - (22-20) = 2-2 = 0
    assert abs(parts["same_state_repaint"] - (-8.0)) < 1e-12
    assert abs(parts["replacement_lifecycle"]) < 1e-12
    assert abs(parts["subsequent_scaling"]) < 1e-12
    assert abs(parts["same_state_control_cf"] - (-7.0)) < 1e-12


def test_decompose_lifecycle_only_shift():
    control_start = {"synthetic_share": 10, "board_slot": 0, "name": "A",
                     "card_id": "a", "recruit_raw": 4}
    treatment_start = {"synthetic_share": 18, "board_slot": 0, "name": "B",
                       "card_id": "b", "recruit_raw": 8}
    control_play = {
        "play_subtype": "open_slot",
        "post_play_pre_realloc": [
            {"slot": 0, "synthetic_share": 10, "name": "A",
             "card_id": "a", "recruit_raw": 4},
        ],
        "cf_a_paint": [
            {"slot": 0, "synthetic_share": 10, "name": "A",
             "card_id": "a", "recruit_raw": 4},
        ],
    }
    treatment_play = {
        "play_subtype": "sell_buy_play",
        "post_realloc": [
            {"slot": 0, "synthetic_share": 18, "name": "B",
             "card_id": "b", "recruit_raw": 8},
        ],
        "cf_b_sticky": [
            {"slot": 0, "synthetic_share": 18, "name": "B",
             "card_id": "b", "recruit_raw": 8},
        ],
    }
    parts = decompose_play_pair(
        control_start, treatment_start, control_play, treatment_play,
    )
    assert abs(parts["same_state_repaint"]) < 1e-12
    assert abs(parts["replacement_lifecycle"] - 8.0) < 1e-12
    assert abs(parts["subsequent_scaling"]) < 1e-12
    assert abs(parts["residual"]) < 1e-12
    assert parts["subtype_mismatch"] is True


def test_decompose_scaling_only_shift():
    control_start = {"synthetic_share": 30, "board_slot": 0, "name": "A",
                     "card_id": "a", "recruit_raw": 4}
    treatment_start = {"synthetic_share": 40, "board_slot": 0, "name": "A",
                       "card_id": "a", "recruit_raw": 4}
    control_play = {
        "play_subtype": "open_slot",
        "post_play_pre_realloc": [
            {"slot": 0, "synthetic_share": 20, "name": "A",
             "card_id": "a", "recruit_raw": 4},
        ],
        "cf_a_paint": [
            {"slot": 0, "synthetic_share": 20, "name": "A",
             "card_id": "a", "recruit_raw": 4},
        ],
    }
    treatment_play = {
        "play_subtype": "open_slot",
        "post_realloc": [
            {"slot": 0, "synthetic_share": 20, "name": "A",
             "card_id": "a", "recruit_raw": 4},
        ],
        "cf_b_sticky": [
            {"slot": 0, "synthetic_share": 20, "name": "A",
             "card_id": "a", "recruit_raw": 4},
        ],
    }
    parts = decompose_play_pair(
        control_start, treatment_start, control_play, treatment_play,
    )
    assert abs(parts["same_state_repaint"]) < 1e-12
    assert abs(parts["replacement_lifecycle"]) < 1e-12
    assert abs(parts["subsequent_scaling"] - 10.0) < 1e-12
    assert abs(parts["residual"]) < 1e-12


def test_tier_mass_lifecycle_does_not_cancel_t1_t3():
    per_tier = {
        "1": {
            "n_pairs": 1307,
            "delta_synth": -7.6,
            "same_state_repaint": -7.6,
            "replacement_lifecycle": 0.0,
            "subsequent_scaling": 0.0,
            "residual": 0.0,
        },
        "3": {
            "n_pairs": 1086,
            "delta_synth": 12.1,
            "same_state_repaint": 12.1,
            "replacement_lifecycle": 0.0,
            "subsequent_scaling": 0.0,
            "residual": 0.0,
        },
    }
    primary = tier_mass_lifecycle(per_tier)
    assert primary["share_of_delta_same_state_repaint"] > 0.70
    assert (primary["share_of_delta_replacement_lifecycle"] or 0.0) < 0.10
    routed = diagnose_phase_3q({"primary": primary})
    assert routed["primary_finding"] == "same_state_repaint_dominates"
    assert "scientifically defensible" in routed["recommended_next_step"]


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
        tracer = PlayLifecycleTracer(0, seed, "obs")
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
    hits = [
        f for f in tracer.fights
        if int(f.get("applied_hp_loss") or 0) > 0
    ]
    assert hits
    saw_play = False
    for ev in tracer.play_events:
        assert ev.get("play_subtype") in PLAY_SUBTYPES
        assert "pre_play" in ev
        assert "incoming" in ev
        assert "post_play_pre_realloc" in ev
        assert "post_realloc" in ev
        assert "cf_a_paint" in ev
        assert "cf_b_sticky" in ev
        assert ev.get("cf_a_shares_sum_ok") is True
        if ev.get("event") == "play":
            saw_play = True
            if ev.get("incoming") is not None:
                assert "recruit_raw" in ev["incoming"]
                assert "tier" in ev["incoming"]
    assert saw_play
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
    def _bag(same_state, lifecycle, scaling, residual=None):
        if residual is None:
            residual = 1.0 - (same_state + lifecycle + scaling)
        return {
            "primary": {
                "share_of_delta_same_state_repaint": same_state,
                "share_of_delta_replacement_lifecycle": lifecycle,
                "share_of_delta_subsequent_scaling": scaling,
                "share_of_delta_residual": residual,
            },
        }

    same = diagnose_phase_3q(_bag(0.80, 0.05, 0.05))
    assert same["primary_finding"] == "same_state_repaint_dominates"
    assert "scientifically defensible" in same["recommended_next_step"]
    assert "no implementation" in same["recommended_next_step"]
    life = diagnose_phase_3q(_bag(0.05, 0.80, 0.05))
    assert life["primary_finding"] == "replacement_lifecycle_dominates"
    assert "event subtype" in life["recommended_next_step"]
    scale = diagnose_phase_3q(_bag(0.05, 0.05, 0.80))
    assert scale["primary_finding"] == "subsequent_scaling_dominates"
    assert "scale-sync" in scale["recommended_next_step"]
    leftover = diagnose_phase_3q(_bag(0.20, 0.20, 0.20))
    assert leftover["primary_finding"] == "ranked_residual_needs_next_observable"
    smoke = diagnose_phase_3q(same, non_evaluative=True)
    assert smoke["primary_finding"] == "measurement_smoke_non_evaluative"
    assert smoke["evaluative"] is False
