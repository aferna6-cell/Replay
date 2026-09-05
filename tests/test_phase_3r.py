"""Phase 3R post-play scale-sync input/timing attribution — observational locks."""

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
from ml.phase_3r_prereg import (
    FEATURE_TOGGLE,
    FEATURE_TOGGLE_DEFAULT,
    FORBIDDEN_RANGES,
    FROZEN_ALPHA,
    HOLD_PRS,
    INPUT_FIELDS,
    INSTRUMENT_TURNS,
    METHODOLOGY_VERSION,
    NESTED_SCALE_SYNC_IDENTITY,
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
    PHASE_3R_LOBBIES,
    PHASE_3R_SEED,
    PRIMARY_TURNS,
    REUSED_SEED_HI,
    REUSED_SEED_LO,
    SAME_TURN_SYNC_IDENTITY,
    SCALE_FLOW_RECONCILE_IDENTITY,
    SCALE_SYNC_COMPONENTS,
    SHARE_DOMINANT,
    assert_seed_range_allowed,
    diagnose_phase_3r,
)
from ml.scale_sync_diagnostic import (
    ScaleSyncTracer,
    body_sync_increment,
    decompose_scale_pair,
    exact_scale_increment,
    first_input_diverge_field,
    rounded_scale_increment,
    tier_mass_scale_sync,
)
from ml.survivor_tier_damage_diagnostic import rules_faithful_hero_damage


def test_methodology_is_3r_v1_default_off():
    assert METHODOLOGY_VERSION == "3r_v1"
    assert FEATURE_TOGGLE == "PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING"
    assert FEATURE_TOGGLE_DEFAULT is False
    assert PHASE_2Q_RECRUIT_VALUE_STATS is False
    assert PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING is False
    assert SHARE_DOMINANT == 0.70
    assert PRIMARY_TURNS == (5, 6)
    assert SCALE_SYNC_COMPONENTS == (
        "pre_sync_input_state",
        "sync_timing_count",
        "membership_allocation",
        "rounding_residue",
        "residual",
    )
    assert "firestone_target" in INPUT_FIELDS
    assert "residual_add" in INPUT_FIELDS
    assert "pre_sync_input_state +" in NESTED_SCALE_SYNC_IDENTITY
    assert "Σ body" in SCALE_FLOW_RECONCILE_IDENTITY or "body increment" in SCALE_FLOW_RECONCILE_IDENTITY
    assert "matched by turn" in SAME_TURN_SYNC_IDENTITY


def test_reuses_2s_dev_and_forbids_confirm():
    assert PHASE_3R_SEED == 14200
    assert PHASE_3R_LOBBIES == 500
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


def test_hold_stack_includes_3q_and_frozen_alpha():
    assert HOLD_PRS == (
        29, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
        50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
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
    assert abs(PHASE_3Q_T1_SYNTH_CONTROL - 22.23794950267789) < 1e-12
    assert abs(PHASE_3Q_T1_SYNTH_TREATMENT - 14.631981637337415) < 1e-12
    assert abs(PHASE_3Q_T3_SYNTH_CONTROL - 7.066298342541437) < 1e-12
    assert abs(PHASE_3Q_T3_SYNTH_TREATMENT - 19.185082872928177) < 1e-12
    d = diagnose_phase_3r()
    assert d["no_merge"] is True
    assert d["no_hero_damage_retune"] is True
    assert d["no_behavior_change"] is True
    assert d["no_2q_rewrite"] is True
    assert d["no_recruit_change"] is True
    assert d["keep_hold_prs"][-1] == 63
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


def test_exact_and_rounded_scale_increment():
    assert abs(exact_scale_increment(10, 40, 8.0) - 2.0) < 1e-12
    assert exact_scale_increment(10, 0, 8.0) == 0.0
    # residual apply: share=10/40, add=2, atk/hp split 6/4 → +1.2 / +0.8
    assert rounded_scale_increment(6, 4, 8.0, 40) == (
        max(1, round(6 + 2.0 * 6 / 10)) + max(1, round(4 + 2.0 * 4 / 10)) - 10
    )
    assert rounded_scale_increment(6, 4, 0.0, 40) == 0


def test_first_input_diverge_is_upstream():
    control = {
        "firestone_target": 100.0,
        "growth_factor": 1.1,
        "board_recruit_raw": 20,
        "residual_add": 8.0,
    }
    treatment = dict(control)
    treatment["residual_add"] = 12.0
    assert first_input_diverge_field(control, treatment) == "residual_add"
    treatment["firestone_target"] = 90.0
    assert first_input_diverge_field(control, treatment) == "firestone_target"


def _play_pair(scale_c, scale_t, sticky=20, paint=20, start_c=None, start_t=None):
    if start_c is None:
        start_c = sticky + scale_c
    if start_t is None:
        start_t = paint + scale_t
    control_start = {"synthetic_share": start_c, "board_slot": 0, "name": "A",
                     "card_id": "a", "recruit_raw": 4}
    treatment_start = {"synthetic_share": start_t, "board_slot": 0, "name": "A",
                       "card_id": "a", "recruit_raw": 4}
    control_play = {
        "play_subtype": "open_slot",
        "post_play_pre_realloc": [
            {"slot": 0, "synthetic_share": sticky, "name": "A",
             "card_id": "a", "recruit_raw": 4},
        ],
        "cf_a_paint": [
            {"slot": 0, "synthetic_share": sticky, "name": "A",
             "card_id": "a", "recruit_raw": 4},
        ],
    }
    treatment_play = {
        "play_subtype": "open_slot",
        "post_realloc": [
            {"slot": 0, "synthetic_share": paint, "name": "A",
             "card_id": "a", "recruit_raw": 4},
        ],
        "cf_b_sticky": [
            {"slot": 0, "synthetic_share": paint, "name": "A",
             "card_id": "a", "recruit_raw": 4},
        ],
    }
    return control_start, treatment_start, control_play, treatment_play


def _sync(turn, residual_add, combat_pre, board_combat, actual, **extra):
    synth_pre = extra.pop("synth_pre", 20)
    synth_post = extra.pop("synth_post", synth_pre + actual)
    rec = {
        "turn": turn,
        "residual_add": residual_add,
        "board_combat_raw": board_combat,
        "pre_slots": [{
            "slot": 0, "name": "A", "card_id": "a", "recruit_raw": 4,
            "combat_raw": combat_pre, "synthetic_share": synth_pre,
        }],
        "post_slots": [{
            "slot": 0, "name": "A", "card_id": "a", "recruit_raw": 4,
            "combat_raw": combat_pre + actual,
            "synthetic_share": synth_post,
        }],
        "firestone_target": extra.pop("firestone_target", 100.0),
        "growth_factor": extra.pop("growth_factor", 1.0),
        "board_recruit_raw": extra.pop("board_recruit_raw", 4),
        "abstract_pool_entering": extra.pop("abstract_pool_entering", 20.0),
        "synth_pool_entering": extra.pop("synth_pool_entering", 20.0),
        "end_of_recruit_pre_scaling_stats": board_combat,
    }
    rec.update(extra)
    return rec


def test_decompose_input_only_shift():
    # share_c = share_t = 0.5; R_c=10, R_t=20; actual=exact
    c0, t0, cp, tp = _play_pair(5, 10)
    c_syncs = [_sync(5, 10.0, 10, 20, 5.0, synth_pre=20, firestone_target=80.0)]
    t_syncs = [_sync(5, 20.0, 10, 20, 10.0, synth_pre=20, firestone_target=90.0)]
    parts = decompose_scale_pair(c0, t0, cp, tp, c_syncs, t_syncs)
    explained = (
        parts["pre_sync_input_state"] + parts["sync_timing_count"]
        + parts["membership_allocation"] + parts["rounding_residue"]
        + parts["residual"]
    )
    assert abs(explained - parts["subsequent_scaling"]) < 1e-12
    assert abs(parts["subsequent_scaling"] - 5.0) < 1e-12
    assert abs(parts["pre_sync_input_state"] - 5.0) < 1e-12
    assert abs(parts["sync_timing_count"]) < 1e-12
    assert abs(parts["membership_allocation"]) < 1e-12
    assert abs(parts["rounding_residue"]) < 1e-12
    assert abs(parts["residual"]) < 1e-12
    assert parts["first_input_diverge_field"] == "firestone_target"


def test_decompose_membership_only_shift():
    # same R=10; share_c=0.4, share_t=0.6; actual=exact
    c0, t0, cp, tp = _play_pair(4, 6)
    c_syncs = [_sync(5, 10.0, 8, 20, 4.0, synth_pre=20)]
    t_syncs = [_sync(5, 10.0, 12, 20, 6.0, synth_pre=20)]
    parts = decompose_scale_pair(c0, t0, cp, tp, c_syncs, t_syncs)
    assert abs(parts["subsequent_scaling"] - 2.0) < 1e-12
    assert abs(parts["pre_sync_input_state"]) < 1e-12
    assert abs(parts["membership_allocation"] - 2.0) < 1e-12
    assert abs(parts["sync_timing_count"]) < 1e-12
    assert abs(parts["rounding_residue"]) < 1e-12
    assert abs(parts["residual"]) < 1e-12


def test_decompose_timing_only_extra_sync():
    c0, t0, cp, tp = _play_pair(4, 10)
    common = _sync(5, 10.0, 10, 20, 4.0, synth_pre=20)
    extra = _sync(6, 12.0, 14, 28, 6.0, synth_pre=24)
    parts = decompose_scale_pair(c0, t0, cp, tp, [common], [common, extra])
    assert abs(parts["subsequent_scaling"] - 6.0) < 1e-12
    assert abs(parts["sync_timing_count"] - 6.0) < 1e-12
    assert abs(parts["pre_sync_input_state"]) < 1e-12
    assert abs(parts["membership_allocation"]) < 1e-12
    assert abs(parts["rounding_residue"]) < 1e-12
    assert abs(parts["residual"]) < 1e-12


def test_decompose_rounding_only_shift():
    # exact_c=1.4 actual_c=1; exact_t=1.6 actual_t=2; R and share differ
    # so pin R and share so exact differs, then overwrite via actual
    c0, t0, cp, tp = _play_pair(1, 2)
    c_syncs = [_sync(5, 3.5, 8, 20, 1.0, synth_pre=20)]  # exact=1.4
    t_syncs = [_sync(5, 4.0, 8, 20, 2.0, synth_pre=20)]  # exact=1.6
    parts = decompose_scale_pair(c0, t0, cp, tp, c_syncs, t_syncs)
    # input = 0.4*(4.0-3.5)=0.2; membership=4.0*(0.4-0.4)=0
    # rounding = (2-1.6)-(1-1.4)=0.4-(-0.4)=0.8
    assert abs(parts["subsequent_scaling"] - 1.0) < 1e-12
    assert abs(parts["pre_sync_input_state"] - 0.2) < 1e-12
    assert abs(parts["membership_allocation"]) < 1e-12
    assert abs(parts["rounding_residue"] - 0.8) < 1e-12
    assert abs(parts["residual"]) < 1e-12


def test_body_sync_increment_matches_slots():
    sync = _sync(5, 10.0, 10, 40, 2.0, synth_pre=7)
    inc = body_sync_increment(sync, 0, {"name": "A", "card_id": "a", "recruit_raw": 4})
    assert inc["present_pre"] is True
    assert inc["present_post"] is True
    assert abs(inc["actual"] - 2.0) < 1e-12
    assert abs(inc["exact"] - 2.5) < 1e-12
    assert abs(inc["share"] - 0.25) < 1e-12


def test_tier_mass_scale_sync_does_not_cancel_t1_t3():
    per_tier = {
        "1": {
            "n_pairs": 1307,
            "subsequent_scaling": -3.66,
            "pre_sync_input_state": -3.66,
            "sync_timing_count": 0.0,
            "membership_allocation": 0.0,
            "rounding_residue": 0.0,
            "residual": 0.0,
        },
        "3": {
            "n_pairs": 1086,
            "subsequent_scaling": 5.84,
            "pre_sync_input_state": 5.84,
            "sync_timing_count": 0.0,
            "membership_allocation": 0.0,
            "rounding_residue": 0.0,
            "residual": 0.0,
        },
    }
    primary = tier_mass_scale_sync(per_tier)
    assert primary["share_of_delta_pre_sync_input_state"] > 0.70
    assert (primary["share_of_delta_sync_timing_count"] or 0.0) < 0.10
    routed = diagnose_phase_3r({"primary": primary})
    assert routed["primary_finding"] == "pre_sync_input_state_dominates"
    assert "first-diverging" in routed["recommended_next_step"] or "upstream" in routed["recommended_next_step"]


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
        tracer = ScaleSyncTracer(0, seed, "obs")
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
    hits = [
        f for f in tracer.fights
        if int(f.get("applied_hp_loss") or 0) > 0
    ]
    assert hits
    saw_play = False
    for ev in tracer.play_events:
        assert ev.get("play_subtype") in (
            "open_slot", "sell_buy_play", "sell_play", "triple",
        )
        if ev.get("event") == "play":
            saw_play = True
    assert saw_play
    turns = []
    for ev in tracer.scale_syncs:
        assert ev.get("kind") == "scale_sync"
        assert "board_recruit_raw" in ev
        assert "abstract_pool_entering" in ev
        assert "synth_pool_entering" in ev
        assert "firestone_target" in ev
        assert "residual_add" in ev
        assert "pre_slots" in ev
        assert "post_slots" in ev
        assert "body_alloc" in ev
        assert ev.get("sync_index") is not None
        turns.append(int(ev.get("turn") or 0))
        assert abs(float(ev.get("board_flow_gap") or 0.0)) < 1e-6
    assert any(t in (5, 6) for t in turns)
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
    def _bag(inp, timing, membership, rounding=0.0, residual=None):
        if residual is None:
            residual = 1.0 - (inp + timing + membership + rounding)
        return {
            "primary": {
                "share_of_delta_pre_sync_input_state": inp,
                "share_of_delta_sync_timing_count": timing,
                "share_of_delta_membership_allocation": membership,
                "share_of_delta_rounding_residue": rounding,
                "share_of_delta_residual": residual,
            },
        }

    inp = diagnose_phase_3r(_bag(0.80, 0.05, 0.05))
    assert inp["primary_finding"] == "pre_sync_input_state_dominates"
    assert "upstream" in inp["recommended_next_step"]
    timed = diagnose_phase_3r(_bag(0.05, 0.80, 0.05))
    assert timed["primary_finding"] == "sync_timing_count_dominates"
    assert "timing fidelity" in timed["recommended_next_step"]
    memb = diagnose_phase_3r(_bag(0.05, 0.05, 0.80))
    assert memb["primary_finding"] == "membership_allocation_dominates"
    assert "open-slot" in memb["recommended_next_step"]
    leftover = diagnose_phase_3r(_bag(0.20, 0.20, 0.20))
    assert leftover["primary_finding"] == "ranked_residual_needs_next_observable"
    named = diagnose_phase_3r({
        **_bag(0.80, 0.05, 0.05),
        "attribution": {"modal_first_input_diverge_field": "growth_factor"},
    })
    assert "growth_factor" in named["recommended_next_step"]
    smoke = diagnose_phase_3r(inp, non_evaluative=True)
    assert smoke["primary_finding"] == "measurement_smoke_non_evaluative"
    assert smoke["evaluative"] is False
