"""Phase 3P synthetic-pool allocation-input attribution — observational locks."""

import random

from hsbg_coach.bg_env import (
    A_BUY0,
    A_PLAY0,
    A_SELL0,
    PHASE_2Q_RECRUIT_VALUE_STATS,
    PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING,
    BGEnv,
    EnvMinion,
    PlayerState,
    greedy_policy,
    reallocate_abstract_pool,
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
from ml.phase_3p_prereg import (
    ALLOCATION_COMPONENTS,
    FEATURE_TOGGLE,
    FEATURE_TOGGLE_DEFAULT,
    FORBIDDEN_RANGES,
    FROZEN_ALPHA,
    HOLD_PRS,
    INSTRUMENT_TURNS,
    METHODOLOGY_VERSION,
    NESTED_ALLOCATION_IDENTITY,
    PAINT_EQUATION_IDENTITY,
    PAINT_RECONCILE_IDENTITY,
    PHASE_3D_BOARD_POOL_MAGNITUDE,
    PHASE_3N_CLASS3,
    PHASE_3N_CLASS3_T5,
    PHASE_3N_CLASS3_T6,
    PHASE_3N_WITHIN_TIER_B,
    PHASE_3O_PRIMARY_N,
    PHASE_3O_SHARE_START_STATS,
    PHASE_3O_SHARE_SYNTH,
    PHASE_3O_T1_SYNTH_CONTROL,
    PHASE_3O_T1_SYNTH_TREATMENT,
    PHASE_3O_T3_SYNTH_CONTROL,
    PHASE_3O_T3_SYNTH_TREATMENT,
    PHASE_3O_T5T6_B,
    PHASE_3P_LOBBIES,
    PHASE_3P_SEED,
    PRIMARY_TURNS,
    REUSED_SEED_HI,
    REUSED_SEED_LO,
    SHARE_DOMINANT,
    assert_seed_range_allowed,
    diagnose_phase_3p,
)
from ml.allocation_input_diagnostic import (
    AllocationInputTracer,
    classify_membership_event,
    decompose_synth_pair,
    exact_proportional_shares,
    paint_weight,
    painted_pool_from_abstract,
    reconstruct_board_paint,
)
from ml.synthetic_allocation_diagnostic import largest_remainder_shares
from ml.survivor_tier_damage_diagnostic import rules_faithful_hero_damage


def test_methodology_is_3p_v1_default_off():
    assert METHODOLOGY_VERSION == "3p_v1"
    assert FEATURE_TOGGLE == "PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING"
    assert FEATURE_TOGGLE_DEFAULT is False
    assert PHASE_2Q_RECRUIT_VALUE_STATS is False
    assert PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING is False
    assert SHARE_DOMINANT == 0.70
    assert PRIMARY_TURNS == (5, 6)
    assert ALLOCATION_COMPONENTS == (
        "pool_magnitude",
        "weight_composition",
        "timing_membership",
        "integer_rounding",
        "residual",
    )
    assert "round(abstract_pool)" in PAINT_EQUATION_IDENTITY
    assert "largest_remainder" in PAINT_EQUATION_IDENTITY
    assert "sum to treatment" in PAINT_RECONCILE_IDENTITY
    assert "pool_magnitude +" in NESTED_ALLOCATION_IDENTITY


def test_reuses_2s_dev_and_forbids_confirm():
    assert PHASE_3P_SEED == 14200
    assert PHASE_3P_LOBBIES == 500
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


def test_hold_stack_includes_3o_and_frozen_alpha():
    assert HOLD_PRS == (
        29, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
        50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    )
    assert FROZEN_ALPHA == 0.5
    assert abs(PHASE_3D_BOARD_POOL_MAGNITUDE - A1_3E) < 1e-12
    assert PHASE_3N_CLASS3 == 1059
    assert PHASE_3N_CLASS3_T5 == 884
    assert PHASE_3N_CLASS3_T6 == 149
    assert PHASE_3O_PRIMARY_N == 1033
    assert abs(PHASE_3N_WITHIN_TIER_B - 0.6883852691218131) < 1e-12
    assert abs(PHASE_3O_T5T6_B - 0.6166505324298197) < 1e-12
    assert abs(PHASE_3O_SHARE_START_STATS - 1.2098231585111623) < 1e-12
    assert abs(PHASE_3O_SHARE_SYNTH - 1.1847200887085545) < 1e-12
    assert abs(PHASE_3O_T1_SYNTH_CONTROL - 22.23794950267789) < 1e-12
    assert abs(PHASE_3O_T1_SYNTH_TREATMENT - 14.631981637337415) < 1e-12
    assert abs(PHASE_3O_T3_SYNTH_CONTROL - 7.066298342541437) < 1e-12
    assert abs(PHASE_3O_T3_SYNTH_TREATMENT - 19.185082872928177) < 1e-12
    d = diagnose_phase_3p()
    assert d["no_merge"] is True
    assert d["no_hero_damage_retune"] is True
    assert d["no_behavior_change"] is True
    assert d["no_2q_rewrite"] is True
    assert d["keep_hold_prs"][-1] == 61
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


def test_painted_pool_is_round_abstract_and_lr_reconciles():
    """Largest-remainder paint: Σ (combat − recruit) == round(abstract_pool)."""
    p = PlayerState(0)
    p.board = [
        EnvMinion("a", "A", 1, 2, 3, [], []),
        EnvMinion("b", "B", 3, 5, 5, [], []),
        EnvMinion("c", "C", 2, 1, 2, [], []),
    ]
    p.abstract_pool = 100.0
    reallocate_abstract_pool(p)
    shares = [
        (m.attack + m.health) - (m.recruit_attack + m.recruit_health)
        for m in p.board
    ]
    assert sum(shares) == painted_pool_from_abstract(p.abstract_pool)
    assert painted_pool_from_abstract(p.abstract_pool) == 100
    expected = largest_remainder_shares(
        [m.recruit_attack + m.recruit_health for m in p.board],
        painted_pool_from_abstract(p.abstract_pool),
    )
    assert shares == expected
    rows = [
        {
            "recruit_raw": m.recruit_attack + m.recruit_health,
            "synthetic_share": s,
            "tier": m.tier,
            "board_slot": i,
        }
        for i, (m, s) in enumerate(zip(p.board, shares))
    ]
    paint = reconstruct_board_paint(rows, p.abstract_pool)
    assert paint["painted_pool"] == 100
    assert paint["shares_sum_to_painted_pool"] is True
    assert paint["painted_matches_expected"] is True
    assert abs(sum(paint["exact_proportional_shares"]) - 100) < 1e-9
    for r in paint["rows"]:
        assert r["paint_weight"] == paint_weight(r["recruit_raw"])
        assert r["board_recruit_denom"] == paint["board_recruit_denom"]
        assert abs(
            r["rounding_contribution"]
            - (r["largest_remainder_share"] - r["exact_proportional_share"])
        ) < 1e-12


def test_zero_raw_body_gets_unit_paint_weight():
    assert paint_weight(0) == 1
    assert paint_weight(-2) == 1
    assert paint_weight(7) == 7
    exact = exact_proportional_shares([0, 10], 11)
    assert abs(sum(exact) - 11) < 1e-12
    assert exact[0] == 1.0
    assert exact[1] == 10.0


def test_membership_event_classifies_sell_play_triple():
    assert classify_membership_event(A_SELL0, [1, 2, 3], [1, 3]) == "sell"
    assert classify_membership_event(A_PLAY0, [1, 2], [1, 2, 9]) == "play"
    assert classify_membership_event(A_PLAY0, [1, 2, 3], [9]) == "triple"
    assert classify_membership_event(A_BUY0, [1, 2, 3], [1]) == "triple"
    assert classify_membership_event(A_SELL0, [1, 2], [1, 2]) is None


def test_decompose_identity_sums_to_delta():
    control = {
        "synthetic_share": 20,
        "painted_pool": 40,
        "weight_share": 0.5,
        "exact_proportional_share": 20.0,
        "largest_remainder_share": 20,
        "last_membership_event": "sell",
    }
    treatment = {
        "synthetic_share": 12,
        "painted_pool": 60,
        "weight_share": 0.25,
        "exact_proportional_share": 15.0,
        "largest_remainder_share": 15,
        "last_membership_event": "play",
    }
    parts = decompose_synth_pair(control, treatment)
    explained = (
        parts["pool_magnitude"] + parts["weight_composition"]
        + parts["timing_membership"] + parts["integer_rounding"]
        + parts["residual"]
    )
    assert abs(explained - parts["delta_synth"]) < 1e-12
    assert abs(parts["residual"]) < 1e-12
    assert parts["delta_synth"] == -8
    # pool: (60-40)*0.25 = 5; composition: 40*(0.25-0.5) = -10
    # timing: (20-20)+(12-15) = -3; rounding: (15-15)-(20-20) = 0
    assert abs(parts["pool_magnitude"] - 5.0) < 1e-12
    assert abs(parts["weight_composition"] - (-10.0)) < 1e-12
    assert abs(parts["timing_membership"] - (-3.0)) < 1e-12
    assert abs(parts["integer_rounding"]) < 1e-12
    assert parts["event_kind_mismatch"] is True


def test_decompose_pool_only_shift():
    control = {
        "synthetic_share": 10,
        "painted_pool": 40,
        "weight_share": 0.25,
        "exact_proportional_share": 10.0,
        "largest_remainder_share": 10,
        "last_membership_event": "play",
    }
    treatment = {
        "synthetic_share": 20,
        "painted_pool": 80,
        "weight_share": 0.25,
        "exact_proportional_share": 20.0,
        "largest_remainder_share": 20,
        "last_membership_event": "play",
    }
    parts = decompose_synth_pair(control, treatment)
    assert abs(parts["pool_magnitude"] - 10.0) < 1e-12
    assert abs(parts["weight_composition"]) < 1e-12
    assert abs(parts["timing_membership"]) < 1e-12
    assert abs(parts["integer_rounding"]) < 1e-12
    assert abs(parts["residual"]) < 1e-12


def test_decompose_weight_only_shift():
    control = {
        "synthetic_share": 20,
        "painted_pool": 40,
        "weight_share": 0.5,
        "exact_proportional_share": 20.0,
        "largest_remainder_share": 20,
        "last_membership_event": "play",
    }
    treatment = {
        "synthetic_share": 10,
        "painted_pool": 40,
        "weight_share": 0.25,
        "exact_proportional_share": 10.0,
        "largest_remainder_share": 10,
        "last_membership_event": "play",
    }
    parts = decompose_synth_pair(control, treatment)
    assert abs(parts["pool_magnitude"]) < 1e-12
    assert abs(parts["weight_composition"] - (-10.0)) < 1e-12
    assert abs(parts["timing_membership"]) < 1e-12
    assert abs(parts["integer_rounding"]) < 1e-12


def test_decompose_sticky_timing_only():
    """Control never reallocates: actual synth ≠ LR(implicit pool)."""
    control = {
        "synthetic_share": 30,
        "painted_pool": 40,
        "weight_share": 0.5,
        "exact_proportional_share": 20.0,
        "largest_remainder_share": 20,
        "last_membership_event": "sell",
    }
    treatment = {
        "synthetic_share": 20,
        "painted_pool": 40,
        "weight_share": 0.5,
        "exact_proportional_share": 20.0,
        "largest_remainder_share": 20,
        "last_membership_event": "sell",
    }
    parts = decompose_synth_pair(control, treatment)
    assert abs(parts["pool_magnitude"]) < 1e-12
    assert abs(parts["weight_composition"]) < 1e-12
    assert abs(parts["timing_membership"] - (-10.0)) < 1e-12
    assert abs(parts["integer_rounding"]) < 1e-12
    assert abs(parts["residual"]) < 1e-12


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
        tracer = AllocationInputTracer(0, seed, "obs")
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
    hits = [
        f for f in tracer.fights
        if int(f.get("applied_hp_loss") or 0) > 0
    ]
    assert hits
    saw_paint = False
    saw_event = False
    for f in hits:
        rows = f.get("start_minions") or []
        if not rows:
            continue
        assert f.get("shares_sum_to_painted_pool") is True
        saw_paint = True
        for r in rows:
            assert "painted_pool" in r
            assert "paint_weight" in r
            assert "board_recruit_denom" in r
            assert "weight_share" in r
            assert "exact_proportional_share" in r
            assert "largest_remainder_share" in r
            assert "rounding_contribution" in r
            assert "last_membership_event" in r
            assert "pre_reallocation_synth" in r
            assert "post_reallocation_synth" in r
            if r.get("last_membership_event") in ("sell", "play", "triple"):
                saw_event = True
        assert f["counterfactual_damage"] == rules_faithful_hero_damage(
            f["winner_tavern_tier"],
            [s["tier"] for s in f["actual_survivors"]],
        )
    assert saw_paint
    assert saw_event


def test_diagnose_routes():
    def _bag(pool, weight, timing, rounding=0.0, residual=None):
        if residual is None:
            residual = 1.0 - (pool + weight + timing + rounding)
        return {
            "primary": {
                "share_of_delta_pool_magnitude": pool,
                "share_of_delta_weight_composition": weight,
                "share_of_delta_timing_membership": timing,
                "share_of_delta_integer_rounding": rounding,
                "share_of_delta_residual": residual,
            },
        }

    pool = diagnose_phase_3p(_bag(0.80, 0.05, 0.05))
    assert pool["primary_finding"] == "pool_magnitude_dominates"
    assert "player pool diverges" in pool["recommended_next_step"]
    wgt = diagnose_phase_3p(_bag(0.05, 0.80, 0.05))
    assert wgt["primary_finding"] == "weight_composition_dominates"
    assert "recruit-raw-proportional" in wgt["recommended_next_step"]
    tim = diagnose_phase_3p(_bag(0.05, 0.05, 0.80))
    assert tim["primary_finding"] == "timing_membership_dominates"
    assert "sell/play" in tim["recommended_next_step"]
    rnd = diagnose_phase_3p(_bag(0.05, 0.05, 0.05, rounding=0.80, residual=0.05))
    assert rnd["primary_finding"] == "rounding_material"
    leftover = diagnose_phase_3p(_bag(0.20, 0.20, 0.20, rounding=0.10))
    assert leftover["primary_finding"] == "ranked_residual_needs_next_observable"
    smoke = diagnose_phase_3p(pool, non_evaluative=True)
    assert smoke["primary_finding"] == "measurement_smoke_non_evaluative"
    assert smoke["evaluative"] is False
