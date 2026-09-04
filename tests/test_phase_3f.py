"""Phase 3F carry divergence timing — observational locks."""

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
from ml.phase_3f_prereg import (
    FEATURE_TOGGLE,
    FEATURE_TOGGLE_DEFAULT,
    FORBIDDEN_RANGES,
    FROZEN_ALPHA,
    HISTORY_LINK_IDENTITY,
    HOLD_PRS,
    IMPACT_ATTACK_IDENTITY,
    INSTRUMENT_TURNS,
    LOW_WINNER_START_TIERS,
    MATERIAL_ABS,
    MATERIAL_REL,
    METHODOLOGY_VERSION,
    PHASE_3D_BOARD_POOL_MAGNITUDE,
    PHASE_3E_CARRY_DELTA,
    PHASE_3E_CARRY_SHARE_OF_A1,
    PHASE_3E_PUNCH_DELTA_CARRY,
    PHASE_3F_LOBBIES,
    PHASE_3F_SEED,
    POOL_FLOW_IDENTITY,
    REUSED_SEED_HI,
    REUSED_SEED_LO,
    SHARE_DOMINANT,
    assert_seed_range_allowed,
    carry_value,
    diagnose_phase_3f,
    first_separation_turn,
    materially_separated,
    share_of_carry_term,
)
from ml.carry_divergence_diagnostic import (
    _delta_by_turn,
    build_seat_trajectories,
    compare_divergence,
    pair_trajectories,
    reconcile_history_links,
)
from ml.pool_lifecycle_diagnostic import PoolLifecycleTracer
from ml.survivor_tier_damage_diagnostic import rules_faithful_hero_damage


def test_methodology_is_3f_v1_default_off():
    assert METHODOLOGY_VERSION == "3f_v1"
    assert FEATURE_TOGGLE == "PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING"
    assert FEATURE_TOGGLE_DEFAULT is False
    assert PHASE_2Q_RECRUIT_VALUE_STATS is False
    assert PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING is False
    assert SHARE_DOMINANT == 0.70
    assert MATERIAL_ABS == 8.0
    assert MATERIAL_REL == 0.10
    assert LOW_WINNER_START_TIERS == (1,)
    assert POOL_FLOW_IDENTITY == (
        "post = pre + add - represented_loss_or_transfer"
    )
    assert IMPACT_ATTACK_IDENTITY == (
        "impact_attack = start_recruit + start_pool_share + combat_delta"
    )
    assert HISTORY_LINK_IDENTITY == (
        "punch.opp_carry = seat_history[turn].attack_pool_recruit_start"
    )
    assert carry_value({"attack_pool_recruit_start": 40}) == 40.0
    assert carry_value({"opp_carry_attack_pool": 12}) == 12.0


def test_reuses_2s_dev_and_forbids_confirm():
    assert PHASE_3F_SEED == 14200
    assert PHASE_3F_LOBBIES == 500
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


def test_hold_stack_includes_3e_and_frozen_alpha():
    assert HOLD_PRS == (
        29, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 50, 51,
    )
    assert FROZEN_ALPHA == 0.5
    assert abs(PHASE_3D_BOARD_POOL_MAGNITUDE - 0.4216721428553852) < 1e-9
    assert abs(A1_3E - PHASE_3D_BOARD_POOL_MAGNITUDE) < 1e-12
    assert abs(PHASE_3E_CARRY_DELTA - 0.30513688784757187) < 1e-9
    assert abs(PHASE_3E_CARRY_SHARE_OF_A1 - 0.7236353954551374) < 1e-9
    assert abs(PHASE_3E_PUNCH_DELTA_CARRY - (-196.33317557443002)) < 1e-9
    d = diagnose_phase_3f()
    assert d["no_merge"] is True
    assert d["no_hero_damage_retune"] is True
    assert d["no_behavior_change"] is True
    assert d["no_2q_rewrite"] is True
    assert d["no_scaling_constant_change"] is True
    assert d["keep_hold_prs"][-1] == 51
    assert d["history_link_identity"] == HISTORY_LINK_IDENTITY


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


def test_material_separation_and_carry_share():
    assert materially_separated(100.0, 100.0) is False
    assert materially_separated(36.0, 35.8) is False
    assert materially_separated(810.0, 614.0) is True
    assert materially_separated(None, 10.0) is False
    sep = first_separation_turn({
        7: (36.0, 35.8),
        8: (80.0, 79.0),
        9: (200.0, 40.0),
        10: (400.0, 50.0),
    })
    assert sep == 9
    assert abs(share_of_carry_term(-196.33317557443002) - 1.0) < 1e-9
    assert share_of_carry_term(0.0) == 0.0
    assert share_of_carry_term(None) is None


def _play(seed, with_hook):
    env = BGEnv(seed=seed)
    tracer = None
    if with_hook:
        tracer = PoolLifecycleTracer(0, seed, "obs")
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
    traj = build_seat_trajectories(tracer.turn_rows, tracer.fights)
    assert traj
    hist = reconcile_history_links(tracer.fights, tracer.turn_rows)
    if hist["n_punch_rows"] > 0:
        assert hist["n_carry_mismatch"] == 0
        assert hist["n_missing_turn_row"] == 0
        assert hist["p_ok"] == 1.0
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


def _seat_turn(seed, seat, turn, carry, *, alive=True, add=0.0, tier=3):
    return {
        "seed": seed,
        "seat": seat,
        "turn": turn,
        "arm": "obs",
        "attack_pool_recruit_start": carry,
        "scale_add_attack": add,
        "alive_at_recruit": alive,
        "alive_at_combat": alive,
        "tier_at_recruit": tier,
        "board_size_post_scale": 7,
        "mean_tier_recruit_start": float(tier),
        "n_alive": 6,
        "flow_ok": True,
        "flow_residual": 0.0,
    }


def _fight(
    seed, turn, winner, loser, *, applied=4, start_tier=1, damaging=1,
    carry=100.0, add=20.0,
):
    return {
        "seed": seed,
        "lobby": 0,
        "turn": turn,
        "kind": "live",
        "ghost": False,
        "seat_a": winner,
        "seat_b": loser,
        "winner_seat": winner,
        "loser_seat": loser,
        "applied_hp_loss": applied,
        "winner_tavern_tier": 4,
        "fight_outcome": "a_win",
        "outcome": "win",
        "start_minions": [{
            "tier": start_tier,
            "n_damaging_hits": damaging,
            "n_hits": damaging,
            "survived": False,
            "recruit_raw": 4,
            "synthetic_share": 2,
            "opp_carry_attack_pool": carry,
            "opp_scale_add_attack": add,
            "opp_attack_pool_recruit_start": carry,
        }],
    }


def test_pair_trajectories_date_divergence_and_condition():
    """Synthetic arms: uncond Δ≈0; punch+low-tier crater appears after filters."""
    control_turns = []
    treat_turns = []
    control_fights = []
    treat_fights = []
    # Eight paired seats, carry matched unconditionally (~100 both arms).
    for seat in range(8):
        for turn in INSTRUMENT_TURNS:
            control_turns.append(_seat_turn(14200, seat, turn, 100.0 + turn))
            treat_turns.append(_seat_turn(14200, seat, turn, 101.0 + turn))
    # Seats 0–1 later appear as punch-opp vs a T1 winner-start; treatment
    # crater only on those seats at the punch turn.
    for seat in (0, 1):
        control_fights.append(_fight(14200, 10, winner=3, loser=seat, start_tier=1, carry=110.0))
        treat_fights.append(_fight(14200, 10, winner=3, loser=seat, start_tier=1, carry=20.0))
        # overwrite T10 carry so the pair actually diverges
        for row in treat_turns:
            if row["seat"] == seat and row["turn"] == 10:
                row["attack_pool_recruit_start"] = 20.0
        for row in control_turns:
            if row["seat"] == seat and row["turn"] == 10:
                row["attack_pool_recruit_start"] = 110.0

    c_traj = build_seat_trajectories(control_turns, control_fights)
    t_traj = build_seat_trajectories(treat_turns, treat_fights)
    assert c_traj[(14200, 0)]["later_punch_included"] is True
    assert c_traj[(14200, 0)]["low_winner_start"] is True
    assert c_traj[(14200, 0)]["first_punch_turn"] == 10
    assert c_traj[(14200, 5)]["later_punch_included"] is False

    pairs = pair_trajectories(c_traj, t_traj)
    assert len(pairs) == 8
    punch_pairs = [p for p in pairs if p["later_punch_included"]]
    assert len(punch_pairs) == 2
    assert punch_pairs[0]["first_separation_turn"] == 10

    uncond = _delta_by_turn(pairs, stage="unconditional")
    punch = _delta_by_turn(pairs, stage="punch_included")
    low = _delta_by_turn(pairs, stage="low_winner_start")
    # T7 still matched; T10 punch-conditioned crater is large.
    assert abs(uncond["7"]["delta_treatment_minus_control"]) < 2.0
    assert punch["10"]["n_pairs"] == 2
    assert punch["10"]["delta_treatment_minus_control"] < -50.0
    assert low["10"]["n_pairs"] == 2

    hist_c = reconcile_history_links(control_fights, control_turns)
    assert hist_c["n_punch_rows"] == 2
    assert hist_c["n_ok"] == 2
    assert hist_c["n_carry_mismatch"] == 0
    assert hist_c["n_history_gap"] == 0


def test_compare_divergence_routes_selection_on_synthetic_crater():
    control = {
        "turn_rows": [
            _seat_turn(14200, 0, t, 800.0 if t == 10 else 100.0)
            for t in INSTRUMENT_TURNS
        ] + [
            _seat_turn(14200, 1, t, 100.0) for t in INSTRUMENT_TURNS
        ],
        "fights": [_fight(14200, 10, winner=1, loser=0, start_tier=1, carry=800.0)],
    }
    treatment = {
        "turn_rows": [
            _seat_turn(14200, 0, t, 50.0 if t == 10 else 100.0)
            for t in INSTRUMENT_TURNS
        ] + [
            _seat_turn(14200, 1, t, 100.0) for t in INSTRUMENT_TURNS
        ],
        "fights": [_fight(14200, 10, winner=1, loser=0, start_tier=1, carry=50.0)],
    }
    cmp = compare_divergence(control, treatment)
    timing = cmp["timing"]
    # Unconditional pooled Δ is diluted by the matched seat-turns.
    assert abs(timing["share_of_3e_carry_unconditional"] or 0.0) < 0.40
    assert (timing["share_of_3e_carry_punch_included"] or 0.0) > 0.70
    assert (timing["share_of_3e_carry_from_selection"] or 0.0) > 0.70
    decision = diagnose_phase_3f(cmp)
    assert decision["primary_finding"] == "selection_outcome_conditioning_dominates"
    assert "selection" in decision["recommended_next_step"]


def test_diagnose_routes_paired_mixed_and_leftover():
    paired = diagnose_phase_3f({
        "timing": {
            "share_of_3e_carry_unconditional": 0.80,
            "share_of_3e_carry_punch_included": 0.85,
            "share_of_3e_carry_low_winner_start": 0.90,
            "share_of_3e_carry_outcome_conditioned": 0.90,
            "share_of_3e_carry_before_conditioning": 0.80,
            "share_of_3e_carry_from_selection": 0.10,
        }
    })
    assert paired["primary_finding"] == "paired_divergence_precedes_conditioning"
    assert "upstream scaling" in paired["recommended_next_step"]

    sel = diagnose_phase_3f({
        "timing": {
            "share_of_3e_carry_unconditional": 0.05,
            "share_of_3e_carry_punch_included": 0.20,
            "share_of_3e_carry_low_winner_start": 0.85,
            "share_of_3e_carry_outcome_conditioned": 0.90,
            "share_of_3e_carry_before_conditioning": 0.05,
            "share_of_3e_carry_from_selection": 0.85,
        }
    })
    assert sel["primary_finding"] == "selection_outcome_conditioning_dominates"
    assert "selection" in sel["recommended_next_step"]

    mixed = diagnose_phase_3f({
        "timing": {
            "share_of_3e_carry_unconditional": 0.40,
            "share_of_3e_carry_punch_included": 0.55,
            "share_of_3e_carry_low_winner_start": 0.60,
            "share_of_3e_carry_outcome_conditioned": 0.60,
            "share_of_3e_carry_before_conditioning": 0.40,
            "share_of_3e_carry_from_selection": 0.20,
        }
    })
    assert mixed["primary_finding"] == "mixed_route_to_larger"
    assert "larger" in mixed["recommended_next_step"]

    leftover = diagnose_phase_3f({
        "timing": {
            "share_of_3e_carry_unconditional": 0.10,
            "share_of_3e_carry_punch_included": 0.12,
            "share_of_3e_carry_low_winner_start": 0.15,
            "share_of_3e_carry_outcome_conditioned": 0.15,
            "share_of_3e_carry_before_conditioning": 0.10,
            "share_of_3e_carry_from_selection": 0.05,
        }
    })
    assert leftover["primary_finding"] == "ranked_residual_needs_next_observable"

    smoke = diagnose_phase_3f(paired, non_evaluative=True)
    assert smoke["primary_finding"] == "measurement_smoke_non_evaluative"
    assert smoke["evaluative"] is False
