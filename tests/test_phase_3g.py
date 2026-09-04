"""Phase 3G punch-sample selection decomposition — observational locks."""

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
from ml.phase_3g_prereg import (
    EARLY_TURNS,
    FEATURE_TOGGLE,
    FEATURE_TOGGLE_DEFAULT,
    FORBIDDEN_RANGES,
    FROZEN_ALPHA,
    HISTORY_LINK_IDENTITY,
    HOLD_PRS,
    IMPACT_ATTACK_IDENTITY,
    INSTRUMENT_TURNS,
    LOW_WINNER_START_TIERS,
    METHODOLOGY_VERSION,
    PHASE_3D_BOARD_POOL_MAGNITUDE,
    PHASE_3E_CARRY_DELTA,
    PHASE_3E_CARRY_SHARE_OF_A1,
    PHASE_3E_PUNCH_DELTA_CARRY,
    PHASE_3F_SELECTION_SHARE,
    PHASE_3F_UNCOND_SHARE,
    PHASE_3G_LOBBIES,
    PHASE_3G_SEED,
    POOL_FLOW_IDENTITY,
    REUSED_SEED_HI,
    REUSED_SEED_LO,
    SHARE_DOMINANT,
    WEIGHT_RECONCILIATION_IDENTITY,
    _alive_bin,
    _role_bin,
    assert_seed_range_allowed,
    diagnose_phase_3g,
    share_of_crater,
)
from ml.carry_divergence_diagnostic import (
    build_seat_trajectories,
    reconcile_history_links,
)
from ml.pool_lifecycle_diagnostic import PoolLifecycleTracer
from ml.punch_selection_diagnostic import (
    collect_punch_sample_rows,
    compare_selection,
    decompose_punch_selection,
    kitagawa_mean_delta,
)
from ml.survivor_tier_damage_diagnostic import rules_faithful_hero_damage


def test_methodology_is_3g_v1_default_off():
    assert METHODOLOGY_VERSION == "3g_v1"
    assert FEATURE_TOGGLE == "PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING"
    assert FEATURE_TOGGLE_DEFAULT is False
    assert PHASE_2Q_RECRUIT_VALUE_STATS is False
    assert PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING is False
    assert SHARE_DOMINANT == 0.70
    assert LOW_WINNER_START_TIERS == (1, 2, 3)
    assert EARLY_TURNS == (7, 8, 9)
    assert POOL_FLOW_IDENTITY == (
        "post = pre + add - represented_loss_or_transfer"
    )
    assert IMPACT_ATTACK_IDENTITY == (
        "impact_attack = start_recruit + start_pool_share + combat_delta"
    )
    assert HISTORY_LINK_IDENTITY == (
        "punch.opp_carry = seat_history[turn].attack_pool_recruit_start"
    )
    assert "sum_k n_arm(k) = N_arm" in WEIGHT_RECONCILIATION_IDENTITY
    assert _alive_bin(2) == "alive_2_3"
    assert _alive_bin(5) == "alive_4_5"
    assert _alive_bin(8) == "alive_6_plus"
    assert _role_bin(2) == "winner_tavern_low"
    assert _role_bin(5) == "winner_tavern_high"


def test_reuses_2s_dev_and_forbids_confirm():
    assert PHASE_3G_SEED == 14200
    assert PHASE_3G_LOBBIES == 500
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


def test_hold_stack_includes_3f_and_frozen_alpha():
    assert HOLD_PRS == (
        29, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
        50, 51, 52,
    )
    assert FROZEN_ALPHA == 0.5
    assert abs(PHASE_3D_BOARD_POOL_MAGNITUDE - 0.4216721428553852) < 1e-9
    assert abs(A1_3E - PHASE_3D_BOARD_POOL_MAGNITUDE) < 1e-12
    assert abs(PHASE_3E_CARRY_DELTA - 0.30513688784757187) < 1e-9
    assert abs(PHASE_3E_CARRY_SHARE_OF_A1 - 0.7236353954551374) < 1e-9
    assert abs(PHASE_3E_PUNCH_DELTA_CARRY - (-196.33317557443002)) < 1e-9
    assert abs(PHASE_3F_UNCOND_SHARE - 0.09084015396406948) < 1e-9
    assert abs(PHASE_3F_SELECTION_SHARE - 0.9091598460359305) < 1e-9
    d = diagnose_phase_3g()
    assert d["no_merge"] is True
    assert d["no_hero_damage_retune"] is True
    assert d["no_behavior_change"] is True
    assert d["no_2q_rewrite"] is True
    assert d["no_scaling_constant_change"] is True
    assert d["keep_hold_prs"][-1] == 52
    assert d["history_link_identity"] == HISTORY_LINK_IDENTITY
    assert d["weight_reconciliation_identity"] == WEIGHT_RECONCILIATION_IDENTITY


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


def test_share_of_crater_signed():
    assert abs(share_of_crater(-196.33317557443002) - 1.0) < 1e-9
    assert abs(share_of_crater(196.33317557443002) - (-1.0)) < 1e-9
    assert share_of_crater(0.0) == 0.0
    assert share_of_crater(None) is None


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
    rows = collect_punch_sample_rows(tracer.fights)
    assert rows
    for r in rows:
        assert r["turn"] in INSTRUMENT_TURNS
        assert r["winner_start_tier"] >= 1
        assert "carry" in r
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


def _row(
    turn, tier, carry, *, n_alive=6, winner_tavern=2, damaging=True,
):
    return {
        "turn": turn,
        "winner_start_tier": tier,
        "tier": tier,
        "carry": float(carry),
        "n_alive": n_alive,
        "opp_n_alive": n_alive,
        "opp_alive": True,
        "winner_tavern_tier": winner_tavern,
        "alive_bin": _alive_bin(n_alive),
        "role_bin": _role_bin(winner_tavern),
        "damaging": damaging,
        "opp_carry_attack_pool": float(carry),
        "attack_pool_recruit_start": float(carry),
    }


def test_kitagawa_reconciles_mixture_and_rate():
    """Equal cell means → all Δ is mixture; equal weights → all Δ is rate."""
    # Mixture only: same within-cell carry, treatment over-weights the low cell.
    control = [_row(7, 1, 80.0) for _ in range(50)] + [
        _row(10, 5, 800.0) for _ in range(50)
    ]
    treat = [_row(7, 1, 80.0) for _ in range(90)] + [
        _row(10, 5, 800.0) for _ in range(10)
    ]
    mix_only = kitagawa_mean_delta(
        control, treat, lambda r: (r["turn"], r["winner_start_tier"]),
    )
    assert abs(mix_only["reconciliation_gap"] or 0.0) < 1e-9
    assert abs(mix_only["rate"]) < 1e-9
    assert mix_only["mixture"] < -1.0

    # Rate only: matched weights, treatment cell means crater.
    control = [_row(7, 1, 800.0) for _ in range(40)] + [
        _row(10, 1, 800.0) for _ in range(40)
    ]
    treat = [_row(7, 1, 80.0) for _ in range(40)] + [
        _row(10, 1, 80.0) for _ in range(40)
    ]
    rate_only = kitagawa_mean_delta(
        control, treat, lambda r: (r["turn"], r["winner_start_tier"]),
    )
    assert abs(rate_only["reconciliation_gap"] or 0.0) < 1e-9
    assert abs(rate_only["mixture"]) < 1e-9
    assert abs(rate_only["rate"] - (-720.0)) < 1e-9


def test_decompose_routes_mixture_on_weight_shift():
    """Treatment piles punch rows onto an early T1 cell with matched means."""
    control = (
        [_row(7, 1, 100.0) for _ in range(20)]
        + [_row(12, 5, 900.0) for _ in range(80)]
    )
    treat = (
        [_row(7, 1, 100.0) for _ in range(80)]
        + [_row(12, 5, 900.0) for _ in range(20)]
    )
    decomp = decompose_punch_selection(control, treat, unpaired_punch=-196.0)
    assert decomp["reconciliation"]["reconciliation_ok"] is True
    assert decomp["reconciliation"]["counts_match_control"] is True
    assert decomp["reconciliation"]["weights_sum_to_one_control"] is True
    assert (decomp["share_mixture_turn_winner_tier"] or 0.0) > 0.70
    assert abs(decomp["share_within_cell_opponent_carry"] or 0.0) < 0.05
    decision = diagnose_phase_3g({"decomposition": decomp})
    assert decision["primary_finding"] == "mixture_role_selection_dominates"
    assert "winner-tier" in decision["recommended_next_step"]


def test_decompose_routes_within_cell_on_matched_weights():
    """Matched turn×tier weights; treatment carry craters inside the cell."""
    control = [_row(10, 1, 800.0) for _ in range(100)]
    treat = [_row(10, 1, 50.0) for _ in range(100)]
    decomp = decompose_punch_selection(control, treat, unpaired_punch=-196.0)
    assert decomp["reconciliation"]["reconciliation_ok"] is True
    assert (decomp["share_within_cell_opponent_carry"] or 0.0) > 0.70
    assert abs(decomp["share_mixture_turn_winner_tier"] or 0.0) < 0.05
    decision = diagnose_phase_3g({"decomposition": decomp})
    assert decision["primary_finding"] == "within_cell_opponent_carry_dominates"
    assert "within-cell" in decision["recommended_next_step"]


def test_nested_role_alive_splits_within_cell():
    """Same turn×tier mix; treatment selects collapsed lobbies with low carry."""
    control = (
        [_row(10, 1, 800.0, n_alive=7, winner_tavern=5) for _ in range(50)]
        + [_row(10, 1, 200.0, n_alive=2, winner_tavern=2) for _ in range(50)]
    )
    treat = (
        [_row(10, 1, 800.0, n_alive=7, winner_tavern=5) for _ in range(10)]
        + [_row(10, 1, 200.0, n_alive=2, winner_tavern=2) for _ in range(90)]
    )
    decomp = decompose_punch_selection(control, treat, unpaired_punch=-196.0)
    assert decomp["reconciliation"]["reconciliation_ok"] is True
    # Outer weights match (all rows are T10×T1), so (2) is the full Δ.
    assert abs(decomp["share_mixture_turn_winner_tier"] or 0.0) < 0.05
    assert (decomp["share_within_cell_opponent_carry"] or 0.0) > 0.70
    # Nested role/alive mix should take most of that within-cell gap.
    assert (decomp["share_role_alive_selection"] or 0.0) > 0.50
    assert decomp["low_tier_early"]["verdict"] in (
        "true_within_cell_pool_deficit",
        "disproportionately_early_low_carry",
        "mixed_early_and_within_cell",
    )


def test_low_tier_early_flags_disproportionate_early_rows():
    """Treatment T1 rows sit on T7 (low carry) vs control T12 (high carry)."""
    control = [_row(12, 1, 900.0) for _ in range(80)]
    treat = [_row(7, 1, 80.0) for _ in range(80)]
    decomp = decompose_punch_selection(control, treat, unpaired_punch=-196.0)
    early = decomp["low_tier_early"]
    assert early["verdict"] == "disproportionately_early_low_carry"
    assert early["treatment"]["p_early"] is not None
    assert early["control"]["p_early"] is not None
    assert early["treatment"]["p_early"] > 0.9
    assert early["control"]["p_early"] < 0.1


def test_compare_selection_history_link_and_ghost_skip():
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
        seed, turn, winner, loser, *, applied=4, start_tier=1,
        carry=100.0, add=20.0, ghost=False,
    ):
        return {
            "seed": seed,
            "lobby": 0,
            "turn": turn,
            "kind": "ghost" if ghost else "live",
            "ghost": ghost,
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
                "n_damaging_hits": 1,
                "n_hits": 1,
                "survived": False,
                "recruit_raw": 4,
                "synthetic_share": 2,
                "opp_carry_attack_pool": carry,
                "opp_scale_add_attack": add,
                "opp_attack_pool_recruit_start": carry,
                "opp_n_alive": 6,
                "opp_alive": True,
            }],
        }

    control = {
        "turn_rows": [
            _seat_turn(
                14200, 0, t, 800.0 if t == 10 else 100.0,
                add=20.0 if t == 10 else 0.0,
            )
            for t in INSTRUMENT_TURNS
        ],
        "fights": [_fight(14200, 10, winner=1, loser=0, start_tier=1, carry=800.0)],
    }
    treatment = {
        "turn_rows": [
            _seat_turn(
                14200, 0, t, 50.0 if t == 10 else 100.0,
                add=20.0 if t == 10 else 0.0,
            )
            for t in INSTRUMENT_TURNS
        ],
        "fights": [_fight(14200, 10, winner=1, loser=0, start_tier=1, carry=50.0)],
    }
    cmp = compare_selection(control, treatment)
    decomp = cmp["decomposition"]
    assert decomp["n_control"] == 1
    assert decomp["n_treatment"] == 1
    assert abs((decomp["observed_delta"] or 0.0) - (-750.0)) < 1e-9
    assert cmp["reconciliation"]["history_link_control"]["n_ok"] == 1
    assert cmp["reconciliation"]["history_link_treatment"]["n_ok"] == 1
    ghost = _fight(14200, 11, winner=1, loser=0, start_tier=1, carry=0.0, ghost=True)
    hist_g = reconcile_history_links([ghost], control["turn_rows"])
    assert hist_g["n_punch_rows"] == 0
    assert hist_g["n_skipped_ghost_or_no_loser"] == 1


def test_diagnose_routes_paired_mixed_and_leftover():
    mix = diagnose_phase_3g({
        "decomposition": {
            "share_mixture_turn_winner_tier": 0.80,
            "share_within_cell_opponent_carry": 0.10,
            "share_role_alive_selection": 0.05,
            "share_leftover": 0.05,
            "share_mixture_plus_role": 0.85,
        }
    })
    assert mix["primary_finding"] == "mixture_role_selection_dominates"
    assert "composition" in mix["recommended_next_step"]

    cell = diagnose_phase_3g({
        "decomposition": {
            "share_mixture_turn_winner_tier": 0.05,
            "share_within_cell_opponent_carry": 0.85,
            "share_role_alive_selection": 0.08,
            "share_leftover": 0.02,
            "share_mixture_plus_role": 0.13,
        }
    })
    assert cell["primary_finding"] == "within_cell_opponent_carry_dominates"
    assert "within-cell" in cell["recommended_next_step"]

    mixed = diagnose_phase_3g({
        "decomposition": {
            "share_mixture_turn_winner_tier": 0.40,
            "share_within_cell_opponent_carry": 0.45,
            "share_role_alive_selection": 0.05,
            "share_leftover": 0.10,
            "share_mixture_plus_role": 0.45,
        }
    })
    assert mixed["primary_finding"] == "mixed_route_to_larger"
    assert "larger" in mixed["recommended_next_step"]

    leftover = diagnose_phase_3g({
        "decomposition": {
            "share_mixture_turn_winner_tier": 0.10,
            "share_within_cell_opponent_carry": 0.12,
            "share_role_alive_selection": 0.05,
            "share_leftover": 0.73,
            "share_mixture_plus_role": 0.15,
        }
    })
    assert leftover["primary_finding"] == "ranked_residual_needs_next_observable"

    smoke = diagnose_phase_3g(mix, non_evaluative=True)
    assert smoke["primary_finding"] == "measurement_smoke_non_evaluative"
    assert smoke["evaluative"] is False
