"""Phase 2X synthetic-allocation vs within-tier survival — observational locks."""

import random

from hsbg_coach.bg_env import (
    PHASE_2Q_RECRUIT_VALUE_STATS,
    PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING,
    BGEnv,
    EnvMinion,
    greedy_policy,
    minion_synthetic_delta,
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
from ml.phase_2x_prereg import (
    FEATURE_TOGGLE,
    FEATURE_TOGGLE_DEFAULT,
    FORBIDDEN_RANGES,
    FROZEN_ALPHA,
    HOLD_PRS,
    INSTRUMENT_TURNS,
    METHODOLOGY_VERSION,
    PHASE_2V_SHARE_WITHIN_TIER,
    PHASE_2V_WITHIN_TIER_B,
    PHASE_2X_LOBBIES,
    PHASE_2X_SEED,
    REUSED_SEED_HI,
    REUSED_SEED_LO,
    SHARE_SYNTHETIC_DOMINANT,
    assert_seed_range_allowed,
    diagnose_phase_2x,
)
from ml.synthetic_allocation_diagnostic import (
    SyntheticAllocationTracer,
    classify_start_minion,
    largest_remainder_shares,
    reweight_within_tier,
)
from ml.survivor_tier_damage_diagnostic import rules_faithful_hero_damage


def test_methodology_is_2x_v1_default_off():
    assert METHODOLOGY_VERSION == "2x_v1"
    assert FEATURE_TOGGLE == "PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING"
    assert FEATURE_TOGGLE_DEFAULT is False
    assert PHASE_2Q_RECRUIT_VALUE_STATS is False
    assert PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING is False
    assert SHARE_SYNTHETIC_DOMINANT == 0.70


def test_reuses_2s_dev_and_forbids_confirm():
    assert PHASE_2X_SEED == 14200
    assert PHASE_2X_LOBBIES == 500
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
    try:
        assert_seed_range_allowed(14199, 2)
        raise AssertionError("expected pre-2S band to fail")
    except ValueError:
        pass


def test_hold_stack_includes_2w_and_frozen_alpha():
    assert HOLD_PRS == (29, 33, 34, 35, 36, 37, 38, 39, 40, 41)
    assert FROZEN_ALPHA == 0.5
    assert abs(PHASE_2V_WITHIN_TIER_B - 1.6782901818400895) < 1e-9
    assert abs(PHASE_2V_SHARE_WITHIN_TIER - 0.4185551426754372) < 1e-9
    d = diagnose_phase_2x()
    assert d["no_merge"] is True
    assert d["no_hero_damage_retune"] is True
    assert d["no_behavior_change"] is True
    assert d["no_2q_rewrite"] is True
    assert d["keep_hold_prs"] == [29, 33, 34, 35, 36, 37, 38, 39, 40, 41]


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


def test_per_minion_synthetic_shares_sum_to_player_pool():
    """Largest-remainder paint: Σ (combat − recruit) == round(abstract_pool)."""
    class _P:
        def __init__(self):
            self.abstract_pool = 100.0
            self.board = [
                EnvMinion("a", "A", 4, 4, 6, [], [], False, 4, 6),
                EnvMinion("b", "B", 5, 8, 8, [], [], False, 8, 8),
                EnvMinion("c", "C", 3, 2, 2, [], [], False, 2, 2),
            ]

    p = _P()
    reallocate_abstract_pool(p)
    shares = [minion_synthetic_delta(m) for m in p.board]
    assert sum(shares) == int(round(p.abstract_pool))
    expected = largest_remainder_shares(
        [int(m.recruit_attack) + int(m.recruit_health) for m in p.board],
        int(round(p.abstract_pool)),
    )
    assert shares == expected


def test_classify_start_minion_records_required_fields():
    body = {
        "name": "Zebra",
        "card_id": "Z",
        "body_id": "a:start:2",
        "tier": 5,
        "golden": True,
        "board_slot": 2,
        "recruit_raw": 10,
        "combat_raw": 40,
        "attacked": True,
        "n_attacks": 1,
    }
    row = classify_start_minion(body, 2, survived=False)
    assert row["tier"] == 5
    assert row["recruit_raw"] == 10
    assert row["synthetic_share"] == 30
    assert row["combat_raw"] == 40
    assert abs(row["synthetic_share_of_combat"] - 0.75) < 1e-9
    assert row["board_slot"] == 2
    assert row["golden"] is True
    assert row["survived"] is False
    assert row["died"] is True
    assert row["attacked"] is True
    assert row["attacked_before_death"] is True


def test_trace_records_attacked_without_changing_rng_or_damage():
    a = [Combatant(5, 5, name="a", tier=3), Combatant(3, 4, name="b", tier=2)]
    b = [Combatant(4, 4, name="c", tier=4)]
    r1 = random.Random(42)
    r2 = random.Random(42)
    raw1 = simulate_once(a, b, r1, tier_a=3, tier_b=2)
    trace = {}
    raw2 = simulate_once(a, b, r2, tier_a=3, tier_b=2, trace=trace)
    assert raw1 == raw2
    assert r1.getstate() == r2.getstate()
    start = list(trace.get("starting_a") or []) + list(trace.get("starting_b") or [])
    assert start
    assert all("attacked" in row and "n_attacks" in row for row in start)
    # The first attacker on the larger (or RNG-tied) side must have swung.
    assert any(row.get("attacked") for row in start)


def _play(seed, with_hook):
    env = BGEnv(seed=seed)
    tracer = None
    if with_hook:
        tracer = SyntheticAllocationTracer(0, seed, "obs")
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


def test_combat_hook_is_observational_same_seed():
    """Hooked vs unhooked: placements, HP, length, and RNG state match."""
    plain = _play(14200, False)
    hooked = _play(14200, True)
    assert hooked["length"] == plain["length"]
    assert hooked["placements"] == plain["placements"]
    assert hooked["hp"] == plain["hp"]
    assert hooked["rng_state"] == plain["rng_state"]
    assert hooked["n_fights"] > 0
    hits = [
        f for f in hooked["tracer"].fights
        if int(f.get("applied_hp_loss") or 0) > 0
    ]
    assert hits
    for f in hits:
        assert f.get("shares_sum_to_pool") is True
        rows = f.get("start_minions") or []
        assert sum(int(r["synthetic_share"]) for r in rows) == int(
            f.get("winner_player_pool") or 0
        )
        assert f["actual_survivor_count"] == f["survivor_count"]
        assert f["counterfactual_damage"] == rules_faithful_hero_damage(
            f["winner_tavern_tier"],
            [s["tier"] for s in f["actual_survivors"]],
        )


def _row(tier, recruit, synth, survived, slot=0, attacked=False):
    combat = recruit + synth
    return {
        "tier": tier,
        "recruit_raw": recruit,
        "synthetic_share": synth,
        "combat_raw": combat,
        "synthetic_share_of_combat": (
            synth / combat if combat else None
        ),
        "survived": survived,
        "died": not survived,
        "board_slot": slot,
        "golden": False,
        "attacked": attacked,
        "n_attacks": 1 if attacked else 0,
    }


def test_reweight_assigns_extra_synthetic_not_position():
    """Same tier + recruit; treatment only adds synthetic and survives more."""
    control = [
        _row(4, 10, 0, True),
        _row(4, 10, 0, False),
    ]
    treatment = [
        _row(4, 10, 20, True),
        _row(4, 10, 20, True),
    ]
    rw = reweight_within_tier(
        control, treatment, n_hits_c=1, n_hits_t=1, observed_B=4.0
    )
    assert rw["within_tier_B"] > 0
    assert rw["share_of_B_synthetic"] is not None
    assert rw["share_of_B_synthetic"] > 0.70
    assert (rw["share_of_B_residual_position"] or 0.0) < 0.20
    assert abs(
        rw["recruit_mix"] + rw["synthetic_allocation"] + rw["residual_position"]
        - rw["within_tier_B"]
    ) < 1e-9


def test_reweight_assigns_residual_when_synth_and_recruit_match():
    """Same tier / recruit / synth; only survival (slot/order) differs."""
    control = [
        _row(4, 10, 10, True, slot=0, attacked=True),
        _row(4, 10, 10, False, slot=1, attacked=False),
        _row(4, 10, 10, False, slot=2, attacked=False),
        _row(4, 10, 10, False, slot=3, attacked=False),
    ]
    treatment = [
        _row(4, 10, 10, True, slot=0, attacked=True),
        _row(4, 10, 10, True, slot=1, attacked=True),
        _row(4, 10, 10, True, slot=2, attacked=True),
        _row(4, 10, 10, False, slot=3, attacked=False),
    ]
    rw = reweight_within_tier(
        control, treatment, n_hits_c=1, n_hits_t=1
    )
    assert rw["within_tier_B"] > 0
    assert rw["share_of_B_residual_position"] is not None
    assert rw["share_of_B_residual_position"] > 0.70
    assert (rw["share_of_B_synthetic"] or 0.0) < 0.20
    assert abs(
        rw["recruit_mix"] + rw["synthetic_allocation"] + rw["residual_position"]
        - rw["within_tier_B"]
    ) < 1e-9


def test_diagnose_routes_three_ways():
    synth = diagnose_phase_2x({
        "reweighting": {
            "share_of_B_synthetic": 0.80,
            "share_of_B_residual_position": 0.10,
            "share_of_B_recruit_mix": 0.10,
            "within_tier_B": 1.678,
        }
    })
    assert synth["primary_finding"] == "synthetic_allocation_dominates"
    assert "allocation rules" in synth["recommended_next_step"]

    pos = diagnose_phase_2x({
        "reweighting": {
            "share_of_B_synthetic": 0.15,
            "share_of_B_residual_position": 0.75,
            "share_of_B_recruit_mix": 0.10,
            "within_tier_B": 1.678,
        }
    })
    assert pos["primary_finding"] == "position_combat_order_dominates"
    assert "combat fidelity" in pos["recommended_next_step"]

    mixed = diagnose_phase_2x({
        "reweighting": {
            "share_of_B_synthetic": 0.40,
            "share_of_B_residual_position": 0.35,
            "share_of_B_recruit_mix": 0.25,
            "within_tier_B": 1.678,
        }
    })
    assert mixed["primary_finding"] == "mixed_or_missing_within_tier_feature"

    smoke = diagnose_phase_2x(synth, non_evaluative=True)
    assert smoke["primary_finding"] == "measurement_smoke_non_evaluative"
    assert smoke["evaluative"] is False
