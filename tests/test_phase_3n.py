"""Phase 3N first-split matched-state damage — observational locks."""

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
from ml.phase_3n_prereg import (
    APPLIED_RECONCILE_IDENTITY,
    COUNTERFACTUAL_IDENTITY,
    DAMAGE_COMPONENTS,
    FEATURE_TOGGLE,
    FEATURE_TOGGLE_DEFAULT,
    FIELD_VS_SURVIVAL_IDENTITY,
    FIRST_DIVERGENCE_COMPONENTS,
    FORBIDDEN_RANGES,
    FROZEN_ALPHA,
    HOLD_PRS,
    INSTRUMENT_TURNS,
    LATE_TURNS,
    METHODOLOGY_VERSION,
    PHASE_3D_BOARD_POOL_MAGNITUDE,
    PHASE_3G_MIXTURE,
    PHASE_3H_LEFTOVER,
    PHASE_3I_PAIRING_SCHEDULE,
    PHASE_3J_ELIGIBILITY,
    PHASE_3K_THIRD_PARTY,
    PHASE_3L_SAME_SEAT_EARLIER,
    PHASE_3M_CLASS1,
    PHASE_3M_SAME_OUTCOME_DAMAGE,
    PHASE_3M_SHARE_DAMAGE,
    PHASE_3N_LOBBIES,
    PHASE_3N_SEED,
    PROXY_ERROR_IDENTITY,
    REUSED_SEED_HI,
    REUSED_SEED_LO,
    ROW_DAMAGE_RECONCILE_IDENTITY,
    SHARE_DOMINANT,
    SOURCE_COMPONENTS,
    assert_seed_range_allowed,
    classify_row_reconcile,
    diagnose_phase_3n,
    share_of_applied,
)
from ml.elimination_chain_diagnostic import (
    attribute_elimination_chain,
    reconcile_eliminations,
    reconcile_hp_flow,
)
from ml.elimination_timing_diagnostic import attribute_elimination_timing
from ml.hp_divergence_diagnostic import attribute_first_divergence
from ml.matched_state_damage_diagnostic import (
    MatchedStateDamageTracer,
    attribute_matched_state_damage,
    compare_matched_state_damage,
    decompose_applied_row,
    extract_fight_state,
    slim_board,
)
from ml.pairing_who_wins_diagnostic import compare_pairing
from ml.punch_selection_diagnostic import (
    collect_punch_sample_rows,
    decompose_punch_selection,
)
from ml.survivor_tier_damage_diagnostic import rules_faithful_hero_damage
from tests.test_phase_3m import (
    _arm_bundle,
    _class1_base,
    _decision,
    _fight,
    _punch,
    _turn,
)


def test_methodology_is_3n_v1_default_off():
    assert METHODOLOGY_VERSION == "3n_v1"
    assert FEATURE_TOGGLE == "PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING"
    assert FEATURE_TOGGLE_DEFAULT is False
    assert PHASE_2Q_RECRUIT_VALUE_STATS is False
    assert PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING is False
    assert SHARE_DOMINANT == 0.70
    assert DAMAGE_COMPONENTS == (
        "winner_tavern_tier",
        "survivor_count",
        "survivor_composition",
        "proxy_formula_error",
        "residual",
    )
    assert SOURCE_COMPONENTS == (
        "pre_fight_board",
        "within_fight_survival",
        "proxy_formula_error",
        "residual",
    )
    assert FIRST_DIVERGENCE_COMPONENTS == (
        "prior_alive_set_or_pairing",
        "same_pairing_outcome_flip",
        "same_outcome_damage",
        "inherited_hp_carry",
        "unreconciled",
    )
    assert "winner_tavern_tier +" in APPLIED_RECONCILE_IDENTITY
    assert "winner_tavern_tier + sum" in COUNTERFACTUAL_IDENTITY
    assert "applied - counterfactual" in PROXY_ERROR_IDENTITY
    assert "pre_fight_board +" in FIELD_VS_SURVIVAL_IDENTITY
    assert "every 3M class-(3)" in ROW_DAMAGE_RECONCILE_IDENTITY


def test_reuses_2s_dev_and_forbids_confirm():
    assert PHASE_3N_SEED == 14200
    assert PHASE_3N_LOBBIES == 500
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


def test_hold_stack_includes_3m_and_frozen_alpha():
    assert HOLD_PRS == (
        29, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
        50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
    )
    assert FROZEN_ALPHA == 0.5
    assert abs(PHASE_3D_BOARD_POOL_MAGNITUDE - A1_3E) < 1e-12
    assert PHASE_3H_LEFTOVER == 7155
    assert PHASE_3I_PAIRING_SCHEDULE == 5952
    assert PHASE_3J_ELIGIBILITY == 5648
    assert PHASE_3K_THIRD_PARTY == 3701
    assert PHASE_3L_SAME_SEAT_EARLIER == 2082
    assert PHASE_3M_CLASS1 == 2082
    assert PHASE_3M_SAME_OUTCOME_DAMAGE == 1059
    assert abs(PHASE_3M_SHARE_DAMAGE - 1059 / 2082) < 1e-12
    d = diagnose_phase_3n()
    assert d["no_merge"] is True
    assert d["no_hero_damage_retune"] is True
    assert d["no_behavior_change"] is True
    assert d["no_2q_rewrite"] is True
    assert d["keep_hold_prs"][-1] == 59
    assert d["history_filters_applied"] is False


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


def test_rules_faithful_is_tavern_plus_survivor_tier_sum():
    assert rules_faithful_hero_damage(5, [6, 4, 1]) == 16
    assert rules_faithful_hero_damage(3, []) == 3


def test_decompose_applied_row_reconciles():
    control = {
        "winner_tavern_tier": 3,
        "survivor_count": 1,
        "survivor_tier_sum": 2,
        "applied": 5,
        "counterfactual": 5,
        "winner_start_tier_mean": 2.0,
    }
    treatment = {
        "winner_tavern_tier": 3,
        "survivor_count": 3,
        "survivor_tier_sum": 7,
        "applied": 10,
        "counterfactual": 10,
        "winner_start_tier_mean": 7 / 3,
    }
    d = decompose_applied_row(control, treatment)
    assert d["delta_applied"] == 5.0
    assert d["winner_tavern_tier"] == 0.0
    assert abs(d["survivor_count"] - 4.0) < 1e-12  # (3-1)*2
    assert abs(d["survivor_composition"] - 1.0) < 1e-12
    assert d["proxy_formula_error"] == 0.0
    assert abs(d["residual"]) < 1e-12
    assert d["five_way_ok"] is True
    assert abs(
        d["winner_tavern_tier"] + d["survivor_count"]
        + d["survivor_composition"] + d["proxy_formula_error"]
        + d["residual"] - d["delta_applied"]
    ) < 1e-12
    # Fielded mix uses n × start mean; survival is leftover of Δsum.
    assert abs(
        d["pre_fight_board"] + d["within_fight_survival"]
        + d["proxy_formula_error"] + d["residual"] - d["delta_applied"]
    ) < 1e-12


def test_decompose_routes_proxy_error():
    control = {
        "winner_tavern_tier": 4,
        "survivor_count": 2,
        "survivor_tier_sum": 8,
        "applied": 4,
        "counterfactual": 12,
        "winner_start_tier_mean": 4.0,
    }
    treatment = {
        "winner_tavern_tier": 4,
        "survivor_count": 2,
        "survivor_tier_sum": 8,
        "applied": 8,
        "counterfactual": 12,
        "winner_start_tier_mean": 4.0,
    }
    d = decompose_applied_row(control, treatment)
    assert d["delta_applied"] == 4.0
    assert d["winner_tavern_tier"] == 0.0
    assert d["survivor_count"] == 0.0
    assert d["survivor_composition"] == 0.0
    assert d["proxy_formula_error"] == 4.0
    assert abs(d["residual"]) < 1e-12


def test_share_and_classify_row_reconcile():
    assert abs(share_of_applied(70.0, denom=100.0) - 0.70) < 1e-9
    assert share_of_applied(None, denom=100.0) is None
    assert classify_row_reconcile(
        class3=True, both_fights=True, hp_flow_ok=True,
        cf_ok=True, five_way_ok=True,
    ) == "reconciled"
    assert classify_row_reconcile(class3=False, both_fights=True) == "unreconciled"
    assert classify_row_reconcile(
        class3=True, both_fights=True, hp_flow_ok=False,
    ) == "unreconciled"


def test_diagnose_routes():
    def _bag(pre, within, proxy, tavern=0.05, count=0.05, composition=0.05):
        residual = 1.0 - (tavern + count + composition + proxy)
        src_res = 1.0 - (pre + within + proxy)
        return {
            "attribution": {
                "share_winner_tavern_tier": tavern,
                "share_survivor_count": count,
                "share_survivor_composition": composition,
                "share_proxy_formula_error": proxy,
                "share_residual": residual,
            },
            "source": {
                "share_pre_fight_board": pre,
                "share_within_fight_survival": within,
                "share_proxy_formula_error": proxy,
                "share_residual": src_res,
            },
        }

    pre = diagnose_phase_3n(_bag(0.80, 0.10, 0.05))
    assert pre["primary_finding"] == "pre_fight_board_dominates"
    assert "T5/T6" in pre["recommended_next_step"]
    within = diagnose_phase_3n(_bag(0.10, 0.80, 0.05))
    assert within["primary_finding"] == "within_fight_survival_dominates"
    assert "combat mechanic" in within["recommended_next_step"]
    proxy = diagnose_phase_3n(_bag(0.10, 0.08, 0.76))
    assert proxy["primary_finding"] == "proxy_formula_error_dominates"
    assert "2U" in proxy["recommended_next_step"]
    mixed = diagnose_phase_3n(_bag(
        0.40, 0.25, 0.10, tavern=0.10, count=0.40, composition=0.35,
    ))
    assert mixed["primary_finding"] == "mixed_route_to_larger"
    leftover = diagnose_phase_3n(_bag(
        0.10, 0.10, 0.05, tavern=0.05, count=0.05, composition=0.05,
    ))
    # residual is large; no represented source ≥30%
    assert leftover["primary_finding"] == "ranked_residual_needs_next_observable"
    smoke = diagnose_phase_3n(mixed, non_evaluative=True)
    assert smoke["primary_finding"] == "measurement_smoke_non_evaluative"
    assert smoke["evaluative"] is False


def _play(seed, with_hook):
    env = BGEnv(seed=seed)
    tracer = None
    if with_hook:
        tracer = MatchedStateDamageTracer(0, seed, "obs")
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
        and int(f.get("turn") or 0) in INSTRUMENT_TURNS
    ]
    assert hits
    for f in hits:
        assert f.get("starting_a") is not None
        assert f.get("starting_winner") is not None
        assert f["counterfactual_damage"] == rules_faithful_hero_damage(
            f["winner_tavern_tier"],
            [s["tier"] for s in f["actual_survivors"]],
        )
        state = extract_fight_state(f, f.get("loser_seat"))
        assert state["cf_ok"] is True
        assert state["winner_board"] or state["survivors"]
        boards = slim_board(f.get("starting_a")) + slim_board(f.get("starting_b"))
        assert boards or state["winner_board"]


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


def _enrich_fight(fight, *, survivors=None, start=None, winner_tavern=4):
    survivors = list(survivors or [
        {"name": "s1", "card_id": "S1", "tier": 4},
        {"name": "s2", "card_id": "S2", "tier": 4},
    ])
    start = list(start or [
        {"name": "w1", "card_id": "W1", "tier": 4, "recruit_raw": 8, "combat_raw": 20},
        {"name": "w2", "card_id": "W2", "tier": 4, "recruit_raw": 8, "combat_raw": 20},
        {"name": "w3", "card_id": "W3", "tier": 4, "recruit_raw": 8, "combat_raw": 20},
    ])
    fight["actual_survivors"] = survivors
    fight["actual_survivor_count"] = len(survivors)
    fight["actual_survivor_tier_sum"] = sum(int(s["tier"]) for s in survivors)
    fight["start_board"] = start
    fight["start_minions"] = start
    fight["starting_winner"] = start
    fight["starting_a"] = start
    fight["starting_b"] = [
        {"name": "l1", "card_id": "L1", "tier": 2, "recruit_raw": 4, "combat_raw": 8},
    ]
    fight["winner_tavern_tier"] = winner_tavern
    fight["counterfactual_damage"] = rules_faithful_hero_damage(
        winner_tavern, [s["tier"] for s in survivors],
    )
    return fight


def test_attribute_class3_records_boards_and_reconciles():
    leftover, control, treatment = _class1_base(origin="damage")
    for f in control["fights"]:
        if int(f.get("turn") or 0) == 7:
            _enrich_fight(
                f,
                survivors=[{"name": "c", "card_id": "C", "tier": 2}],
                start=[
                    {"name": "c1", "card_id": "C1", "tier": 2},
                    {"name": "c2", "card_id": "C2", "tier": 2},
                ],
                winner_tavern=3,
            )
            f["applied_hp_loss"] = 4
    for f in treatment["fights"]:
        if int(f.get("turn") or 0) == 7:
            _enrich_fight(
                f,
                survivors=[
                    {"name": "t1", "card_id": "T1", "tier": 2},
                    {"name": "t2", "card_id": "T2", "tier": 3},
                    {"name": "t3", "card_id": "T3", "tier": 2},
                ],
                start=[
                    {"name": "t1", "card_id": "T1", "tier": 2},
                    {"name": "t2", "card_id": "T2", "tier": 3},
                    {"name": "t3", "card_id": "T3", "tier": 2},
                ],
                winner_tavern=3,
            )
            f["applied_hp_loss"] = 8
    chain = attribute_elimination_chain(
        control, treatment, leftover_rows=leftover, treatment_punch=[],
    )
    assert (chain["share_same_seat_earlier_elimination"] or 0.0) > 0.70
    first = attribute_first_divergence(
        control, treatment, leftover_rows=leftover, treatment_punch=[],
    )
    assert (first["share_same_outcome_damage"] or 0.0) > 0.70
    attr = attribute_matched_state_damage(
        control, treatment, leftover_rows=leftover, treatment_punch=[],
    )
    assert attr["n_same_outcome_damage"] >= 1
    assert attr["row_five_way_ok"] is True
    assert attr["reconciliation_ok"] is True
    examples = attr["examples"]
    assert examples
    rec = examples[0]
    assert rec["class"] == "same_outcome_damage"
    assert rec["row_class"] == "reconciled"
    assert rec["control_state"]["winner_board"]
    assert rec["treatment_state"]["winner_board"]
    assert rec["control_state"]["survivors"]
    assert rec["treatment_counterfactual"] == 3 + 7
    assert rec["control_counterfactual"] == 3 + 2
    decomp = rec["decomposition"]
    assert decomp["five_way_ok"] is True
    assert decomp["delta_applied"] == 4.0
    decision = diagnose_phase_3n({"attribution": attr, "source": {
        "share_pre_fight_board": attr.get("share_pre_fight_board_kitagawa"),
        "share_within_fight_survival": attr.get(
            "share_within_fight_survival_kitagawa"
        ),
        "share_proxy_formula_error": attr.get(
            "share_proxy_formula_error_kitagawa"
        ),
        "share_residual": 0.0,
    }})
    assert decision["evaluative"] is True
    assert decision["no_hero_damage_retune"] is True


def test_3g_mixture_lock_on_weight_shift_still_holds():
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
    assert abs(PHASE_3G_MIXTURE - (-196.52943934946725)) < 1e-9


def test_compare_matched_state_reproduces_3m_on_synthetic():
    leftover, control, treatment = _class1_base(origin="damage")
    from ml.elimination_timing_diagnostic import compare_elimination
    from ml.elimination_chain_diagnostic import compare_chain
    timing = compare_elimination(control, treatment)
    chain = compare_chain(control, treatment, timing=timing)
    cmp = compare_matched_state_damage(
        control, treatment, timing=timing, chain=chain,
    )
    assert cmp["attribution"]["reconciliation_ok"] is True
    assert cmp["reconciliation"]["applied_reconciliation_ok"] is True
    first = cmp["first_divergence_3m"]
    assert first["reconciliation_ok"] is True
    assert (first.get("share_same_outcome_damage") or 0.0) > 0.70
