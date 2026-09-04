"""Phase 2W Firestone composition vs 2Q selection — join/weight/route locks."""

from hsbg_coach.bg_env import (
    PHASE_2Q_RECRUIT_VALUE_STATS,
    PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING,
    BGEnv,
    EnvMinion,
    greedy_policy,
)
from ml.firestone_composition_reference import (
    base_card_id,
    build_firestone_reference,
    is_golden_card_id,
    join_minion,
    load_joined_boards,
    load_lookup,
    resolve_card,
)
from ml.phase_2s_prereg import (
    GATE_GAME_LENGTH_DELTA_FLOOR,
    GATE_MEAN_COMBAT_LOSS_MAX,
    GATE_REPLACE_RATE_MIN,
    GATE_T10_POST_SCALE_DELTA_FLOOR,
    GATE_T10_POST_SCALE_MIN,
)
from ml.phase_2w_diagnostic import (
    FirestoneBoardTracer,
    classify_sim_minion,
)
from ml.phase_2w_prereg import (
    COVERAGE_JOIN_MIN,
    COVERAGE_N_BOARDS_MIN,
    FEATURE_TOGGLE,
    FEATURE_TOGGLE_DEFAULT,
    FORBIDDEN_RANGES,
    FROZEN_ALPHA,
    HIGH_TIER_MIN,
    HOLD_PRS,
    LATE_TURNS,
    METHODOLOGY_VERSION,
    PHASE_2W_LOBBIES,
    PHASE_2W_SEED,
    REUSED_SEED_HI,
    REUSED_SEED_LO,
    assert_seed_range_allowed,
    diagnose_phase_2w,
)


def test_methodology_is_2w_v1_default_off():
    assert METHODOLOGY_VERSION == "2w_v1"
    assert FEATURE_TOGGLE == "PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING"
    assert FEATURE_TOGGLE_DEFAULT is False
    assert PHASE_2Q_RECRUIT_VALUE_STATS is False
    assert PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING is False
    assert HIGH_TIER_MIN == 4
    assert LATE_TURNS == (12, 13, 14)


def test_reuses_2s_dev_and_forbids_confirm():
    assert PHASE_2W_SEED == 14200
    assert PHASE_2W_LOBBIES == 500
    assert REUSED_SEED_LO == 14200
    assert REUSED_SEED_HI == 14699
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


def test_hold_stack_includes_2v_and_frozen_alpha():
    assert HOLD_PRS == (29, 33, 34, 35, 36, 37, 38, 39, 40)
    assert FROZEN_ALPHA == 0.5
    d = diagnose_phase_2w()
    assert d["no_merge"] is True
    assert d["no_hero_damage_retune"] is True
    assert d["no_behavior_change"] is True
    assert d["firestone_is_final_board_data"] is True
    assert d["keep_hold_prs"] == [29, 33, 34, 35, 36, 37, 38, 39, 40]


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


def test_golden_card_id_join_strips_suffix():
    assert is_golden_card_id("BG36_851_G") is True
    assert is_golden_card_id("BG26_148") is False
    assert base_card_id("BG36_851_G") == "BG36_851"
    lookup = load_lookup()
    ck, golden, path = resolve_card("BG36_851_G", "Spark Snapper",
                                    lookup["kb_id"], lookup["kb_name"])
    assert ck is not None
    assert golden is True
    assert path in ("base_card_id", "name", "card_id")
    assert ck.tier is not None
    row = join_minion(
        {"cardId": "BG36_851_G", "name": "Spark Snapper", "atk": 10, "health": 10},
        lookup,
    )
    assert row["joined"] is True
    assert row["golden"] is True
    assert row["printed_tier"] == ck.tier
    assert row["printed_raw"] == float((ck.attack + ck.health) * 2)


def test_plain_card_id_and_name_fallback_join():
    lookup = load_lookup()
    row = join_minion(
        {"cardId": "BG26_148", "name": "Scrap Scraper", "atk": 4, "health": 4,
         "tribe": "Mech"},
        lookup,
    )
    assert row["joined"] is True
    assert row["golden"] is False
    assert row["printed_tier"] >= 1
    assert row["resolve_path"] in ("card_id", "base_card_id", "name")
    missing = join_minion(
        {"cardId": "NOT_A_REAL_CARD_ID", "name": "Scrap Scraper"},
        lookup,
    )
    assert missing["joined"] is True
    assert missing["resolve_path"] == "name"


def test_firestone_join_and_weight_reconciliation():
    joined = load_joined_boards()
    ref = build_firestone_reference()
    rec = ref["reconciliation"]
    assert rec["n_minions"] == joined["n_minions"]
    assert rec["join_plus_unresolved"] == rec["n_minions"]
    assert rec["n_unresolved"] == 0
    assert abs(rec["weight_delta"]) < 1e-6
    assert rec["sum_board_count"] == rec["sum_example_weight"]
    assert ref["coverage"]["join_rate"] == 1.0
    assert ref["coverage"]["n_example_boards"] >= COVERAGE_N_BOARDS_MIN
    assert ref["coverage"]["join_rate"] >= COVERAGE_JOIN_MIN
    assert ref["weighted"]["weight_reconcile"] is True
    w = ref["weighted"]
    u = ref["unweighted"]
    # Weighting must change the mix when popular archetypes differ from rare ones.
    assert w["n_example_boards"] == u["n_example_boards"] == 57
    assert w["mean_printed_tier"] is not None
    assert 0.0 <= w["t4_plus_share"] <= 1.0
    assert 0.0 <= w["golden_share"] <= 1.0
    hist = w["tier_histogram"]
    assert abs(sum(hist.values()) - 1.0) < 1e-6
    assert ref["meta"]["is_final_board_data"] is True


def test_classify_sim_minion_uses_printed_not_combat():
    lookup = load_lookup()
    m = EnvMinion("BG26_148", "Scrap Scraper", 4, 80, 80, ["Mech"], [], False, 4, 5)
    row = classify_sim_minion(m, lookup, 0)
    assert row["joined"] is True
    assert row["printed_tier"] == 4 or row["printed_tier"] >= 1
    assert row["combat_raw"] == 160
    assert row["recruit_raw"] == 9
    assert row["printed_raw"] < row["combat_raw"]


def _play(seed, with_hook):
    env = BGEnv(seed=seed)
    tracer = None
    if with_hook:
        tracer = FirestoneBoardTracer(0, seed, "obs", load_lookup())
    recs = env.play_scripted(
        [greedy_policy] * env.n_players,
        recruit_tracer=tracer,
    )
    return {
        "length": max((r["turn"] for r in recs), default=0),
        "placements": [p.placement for p in env.players],
        "hp": [p.hp for p in env.players],
        "rng_state": env.rng.getstate(),
        "n_last": len(tracer.last_boards) if tracer else 0,
        "tracer": tracer,
    }


def test_board_tracer_is_observational_same_seed():
    plain = _play(14200, False)
    hooked = _play(14200, True)
    assert hooked["length"] == plain["length"]
    assert hooked["placements"] == plain["placements"]
    assert hooked["hp"] == plain["hp"]
    assert hooked["rng_state"] == plain["rng_state"]
    assert hooked["n_last"] == 8
    assert all(b.get("last_turn") for b in hooked["tracer"].last_boards)


def test_diagnose_routes_three_ways():
    coverage = {
        "join_rate": 1.0,
        "n_example_boards": 57,
        "n_unique_joined_cards": 80,
        "pool_name_rate": 0.96,
    }
    high = diagnose_phase_2w({
        "coverage": coverage,
        "last_alive": {
            "t4_share_treatment_minus_firestone": 0.15,
            "t4_share_treatment_minus_control": 0.20,
            "mean_printed_tier_treatment_minus_firestone": 0.40,
            "mean_printed_tier_treatment_minus_control": 0.50,
            "mean_printed_raw_treatment_minus_firestone": 3.0,
            "mean_printed_raw_treatment_minus_control": 4.0,
        },
    })
    assert high["primary_finding"] == "treatment_high_tier_raw_vs_control_and_firestone"
    assert "2Q" in high["recommended_next_step"]

    match = diagnose_phase_2w({
        "coverage": coverage,
        "last_alive": {
            "t4_share_treatment_minus_firestone": 0.02,
            "t4_share_treatment_minus_control": 0.25,
            "mean_printed_tier_treatment_minus_firestone": -0.05,
            "mean_printed_tier_treatment_minus_control": 0.80,
            "mean_printed_raw_treatment_minus_firestone": 0.4,
            "mean_printed_raw_treatment_minus_control": 3.0,
        },
    })
    assert match["primary_finding"] == "treatment_matches_firestone"
    assert "combat/scaling" in match["recommended_next_step"]

    weak = diagnose_phase_2w({
        "coverage": {
            "join_rate": 0.40,
            "n_example_boards": 5,
            "n_unique_joined_cards": 8,
            "pool_name_rate": 0.20,
        },
        "last_alive": {},
    })
    assert weak["primary_finding"] == "firestone_coverage_too_weak"

    smoke = diagnose_phase_2w(high, non_evaluative=True)
    assert smoke["primary_finding"] == "measurement_smoke_non_evaluative"
    assert smoke["evaluative"] is False
