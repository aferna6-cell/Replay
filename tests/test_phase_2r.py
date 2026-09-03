"""Tests for Phase 2R replacement churn / combat-loss diagnostic."""

import json
from pathlib import Path

from hsbg_coach.bg_env import (
    A_BUY0,
    A_END,
    A_PLAY0,
    A_ROLL,
    A_SELL0,
    PHASE_2Q_RECRUIT_VALUE_STATS,
    EnvMinion,
)
from ml.replacement_churn_diagnostic import (
    CHURN_EXPLAINS_FRACTION,
    FORBIDDEN_RANGES,
    METHODOLOGY_VERSION,
    PHASE_2R_SEED,
    ReplacementChurnTracer,
    assert_seed_range_allowed,
    compare_control_treatment,
    diagnose_phase_2r,
    recompute_churn_explains_t10,
    run_greedy_control,
    run_greedy_treatment,
    summarize_churn_arm,
)

_PHASE_2R_DIR = Path("results/sim_fidelity_phase_2r")
# Published 2r_v1 greedy identity (independent recompute must match).
_PUBLISHED_CHURN_FRAC_T10 = 0.9938216477777939


class _FakePlayer:
    def __init__(self, board, alive=True):
        self.board = list(board)
        self.alive = alive

    def strength(self):
        return sum(int(m.attack) + int(m.health) for m in self.board)


def _view(name, attack, health, recruit_attack=None, recruit_health=None, golden=False):
    return EnvMinion(
        "id-" + name, name, 3, attack, health, [], [], golden,
        recruit_attack, recruit_health,
    ).view()


def _minion(name, attack, health, recruit_attack=None, recruit_health=None, golden=False):
    return EnvMinion(
        "id-" + name, name, 3, attack, health, [], [], golden,
        recruit_attack, recruit_health,
    )


def _drive(tracer, seat, turn, action, obs, ended=False, player=None):
    tracer.before_action(seat, turn, 0, obs, [])
    tracer.after_action(seat, turn, 0, action, ended, player=player)


def test_methodology_is_2r_v1():
    assert METHODOLOGY_VERSION == "2r_v1"


def test_phase_2q_toggle_still_defaults_off():
    assert PHASE_2Q_RECRUIT_VALUE_STATS is False


def test_phase_2r_forbidden_ranges_cover_confirm_and_priors():
    assert (11500, 11699) in FORBIDDEN_RANGES
    assert (13200, 13699) in FORBIDDEN_RANGES  # 2Q consumed
    assert (12700, 13199) in FORBIDDEN_RANGES


def test_phase_2r_seed_guard():
    try:
        assert_seed_range_allowed(11500, 10)
        raise AssertionError("expected overlap failure")
    except ValueError:
        pass
    try:
        assert_seed_range_allowed(13200, 10)
        raise AssertionError("expected 2Q overlap failure")
    except ValueError:
        pass
    assert_seed_range_allowed(PHASE_2R_SEED, 500)


def test_diagnose_routes_churn_explains():
    greedy_cmp = {
        "deltas": {
            "full_board_replace_rate": 0.27,
            "sum_combat_strength_removed": 5.0e6,
            "mean_combat_loss_per_replacement": 40.0,
            "post_scale_over_firestone_t10": -0.48,
            "mean_game_length": -2.5,
            "treatment_post_stats_deficit_t10": 80.0,
            "excess_mean_net_loss_t10": 55.0,
            "cumulative_excess_net_loss_t8_t10": 90.0,
            "churn_explains_fraction_t10": 0.70,
        }
    }
    d = diagnose_phase_2r(greedy_cmp)
    assert d["primary_finding"] == "replacement_churn_loss_explains_macro_collapse"
    assert d["keep_pr_29_hold"] is True
    assert d["keep_pr_33_hold"] is True
    assert d["feature_toggle_default_off"] is True
    assert d["churn_explains_threshold"] == CHURN_EXPLAINS_FRACTION


def test_diagnose_routes_residual_coupling():
    greedy_cmp = {
        "deltas": {
            "full_board_replace_rate": 0.27,
            "sum_combat_strength_removed": 1.0e5,
            "mean_combat_loss_per_replacement": 10.0,
            "post_scale_over_firestone_t10": -0.48,
            "mean_game_length": -2.5,
            "treatment_post_stats_deficit_t10": 80.0,
            "excess_mean_net_loss_t10": 10.0,
            "cumulative_excess_net_loss_t8_t10": 15.0,
            "churn_explains_fraction_t10": 0.12,
        }
    }
    d = diagnose_phase_2r(greedy_cmp)
    assert d["primary_finding"] == "residual_or_pace_coupling_dominates"


def test_greedy_smoke_two_lobbies_instruments_replacements():
    raw_c = run_greedy_control(2, PHASE_2R_SEED)
    raw_t = run_greedy_treatment(2, PHASE_2R_SEED)
    c = summarize_churn_arm(raw_c)
    t = summarize_churn_arm(raw_t)
    assert c["recruit_value_stats"] is False
    assert t["recruit_value_stats"] is True
    assert "per_turn_decomposition" in c
    assert "8" in c["per_turn_decomposition"]
    assert "14" in c["per_turn_decomposition"]
    assert "replacement_loss_distribution" in c
    assert "post_scale_firestone_ratios" in c
    assert "alive_curve_t8_t14" in c
    cmp = compare_control_treatment(c, t)
    assert "paired_post_scale_firestone_ratios" in cmp
    assert "paired_alive_curve" in cmp
    assert "per_turn_decomposition_delta" in cmp
    assert "churn_explains_fraction_t10" in (cmp.get("deltas") or {})
    # Treatment should complete at least as many replacements on tiny sample
    # in the typical direction (not a hard gate — just sanity on keys).
    assert c["n_completed_replacements"] is not None
    assert t["n_completed_replacements"] is not None
    d = diagnose_phase_2r(cmp)
    assert d["primary_finding"] in {
        "replacement_churn_loss_explains_macro_collapse",
        "residual_or_pace_coupling_dominates",
        "churn_up_without_macro_collapse",
        "inconclusive",
    }
    # Toggle must remain OFF after arm contexts exit.
    assert PHASE_2Q_RECRUIT_VALUE_STATS is False


def test_independent_recompute_churn_fraction_from_committed_artifacts():
    """99.38% must survive a from-scratch T8–T10 identity on committed tables."""
    pt = json.loads((_PHASE_2R_DIR / "per_turn_decomposition_greedy.json").read_text())
    decision = json.loads((_PHASE_2R_DIR / "decision.json").read_text())

    # Hand-rolled identity — do not call compare_control_treatment here.
    cum = 0.0
    for t in ("8", "9", "10"):
        cum += (
            float(pt["treatment"][t]["mean_net_after_residual"])
            - float(pt["control"][t]["mean_net_after_residual"])
        )
    deficit = (
        float(pt["control"]["10"]["mean_post_scaling_stats"])
        - float(pt["treatment"]["10"]["mean_post_scaling_stats"])
    )
    frac = cum / deficit
    assert abs(frac - _PUBLISHED_CHURN_FRAC_T10) < 1e-12
    assert abs(frac - float(decision["churn_explains_fraction_t10"])) < 1e-12
    assert frac >= CHURN_EXPLAINS_FRACTION

    helper = recompute_churn_explains_t10(pt["control"], pt["treatment"])
    assert abs(helper["churn_explains_fraction_t10"] - frac) < 1e-12
    # Greedy T10 has zero replacements; same-turn slice is residual undershoot.
    assert pt["control"]["10"]["n_replacements"] == 0
    assert pt["treatment"]["10"]["n_replacements"] == 0
    assert helper["churn_explains_fraction_t10_same_turn"] < 0.40

    j = json.loads((_PHASE_2R_DIR / "per_turn_decomposition_phase_2j.json").read_text())
    j_helper = recompute_churn_explains_t10(j["control"], j["treatment"])
    assert abs(j_helper["churn_explains_fraction_t10"] - 1.0037738635001332) < 1e-12


def test_committed_example_events_obey_loss_identities():
    raw = json.loads(
        (_PHASE_2R_DIR / "example_replacement_events_greedy.json").read_text()
    )
    n = 0
    for arm in ("control", "treatment"):
        for ev in raw[arm]:
            n += 1
            assert ev["combat_strength_loss"] == (
                ev["incumbent_combat_raw"] - ev["candidate_combat_raw"]
            )
            assert ev["recruit_value_gain"] == (
                ev["candidate_recruit_raw"] - ev["incumbent_recruit_raw"]
            )
            if ev.get("net_board_combat_delta") is not None:
                assert ev["net_board_combat_delta"] == -ev["combat_strength_loss"]
    assert n >= 20


def test_no_replacement_abandoned_sell_is_not_counted():
    tracer = ReplacementChurnTracer(0, 13700, "qa")
    board = [_view(f"B{i}", 20, 20, 2, 2) for i in range(7)]
    obs = {"board": board, "shop": [_view("Shop", 5, 5, 5, 5)], "hand": []}
    tracer.begin_seat_recruit(0, 9, _FakePlayer([]))
    _drive(tracer, 0, 9, A_SELL0, obs)
    assert 0 in tracer._pending
    # Roll abandons the incomplete sell — no completed replacement.
    _drive(tracer, 0, 9, A_ROLL, {"board": board[1:], "shop": [], "hand": []})
    assert tracer.replacement_events == []
    assert tracer.incomplete_abandons == 1
    assert 0 not in tracer._pending
    # End-of-recruit with a leftover pending also abandons, not completes.
    _drive(tracer, 0, 9, A_SELL0, obs)
    tracer.end_seat_recruit(0, 9, _FakePlayer([]))
    assert tracer.replacement_events == []
    assert tracer.incomplete_abandons == 2


def test_known_loss_sell_play_matches_combat_minus_candidate():
    tracer = ReplacementChurnTracer(0, 13700, "qa")
    inc = _view("Incumbent", 40, 30, 4, 3)          # combat 70, recruit 7
    cand = _view("Candidate", 5, 5, 5, 5)           # combat 10, recruit 10
    fillers = [_view(f"F{i}", 10, 10, 2, 2) for i in range(6)]
    board = [inc] + fillers
    hand = [cand]
    after = [_minion("Candidate", 5, 5, 5, 5)] + [
        _minion(f"F{i}", 10, 10, 2, 2) for i in range(6)
    ]
    tracer.begin_seat_recruit(0, 9, _FakePlayer([]))
    _drive(tracer, 0, 9, A_SELL0, {"board": board, "shop": [], "hand": hand})
    _drive(
        tracer, 0, 9, A_PLAY0,
        {"board": fillers, "shop": [], "hand": hand},
        player=_FakePlayer(after),
    )
    assert len(tracer.replacement_events) == 1
    ev = tracer.replacement_events[0]
    assert ev["source"] == "hand"
    assert ev["combat_strength_loss"] == 60.0
    assert ev["recruit_value_gain"] == 3.0
    assert ev["incumbent_golden"] is False
    acc = tracer._acc(0, 9)
    assert acc["replacements"] == 1
    assert acc["combat_removed"] == 60.0


def test_known_loss_golden_incumbent_uses_doubled_recruit():
    tracer = ReplacementChurnTracer(0, 13700, "qa")
    inc = _view("GoldenInc", 80, 80, 8, 8, golden=True)  # combat 160, recruit 16
    cand = _view("Plain", 6, 6, 6, 6)
    fillers = [_view(f"F{i}", 8, 8, 3, 3) for i in range(6)]
    after = [_minion("Plain", 6, 6, 6, 6)] + [
        _minion(f"F{i}", 8, 8, 3, 3) for i in range(6)
    ]
    tracer.begin_seat_recruit(0, 11, _FakePlayer([]))
    _drive(tracer, 0, 11, A_SELL0, {"board": [inc] + fillers, "shop": [], "hand": [cand]})
    _drive(
        tracer, 0, 11, A_PLAY0,
        {"board": fillers, "shop": [], "hand": [cand]},
        player=_FakePlayer(after),
    )
    ev = tracer.replacement_events[0]
    assert ev["incumbent_golden"] is True
    assert ev["combat_strength_loss"] == 148.0
    assert ev["recruit_value_gain"] == -4.0
    assert ev["incumbent_recruit_raw"] == 16.0


def test_no_double_count_sell_buy_play():
    tracer = ReplacementChurnTracer(0, 13700, "qa")
    inc = _view("Inc", 50, 50, 5, 5)
    shop_c = _view("ShopCand", 8, 8, 8, 8)
    fillers = [_view(f"F{i}", 9, 9, 3, 3) for i in range(6)]
    after = [_minion("ShopCand", 8, 8, 8, 8)] + [
        _minion(f"F{i}", 9, 9, 3, 3) for i in range(6)
    ]
    tracer.begin_seat_recruit(0, 9, _FakePlayer([]))
    _drive(tracer, 0, 9, A_SELL0, {
        "board": [inc] + fillers, "shop": [shop_c], "hand": [],
    })
    # Buy records the candidate; must not complete yet.
    _drive(tracer, 0, 9, A_BUY0, {
        "board": fillers, "shop": [shop_c], "hand": [],
    })
    assert tracer.replacement_events == []
    assert tracer._pending[0]["source"] == "shop"
    # Second buy must not create a second candidate / second event.
    other = _view("OtherShop", 20, 20, 20, 20)
    _drive(tracer, 0, 9, A_BUY0 + 1, {
        "board": fillers, "shop": [shop_c, other], "hand": [shop_c],
    })
    assert tracer.replacement_events == []
    assert tracer._pending[0]["candidate"]["name"] == "ShopCand"
    _drive(
        tracer, 0, 9, A_PLAY0,
        {"board": fillers, "shop": [other], "hand": [shop_c]},
        player=_FakePlayer(after),
    )
    assert len(tracer.replacement_events) == 1
    ev = tracer.replacement_events[0]
    assert ev["source"] == "shop"
    assert ev["combat_strength_loss"] == 84.0  # 100 - 16
    assert ev["recruit_value_gain"] == 6.0     # 16 - 10
    assert tracer._acc(0, 9)["replacements"] == 1


def test_no_double_count_two_completed_replacements_same_turn():
    tracer = ReplacementChurnTracer(0, 13700, "qa")
    tracer.begin_seat_recruit(0, 9, _FakePlayer([]))
    for i, (inc_atk, cand_atk) in enumerate(((40, 5), (30, 6))):
        inc = _view(f"Inc{i}", inc_atk, inc_atk, 4, 4)
        cand = _view(f"Cand{i}", cand_atk, cand_atk, cand_atk, cand_atk)
        fillers = [_view(f"F{i}{j}", 8, 8, 2, 2) for j in range(6)]
        after = [_minion(f"Cand{i}", cand_atk, cand_atk, cand_atk, cand_atk)] + [
            _minion(f"F{i}{j}", 8, 8, 2, 2) for j in range(6)
        ]
        _drive(tracer, 0, 9, A_SELL0, {
            "board": [inc] + fillers, "shop": [], "hand": [cand],
        })
        _drive(
            tracer, 0, 9, A_PLAY0,
            {"board": fillers, "shop": [], "hand": [cand]},
            player=_FakePlayer(after),
        )
    assert len(tracer.replacement_events) == 2
    assert tracer._acc(0, 9)["replacements"] == 2
    # 80-10 + 60-12 = 118; one event each, no third phantom.
    assert tracer._acc(0, 9)["combat_removed"] == 118.0


def test_residual_recovery_net_is_removed_minus_residual():
    """Positive net_after_residual ⇒ residual did not recover combat removed."""
    tracer = ReplacementChurnTracer(0, 13700, "qa")
    # Do not begin_seat_recruit: that would enqueue a ScalingBudgetTracer pending
    # and overwrite the injected residual_add on after_scale_all.
    acc = tracer._acc(0, 9)
    acc["replacements"] = 2
    acc["combat_removed"] = 96.0
    acc["recruit_gain"] = 8.0
    # Inject a scaling record as after_scale_all would read it.
    tracer.scaling.records.append({
        "lobby": 0, "seat": 0, "turn": 9,
        "residual_add": 40.0,
        "scaling_delta": 40.0,
        "start_of_recruit_stats": 200.0,
        "end_of_recruit_pre_scaling_stats": 104.0,
        "post_scaling_stats": 144.0,
        "recruit_delta": -96.0,
        "firestone_target": 400.0,
        "post_scale_over_firestone": 0.36,
        "pre_scale_over_firestone": 0.26,
    })

    class _Env:
        turn = 9
        players = [_FakePlayer([_minion("A", 20, 20, 5, 5)])]

    tracer.after_scale_all(_Env())
    row = tracer.turn_rows[0]
    assert row["net_after_residual"] == 56.0  # 96 - 40
    assert abs(row["residual_recovery_ratio"] - 40.0 / 96.0) < 1e-12


def test_next_turn_carry_links_post_scale_to_next_strength():
    tracer = ReplacementChurnTracer(0, 13700, "qa")
    tracer._prev_post_scale[0] = {"turn": 9, "next_turn_carried_strength": None}
    nxt = _FakePlayer([_minion("Carry", 30, 40, 4, 4)])
    tracer.begin_seat_recruit(0, 10, nxt)
    assert tracer._prev_post_scale[0]["next_turn_carried_strength"] == 70.0
    assert tracer._prev_post_scale[0]["next_turn_alive"] is True
    # Wrong-turn prev row must not be overwritten.
    tracer._prev_post_scale[1] = {"turn": 8, "next_turn_carried_strength": None}
    tracer.begin_seat_recruit(1, 10, nxt)
    assert tracer._prev_post_scale[1]["next_turn_carried_strength"] is None
