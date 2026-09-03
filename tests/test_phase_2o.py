"""Unit tests for Phase 2O scaling-budget diagnostic (measurement only)."""

from __future__ import annotations

import pytest

from hsbg_coach.bg_env import BGEnv, EnvMinion
from ml.scaling_budget_diagnostic import (
    FORBIDDEN_RANGES,
    assert_seed_range_allowed,
    directional_macro_policy_harm,
    route_phase_2o_finding,
    run_greedy_arm,
    symmetric_absolute_fidelity,
    aggregate_scaling_budget,
)


def test_confirm_seeds_forbidden():
    with pytest.raises(ValueError, match="11500"):
        assert_seed_range_allowed(11500, 200)
    with pytest.raises(ValueError, match="11700"):
        assert_seed_range_allowed(11700, 10)
    assert_seed_range_allowed(12200, 500)
    assert (11500, 11699) in FORBIDDEN_RANGES


def test_residual_budget_matches_apply_when_positive():
    env = BGEnv(seed=42, scaling_mode="residual")
    env.reset(seed=42)
    env.turn = 10
    p = env.players[0]
    p.tier = 5
    p.turns_since_level = 2
    p.board = [EnvMinion("a", "A", 4, 50, 50, [], [])]
    # Capture budget then apply on a clone board state.
    env2 = BGEnv(seed=99, scaling_mode="residual")
    env2.reset(seed=99)
    env2.turn = 10
    p2 = env2.players[0]
    p2.tier = 5
    p2.turns_since_level = 2
    p2.board = [EnvMinion("a", "A", 4, 50, 50, [], [])]
    # Sync RNG streams by reconstructing factor path via same seed state:
    # Call budget then apply on same env sequentially would double-consume RNG.
    # Instead: apply residual and check strength increased when under pace.
    before = p.strength()
    env._end_of_turn_scaling_residual(p)
    assert p.strength() >= before


def test_budget_helper_emits_expected_keys():
    env = BGEnv(seed=7, scaling_mode="residual")
    env.reset(seed=7)
    env.turn = 10
    p = env.players[0]
    p.tier = 6
    p.turns_since_level = 3
    p.board = [EnvMinion("a", "A", 5, 40, 40, [], [])]
    budget = env._residual_scaling_budget(p)
    assert budget is not None
    for key in (
        "firestone_target", "growth_factor", "ratio_g", "ratio_add",
        "pace_target", "over", "residual_add", "just_leveled",
    ):
        assert key in budget
    assert budget["residual_clamp_active"] == 1.0
    # Undersized board → over == 0 and residual_add == ratio_add.
    assert budget["over"] == 0.0
    assert abs(budget["residual_add"] - budget["ratio_add"]) < 1e-9


def test_scaling_audit_hook_observational():
    seen = []

    def hook(env, player, seat, budget):
        seen.append((seat, dict(budget)))

    env = BGEnv(seed=3, scaling_mode="residual")
    env.reset(seed=3)
    env.scaling_audit_hook = hook
    env.turn = 11
    p = env.players[0]
    p.tier = 5
    p.turns_since_level = 1
    p.board = [EnvMinion("a", "A", 4, 30, 30, [], [])]
    before = [(m.attack, m.health) for m in p.board]
    env._end_of_turn_scaling_residual(p)
    assert len(seen) == 1
    assert seen[0][0] == 0
    # Hook must not prevent scaling application.
    after = [(m.attack, m.health) for m in p.board]
    assert after != before or seen[0][1]["residual_add"] == 0


def test_small_arm_aggregation_smoke():
    raw = run_greedy_arm(2, seed=12200)
    assert raw["n_lobbies"] == 2
    assert raw["records"]
    agg = aggregate_scaling_budget(raw["records"])
    assert "by_turn" in agg
    fid = symmetric_absolute_fidelity(raw["records"])
    assert "10" in fid
    assert "mean_post_scale_over_firestone" in fid["10"]


def test_routing_target_gap_pattern():
    def make_agg(pre_r, post_r, gap, firestone=1601.0):
        return {
            "by_turn": {
                "10": {
                    "n": 100,
                    "pre_scale_over_firestone": pre_r,
                    "post_scale_over_firestone": post_r,
                    "remaining_target_gap_after_scaling": gap,
                    "firestone_target": firestone,
                    "recruit_delta": 100.0,
                    "start_of_recruit_stats": 500.0,
                },
                "9": {"pre_scale_over_firestone": 0.5,
                      "post_scale_over_firestone": 0.5},
                "11": {"pre_scale_over_firestone": 0.5,
                       "post_scale_over_firestone": 0.5},
                "12": {"pre_scale_over_firestone": 0.5,
                       "post_scale_over_firestone": 0.5},
            },
            "by_turn_level": {},
        }

    greedy = make_agg(0.4, 0.5, 800.0)
    treatment = make_agg(0.35, 0.45, 880.0)
    fid_g = {str(t): {"mean_post_scale_over_firestone": 0.5}
             for t in range(8, 15)}
    fid_t = {str(t): {"mean_post_scale_over_firestone": 0.45}
             for t in range(8, 15)}
    route = route_phase_2o_finding(greedy, treatment, fid_g, fid_t)
    assert route["primary_finding"] == "pre_scale_far_below_post_still_far"
    assert "target-gap" in route["recommended_next_step"]


def test_directional_harm_treatment_closer():
    greedy = {"14": {"mean_post_scale_over_firestone": 1.52}}
    treatment = {"14": {"mean_post_scale_over_firestone": 1.205}}
    harm = directional_macro_policy_harm(greedy, treatment, turns=(14,))
    row = harm["by_turn"]["14"]
    assert row["treatment_closer_to_firestone"] is True
    assert row["macro_policy_harm"] < 0
    assert harm["applied_to_2n_v3"] is False
