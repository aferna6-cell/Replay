"""Phase 2S preregistration locks — no simulator behavior."""

from hsbg_coach.bg_env import PHASE_2Q_RECRUIT_VALUE_STATS
from ml.phase_2s_prereg import (
    FEATURE_TOGGLE,
    FEATURE_TOGGLE_DEFAULT,
    FORBIDDEN_RANGES,
    FROZEN_ALPHA,
    GATE_GAME_LENGTH_DELTA_FLOOR,
    GATE_MEAN_COMBAT_LOSS_MAX,
    GATE_REPLACE_RATE_MIN,
    GATE_T10_POST_SCALE_DELTA_FLOOR,
    GATE_T10_POST_SCALE_MIN,
    HOLD_PRS,
    METHODOLOGY_VERSION,
    PHASE_2S_LOBBIES,
    PHASE_2S_SEED,
    assert_seed_range_allowed,
    diagnose_phase_2s,
    evaluate_phase_2s_gates,
)


def test_methodology_is_prereg_only():
    assert METHODOLOGY_VERSION == "2s_v0_prereg"
    assert FEATURE_TOGGLE == "PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING"
    assert FEATURE_TOGGLE_DEFAULT is False
    # Toggle must not be wired into bg_env yet.
    import hsbg_coach.bg_env as env
    assert not hasattr(env, FEATURE_TOGGLE)


def test_2q_toggle_remains_default_off():
    assert PHASE_2Q_RECRUIT_VALUE_STATS is False


def test_fresh_dev_seeds_above_14199():
    assert PHASE_2S_SEED == 14200
    assert PHASE_2S_LOBBIES == 500
    assert PHASE_2S_SEED > 14199
    assert (13700, 14199) in FORBIDDEN_RANGES
    assert (11500, 11699) in FORBIDDEN_RANGES
    assert_seed_range_allowed(PHASE_2S_SEED, PHASE_2S_LOBBIES)
    try:
        assert_seed_range_allowed(14199, 1)
        raise AssertionError("expected seed ≤14199 to fail")
    except ValueError:
        pass
    try:
        assert_seed_range_allowed(11500, 10)
        raise AssertionError("expected confirm overlap to fail")
    except ValueError:
        pass
    try:
        assert_seed_range_allowed(13700, 10)
        raise AssertionError("expected 2R overlap to fail")
    except ValueError:
        pass


def test_hold_stack_and_frozen_alpha():
    assert HOLD_PRS == (29, 33, 34)
    assert FROZEN_ALPHA == 0.5
    d = diagnose_phase_2s()
    assert d["primary_finding"] == "preregistered_not_run"
    assert d["no_merge"] is True
    assert d["keep_hold_prs"] == [29, 33, 34]


def test_predeclared_gates_and_routing():
    assert GATE_REPLACE_RATE_MIN == 0.10
    assert GATE_T10_POST_SCALE_MIN == 0.85
    assert GATE_T10_POST_SCALE_DELTA_FLOOR == -0.10
    assert GATE_GAME_LENGTH_DELTA_FLOOR == -0.50
    assert GATE_MEAN_COMBAT_LOSS_MAX == 20.0

    recover = evaluate_phase_2s_gates({
        "deltas": {
            "post_scale_over_firestone_t10": -0.02,
            "mean_game_length": -0.1,
        },
        "treatment": {
            "full_board_replace_rate": 0.25,
            "post_scale_over_firestone_t10": 0.94,
            "mean_combat_loss_per_replacement": 4.0,
        },
    })
    assert recover["route"] == "board_level_scaling_recovers_macro"
    assert recover["gates_passed"] == 5
    assert recover["keep_pr_34_hold"] is True
    assert recover["no_alpha_retune"] is True
    assert recover["no_scaling_retune"] is True

    collapse = evaluate_phase_2s_gates({
        "deltas": {
            "post_scale_over_firestone_t10": -0.48,
            "mean_game_length": -2.6,
        },
        "treatment": {
            "full_board_replace_rate": 0.30,
            "post_scale_over_firestone_t10": 0.47,
            "mean_combat_loss_per_replacement": 96.0,
        },
    })
    assert collapse["route"] == "representation_insufficient"
    assert collapse["gates"]["replacement_rate_held"] is True
    assert collapse["gates"]["post_scale_t10_near_firestone"] is False

    regress = evaluate_phase_2s_gates({
        "deltas": {
            "post_scale_over_firestone_t10": 0.0,
            "mean_game_length": 0.0,
        },
        "treatment": {
            "full_board_replace_rate": 0.02,
            "post_scale_over_firestone_t10": 0.95,
            "mean_combat_loss_per_replacement": 0.0,
        },
    })
    assert regress["route"] == "selection_regressed"
