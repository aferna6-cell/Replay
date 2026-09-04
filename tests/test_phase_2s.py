"""Phase 2S board-level abstract pool — locks + conservation tests."""

from hsbg_coach.bg_env import (
    A_BUY0,
    A_PLAY0,
    A_SELL0,
    PHASE_2Q_RECRUIT_VALUE_STATS,
    PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING,
    BGEnv,
    EnvMinion,
    PlayerState,
    board_level_abstract_scaling_enabled,
    board_synthetic_total,
    combat_raw,
    reallocate_abstract_pool,
    recruit_raw,
    recruit_value_stats_enabled,
    valuation_raw,
)
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


def test_methodology_is_2s_v1_default_off():
    assert METHODOLOGY_VERSION == "2s_v1"
    assert FEATURE_TOGGLE == "PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING"
    assert FEATURE_TOGGLE_DEFAULT is False
    assert PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING is False
    import hsbg_coach.bg_env as env
    assert getattr(env, FEATURE_TOGGLE) is False


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
    assert HOLD_PRS == (29, 33, 34, 35)
    assert FROZEN_ALPHA == 0.5
    d = diagnose_phase_2s()
    assert d["primary_finding"] == "implemented_not_evaluated"
    assert d["no_merge"] is True
    assert d["keep_hold_prs"] == [29, 33, 34, 35]
    assert d["2q_remains_treatment_selector"] is True


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
    assert recover["keep_pr_35_hold"] is True
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

    smoke = diagnose_phase_2s(collapse, non_evaluative=True)
    assert smoke["primary_finding"] == "implementation_smoke_non_evaluative"
    assert smoke["evaluative"] is False


def test_sell_loses_base_golden_but_not_synthetic_pool():
    """(1) Sell drops printed/golden recruit; synthetic pool is conserved."""
    env = BGEnv(seed=0)
    env.reset(seed=0)
    p = env.players[0]
    inflated = EnvMinion("a", "Inflated", 3, 4, 3, [], [])
    inflated.attack, inflated.health = 40, 30          # synthetic 63
    golden = EnvMinion(
        "b", "GoldenBody", 2, 4, 4, [], [], True, 8, 8,
    )
    # golden recruit 8/8; extra synthetic 10
    golden.attack, golden.health = 13, 13
    p.board = [inflated, golden]
    p.gold = 5
    with board_level_abstract_scaling_enabled(True):
        env._apply(0, A_SELL0)   # sell inflated (recruit 7)
        assert len(p.board) == 1
        assert p.board[0].name == "GoldenBody"
        assert p.board[0].recruit_attack == 8
        assert p.board[0].recruit_health == 8
        assert p.abstract_pool == 63 + 10
        assert p.strength() == 16 + 73
        assert p.recruit_strength() == 16
        env._apply(0, A_SELL0)   # sell golden (recruit 16)
        assert p.board == []
        assert p.abstract_pool == 73
        assert p.strength() == 0
        assert p.gold == 7


def test_sell_buy_play_conserves_abstract_pool():
    """(2) sell → buy → play keeps the synthetic pool; only recruit changes."""
    env = BGEnv(seed=1)
    env.reset(seed=1)
    p = env.players[0]
    keep = EnvMinion("k", "KeepMe", 3, 5, 5, [], [])
    keep.attack, keep.health = 25, 25                   # synth 40
    sold = EnvMinion("s", "Sold", 2, 3, 3, [], [])
    sold.attack, sold.health = 13, 13                   # synth 20
    newbie = EnvMinion("n", "New", 3, 6, 6, [], [])
    p.board = [keep, sold]
    p.shop = [newbie]
    p.gold = 10
    with board_level_abstract_scaling_enabled(True):
        env._apply(0, A_SELL0 + 1)
        assert p.abstract_pool == 60
        env._apply(0, A_BUY0)
        assert p.abstract_pool == 60
        assert p.recruit_strength() == 10
        env._apply(0, A_PLAY0)
        assert p.abstract_pool == 60
        assert {m.name for m in p.board} == {"KeepMe", "New"}
        assert p.recruit_strength() == 10 + 12
        assert p.strength() == 22 + 60
        assert board_synthetic_total(p.board) == 60


def test_no_replacement_path_identical_off_equivalent_on_before_scale():
    """(3) No sells: 2S OFF is today's path; 2S ON matches before/after scale."""
    def _prep(seed: int) -> BGEnv:
        env = BGEnv(seed=seed, scaling_mode="residual")
        env.reset(seed=seed)
        env.turn = 8
        p = env.players[0]
        p.tier = 4
        p.turns_since_level = 2
        p.board = [
            EnvMinion("a", "A", 3, 10, 10, [], []),
            EnvMinion("b", "B", 3, 8, 6, [], []),
        ]
        return env

    off = _prep(5)
    on = _prep(5)
    assert [(m.attack, m.health, m.recruit_attack, m.recruit_health)
            for m in off.players[0].board] == [
        (m.attack, m.health, m.recruit_attack, m.recruit_health)
        for m in on.players[0].board
    ]
    # 2S OFF residual apply (today's path).
    off._end_of_turn_scaling_residual(off.players[0])
    # 2S ON uses the same residual apply, then syncs the pool (no re-paint).
    with board_level_abstract_scaling_enabled(True):
        on._end_of_turn_scaling(on.players[0])
    off_board = [(m.attack, m.health, m.recruit_attack, m.recruit_health)
                 for m in off.players[0].board]
    on_board = [(m.attack, m.health, m.recruit_attack, m.recruit_health)
                for m in on.players[0].board]
    assert off_board == on_board
    assert on.players[0].abstract_pool == float(board_synthetic_total(on.players[0].board))
    # Before the next scale, no-replacement combat is unchanged (no realloc).
    assert on.players[0].strength() == off.players[0].strength()

    # 2S OFF sell still destroys on-body synthetic (default behavior).
    destroy = _prep(6)
    a = destroy.players[0].board[0]
    a.attack, a.health = 40, 30
    other = destroy.players[0].board[1]
    destroy._apply(0, A_SELL0)
    assert destroy.players[0].abstract_pool == 0
    assert destroy.players[0].board[0].attack == other.attack
    assert destroy.players[0].board[0].health == other.health


def test_combat_uses_pool_recruit_valuation_excludes_pool():
    """(4) Combat = recruit + pool; 2Q valuation never sees the pool."""
    m = EnvMinion("a", "A", 3, 4, 3, [], [])
    m.attack, m.health = 40, 30
    p = PlayerState(0)
    p.board = [m]
    p.abstract_pool = 63.0
    assert p.strength() == 70
    assert recruit_raw(m.view()) == 7
    assert combat_raw(m.view()) == 70
    with recruit_value_stats_enabled(True):
        assert valuation_raw(m.view()) == 7.0
    assert PHASE_2Q_RECRUIT_VALUE_STATS is False
    assert valuation_raw(m.view()) == 70.0

    reallocate_abstract_pool(p)
    assert p.strength() == 7 + 63
    assert p.recruit_strength() == 7
    assert recruit_raw(p.board[0].view()) == 7
    assert combat_raw(p.board[0].view()) == 70
    with recruit_value_stats_enabled(True):
        assert valuation_raw(p.board[0].view()) == 7.0


def test_residual_budget_math_unchanged_with_2s_on():
    """α / residual / ratio constants and budget intermediates stay identical."""
    def _budget(enabled: bool):
        env = BGEnv(seed=7, scaling_mode="residual")
        env.reset(seed=7)
        env.turn = 10
        p = env.players[0]
        p.tier = 5
        p.turns_since_level = 2
        p.board = [EnvMinion("a", "A", 4, 20, 20, [], [])]
        with board_level_abstract_scaling_enabled(enabled):
            return env._residual_scaling_budget(p)

    assert _budget(False) == _budget(True)
    assert FROZEN_ALPHA == 0.5
