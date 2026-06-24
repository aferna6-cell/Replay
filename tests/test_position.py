"""Positioning optimizer tests."""

from hsbg_coach.sim import Combatant as C
from hsbg_coach.position import (
    optimize_vs, optimize_vs_field, positioning_advice, MAX_EXACT,
)


def test_winning_board_keeps_current_among_best():
    my = [C(3, 3), C(2, 4)]
    ranked = optimize_vs(my, [], runs=50, top=999)
    assert all(r.win_pct == 1.0 for r in ranked)        # beats empty every way
    assert any(r.is_current for r in ranked)            # current order is present


def test_best_is_at_least_current():
    my = [C(2, 2), C(3, 1), C(1, 4)]
    ranked = optimize_vs(my, [C(2, 3), C(2, 2)], runs=120, seed=1, top=999)
    best = ranked[0]
    current = next(r for r in ranked if r.is_current)
    assert best.win_pct >= current.win_pct              # best can't be worse than current


def test_exact_permutations_for_small_board():
    my = [C(1, 1), C(2, 2), C(3, 3)]                    # n=3 -> 3! = 6 orderings
    ranked = optimize_vs(my, [C(2, 2)], runs=20, top=999)
    assert len(ranked) == 6


def test_sampling_for_large_board():
    my = [C(1, 1)] * 7                                  # n=7 > MAX_EXACT -> sampled
    assert 7 > MAX_EXACT
    ranked = optimize_vs(my, [C(1, 1)], runs=10, sample=12, top=999)
    assert len(ranked) <= 12
    assert any(r.is_current for r in ranked)            # current order always included


def test_field_averaging_aggregates_runs():
    my = [C(3, 3)]
    enemies = [[C(1, 1)], [C(2, 2)]]
    ranked = optimize_vs_field(my, enemies, runs=50, top=1)
    assert ranked[0].result.runs == 100                # 50 runs x 2 enemies


def test_advice_returns_string():
    advice = positioning_advice([C(2, 2), C(3, 3)], [[C(2, 2)]], runs=50)
    assert isinstance(advice, str) and advice


def test_advice_for_certain_win():
    advice = positioning_advice([C(5, 5)], [[]], runs=20)
    assert "best" in advice.lower() or "fine" in advice.lower()
