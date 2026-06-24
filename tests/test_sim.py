"""Combat simulator tests with deterministic scenarios.

Where outcomes are forced by the rules we assert exact results; where attack
order is random we assert probability bounds with a fixed seed.
"""

from hsbg_coach.sim import Combatant, simulate, simulate_once
import random


def C(attack, health, **kw):
    return Combatant(attack=attack, health=health, **kw)


def test_board_vs_empty_always_wins():
    r = simulate([C(3, 3)], [], runs=200, seed=1)
    assert r.win_pct == 1.0
    assert r.avg_damage_dealt > 0


def test_empty_vs_empty_is_tie():
    r = simulate([], [], runs=50, seed=1)
    assert r.tie_pct == 1.0


def test_bigger_minion_beats_smaller():
    # 5/5 vs 1/1: the big minion survives the trade and wins every time.
    r = simulate([C(5, 5)], [C(1, 1)], runs=200, seed=2)
    assert r.win_pct == 1.0
    assert r.losses == 0


def test_mirror_match_is_balanced():
    board = [C(3, 3), C(2, 4)]
    r = simulate([m.copy() for m in board], [m.copy() for m in board],
                 runs=2000, seed=3)
    # Symmetric boards: wins and losses should be close; ties possible.
    assert abs(r.win_pct - r.loss_pct) < 0.12
    assert r.wins + r.ties + r.losses == r.runs


def test_divine_shield_absorbs_one_hit():
    # 2/1 with divine shield vs 2/1: shielded minion eats the first hit and the
    # unshielded one dies, so A wins.
    r = simulate([C(2, 1, divine_shield=True)], [C(2, 1)], runs=200, seed=4)
    assert r.win_pct == 1.0


def test_poisonous_trades_up():
    # 1/1 poisonous vs 10/10: poison kills the big minion; both die -> tie here
    # (poison minion takes 10 and dies too), so neither wins.
    r = simulate([C(1, 1, poisonous=True)], [C(10, 10)], runs=200, seed=5)
    assert r.tie_pct == 1.0


def test_taunt_is_targeted_first():
    rng = random.Random(0)
    # Attacker 2/2 vs [10/10 taunt, 1/1]. Taunt must be hit first; a single 2/2
    # cannot kill a 10/10, so B always survives -> B wins.
    result = simulate_once([C(2, 2)], [C(10, 10, taunt=True), C(1, 1)], rng)
    assert result < 0  # B wins


def test_reborn_minion_comes_back():
    # 1/1 reborn vs 1/1: the reborn minion dies, returns as 1/1, and finishes
    # the lone enemy -> A wins more often than a plain 1/1 mirror would.
    r = simulate([C(1, 1, reborn=True)], [C(1, 1)], runs=500, seed=6)
    assert r.win_pct > 0.6


def test_probabilities_sum_to_one():
    r = simulate([C(3, 2), C(2, 3)], [C(4, 4)], runs=1000, seed=7)
    assert abs(r.win_pct + r.tie_pct + r.loss_pct - 1.0) < 1e-9


def test_deterministic_given_seed():
    a, b = [C(3, 3), C(2, 2)], [C(2, 4)]
    r1 = simulate([m.copy() for m in a], [m.copy() for m in b], runs=300, seed=42)
    r2 = simulate([m.copy() for m in a], [m.copy() for m in b], runs=300, seed=42)
    assert (r1.wins, r1.ties, r1.losses) == (r2.wins, r2.ties, r2.losses)
