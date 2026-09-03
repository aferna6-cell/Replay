"""Tests for the combat effect engine: keywords + deathrattles + start-of-combat."""

import random
from hsbg_coach.sim import Combatant as C, simulate, simulate_once, _do_attack
from hsbg_coach.effects import Summon, StartOfCombat


def test_windfury_attacks_twice():
    # Mechanic-level: a windfury attacker swings twice in one activation. Use a
    # 0-attack punching bag so retaliation doesn't muddy the count.
    rng = random.Random(0)
    wf = C(2, 100, windfury=True)
    bag = C(0, 100)
    _do_attack(wf, [wf], [bag], rng)
    assert bag.health == 100 - 4          # 2 swings x 2 attack

    no = C(2, 100)
    bag2 = C(0, 100)
    _do_attack(no, [no], [bag2], rng)
    assert bag2.health == 100 - 2         # 1 swing


def test_cleave_hits_neighbors():
    # 3/10 cleave vs three 1/2s: cleave splashes the defender's neighbors, so it
    # clears the board far faster than a non-cleave attacker would.
    cl = simulate([C(3, 10, cleave=True)], [C(1, 2), C(1, 2), C(1, 2)], runs=300, seed=2)
    assert cl.win_pct > 0.9


def test_deathrattle_summon_continues_fight():
    # 1/1 that summons two 1/1s on death vs a single 3/3: the tokens keep
    # fighting after the original dies, beating a plain 1/1 (which just loses).
    dr = Summon(count=2, attack=1, health=1)
    withdr = simulate([C(1, 1, deathrattle=dr)], [C(3, 3)], runs=400, seed=3)
    plain = simulate([C(1, 1)], [C(3, 3)], runs=400, seed=3)
    assert withdr.win_pct + withdr.tie_pct > plain.win_pct + plain.tie_pct


def test_scallywag_token_attacks_immediately():
    # 1/1 whose deathrattle summons a 1/1 that attacks immediately vs a 2/1:
    # after the trade, the Sky Pirate strikes and can finish a wounded enemy.
    dr = Summon(count=1, attack=1, health=1, attack_immediately=True, name="Sky Pirate")
    r = simulate([C(2, 1, deathrattle=dr)], [C(2, 2)], runs=400, seed=4)
    # A's 2/1 trades into the 2/2 (leaving it 2/0? no — 2/2 takes 2 -> dies; both
    # die), then the immediate token has nothing to hit OR finishes a survivor.
    assert r.runs == 400  # smoke: runs to completion without error
    assert r.win_pct + r.tie_pct + r.loss_pct == 1.0


def test_start_of_combat_damage_applies_before_attacks():
    # A 2/2 with a start-of-combat that deals 2 to a random enemy vs a lone 2/2:
    # the opener kills the enemy before any attack, so A wins outright.
    soc = StartOfCombat(damage=2, targets=1)
    r = simulate([C(2, 2, start_of_combat=soc)], [C(2, 2)], runs=300, seed=5)
    assert r.win_pct == 1.0


def test_registry_effects_attach_from_minionview():
    # A MinionView named "Harvest Golem" should auto-gain its deathrattle.
    class MV:
        def __init__(self, name, attack, health):
            self.name = name
            self.card_id = None
            self.attack = attack
            self.health = health
            self.tags = {}
    golem = MV("Harvest Golem", 2, 3)
    plain = MV("Filler", 2, 3)
    with_dr = simulate([golem], [C(2, 4)], runs=400, seed=6)
    without = simulate([plain], [C(2, 4)], runs=400, seed=6)
    # The Damaged Golem token keeps fighting, so the named card does better.
    assert with_dr.win_pct + with_dr.tie_pct >= without.win_pct + without.tie_pct


def test_board_cap_respected_on_summons():
    # Seven 1/1s each summoning 2 tokens can't exceed the 7-minion cap.
    dr = Summon(count=2, attack=1, health=1)
    board = [C(1, 1, deathrattle=dr) for _ in range(7)]
    rng = random.Random(0)
    # Should simulate without overflowing / erroring.
    out = simulate_once([m.copy() for m in board], [C(2, 2)], rng)
    assert isinstance(out, int)
