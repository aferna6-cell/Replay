"""Trinkets follow their HSReplay guides the way heroes follow theirs."""
from hsbg_coach import lobby_playbook as lp
from hsbg_coach.draft import rank_trinkets
from hsbg_coach.stats import StatsDB, TrinketStats
from hsbg_coach.trinket_comps import fit_for_name, pick_adjust

LOBBY = ["Beast", "Demon", "Mech", "Murloc", "Undead"]


def _db(*names, avg=4.0):
    return StatsDB([], [], [TrinketStats(name=n, card_id="", average_position=avg)
                            for n in names])


def test_guide_parsing_reads_comps_cards_and_dead_tribes():
    # "Commit Attack Scaling Undead." — comp named in words.
    assert "Undead - Attack Scaling" in fit_for_name("Staff of the Scourge").comps
    # [[card]] marks map to comps and become trinket buys.
    band = fit_for_name("Beetle Band")
    assert "Beasts - Beetles" in band.comps and "Banana Slamma" in band.buys
    # "Commit to Goldrinn Beasts or Leviathan Beasts Comp."
    assert "Beasts - Leviathan" in fit_for_name("Slamma Sticker").comps
    # "Commit to some naga comp." — Nagas are not in this patch.
    assert fit_for_name("Rusty Trident").dead
    assert not fit_for_name("Worn Treasure Map").leans      # "Always decent."


def test_locked_plan_picks_the_trinket_its_guide_is_for():
    snap = {"playbook_plan": "Undead - Attack Scaling", "available_tribes": LOBBY,
            "board": []}
    out = rank_trinkets(["Tiger Carving", "Staff of the Scourge"],
                        _db("Tiger Carving", "Staff of the Scourge"), snapshot=snap)
    assert out[0].name == "Staff of the Scourge", [(c.name, c.reason) for c in out]
    assert "(your PLAN)" in out[0].reason
    assert "not your PLAN" in out[1].reason


def test_plan_guide_beats_a_trinket_that_wins_more_lobbies():
    snap = {"playbook_plan": "Undead - Attack Scaling", "available_tribes": LOBBY}
    db = StatsDB([], [], [TrinketStats(name="Tiger Carving", card_id="", average_position=3.6),
                          TrinketStats(name="Staff of the Scourge", card_id="",
                                       average_position=4.2)])
    out = rank_trinkets(["Tiger Carving", "Staff of the Scourge"], db, snapshot=snap)
    assert out[0].name == "Staff of the Scourge"


def test_dead_guide_is_demoted():
    d, bits = pick_adjust("Rusty Trident", {"available_tribes": LOBBY})
    assert d > 0 and "not in this patch" in bits[0]
    out = rank_trinkets(["Rusty Trident", "Worn Treasure Map"],
                        _db("Rusty Trident", "Worn Treasure Map"),
                        snapshot={"available_tribes": LOBBY})
    assert out[0].name == "Worn Treasure Map"


def test_before_lock_the_lobby_decides():
    no_undead = ["Beast", "Demon", "Mech", "Murloc", "Pirate"]
    d, bits = pick_adjust("Staff of the Scourge", {"available_tribes": no_undead})
    assert d > 0 and "not in this lobby" in bits[0]
    d2, bits2 = pick_adjust("Staff of the Scourge", {"available_tribes": LOBBY})
    assert d2 <= 0


def test_equipped_trinket_guide_breaks_same_tier_comp_ties():
    beasts = ["Beast"]
    slamma = [fit_for_name("Slamma Sticker")]        # Leviathan Beasts
    mama = [fit_for_name("Mama Bear Sticker")]       # Summons
    a = lp.resolve_enabler("Sewer Lord", beasts, [], trinkets=slamma)
    b = lp.resolve_enabler("Sewer Lord", beasts, [], trinkets=mama)
    assert a is not None and b is not None
    assert a.name == "Beasts - Leviathan" and b.name == "Beasts - Summons"


def test_equipped_trinket_guide_cards_are_plan_support():
    snap = {"playbook_plan": "Beasts - Summons", "available_tribes": LOBBY,
            "trinkets": [{"name": "Mama Bear Sticker"}],
            "shop": [{"name": "Forest Rover"}], "hand": []}
    ci = lp.comp_by_name("Beasts - Summons")
    assert "Forest Rover" not in ci.cards
    assert lp.plan_support(snap, ci).get("Forest Rover") == \
        "Mama Bear Sticker HSReplay guide buy"


def test_off_plan_trinket_guide_cards_are_not_support():
    snap = {"playbook_plan": "Undead - Attack Scaling", "available_tribes": LOBBY,
            "trinkets": [{"name": "Beetle Band"}],
            "shop": [{"name": "Turquoise Skitterer"}], "hand": []}
    ci = lp.comp_by_name("Undead - Attack Scaling")
    assert "Turquoise Skitterer" not in lp.plan_support(snap, ci)


def test_equipped_trinket_guide_adds_its_tribe_to_strong():
    ranked = lp.rank_lobby_tribes(["Beast", "Mech", "Murloc", "Pirate", "Quilboar"])
    assert lp.strong_tribes(ranked) == ["Beast", "Murloc"]
    fit = fit_for_name("Accord-o-Tron Portrait")          # Magnetic Mechs (A)
    assert lp.strong_tribes(ranked, trinkets=[fit]) == ["Beast", "Mech"]
