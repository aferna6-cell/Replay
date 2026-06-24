"""Firestone stats normalization tests (offline, on small recorded fixtures).

The fixtures mirror the real Firestone schema confirmed live on 2026-06-24.
"""

from hsbg_coach.firestone_stats import (
    normalize_heroes, normalize_comps, normalize_cards, normalize_trinkets,
    inject_core_cards, _playstyle, _archetype_tribe, _tier,
)

# Mini raw payloads in Firestone's real shape.
RAW_HEROES = {
    "dataPoints": 10000,
    "heroStats": [
        {
            "heroCardId": "TB_BaconShop_HERO_18", "dataPoints": 5000,
            "totalPicked": 4800, "averagePosition": 3.69,
            "combatWinrate": [{"turn": t, "winrate": w} for t, w in
                              [(2, 40), (3, 45), (4, 48), (5, 50),
                               (9, 58), (10, 60), (11, 61), (12, 62), (13, 63)]],
            "tribeStats": [
                {"tribe": 15, "dataPoints": 2000, "averagePosition": 3.4},  # Demon
                {"tribe": 24, "dataPoints": 2000, "averagePosition": 3.6},  # Dragon
                {"tribe": 20, "dataPoints": 800, "averagePosition": 3.9},   # Beast
            ],
        },
        {
            "heroCardId": "TB_BaconShop_HERO_TINY", "dataPoints": 100,  # below min
            "totalPicked": 90, "averagePosition": 4.9, "combatWinrate": [],
            "tribeStats": [],
        },
    ],
}

RAW_COMPS = {
    "compStats": [
        {"archetype": "murloc_scam", "dataPoints": 47000, "averagePlacement": 2.82},
        {"archetype": "mech_magnet", "dataPoints": 47243, "averagePlacement": 2.91},
        {"archetype": "tiny_sample", "dataPoints": 10, "averagePlacement": 1.0},  # filtered
    ],
}


def test_hero_normalization_maps_fields_and_tribes():
    names = {"TB_BaconShop_HERO_18": "Patches the Pirate"}
    out = normalize_heroes(RAW_HEROES, names)
    assert len(out) == 1                              # tiny hero filtered by min_data
    h = out[0]
    assert h["name"] == "Patches the Pirate"
    assert h["averagePosition"] == 3.69
    assert h["bestTribes"][:2] == ["Demon", "Dragon"]  # sorted by avg position
    assert h["playstyle"] == "economy"                 # late winrate >> early


def test_playstyle_derivation():
    late = [{"turn": t, "winrate": 60} for t in range(9, 14)] + \
           [{"turn": t, "winrate": 40} for t in range(2, 6)]
    early = [{"turn": t, "winrate": 60} for t in range(2, 6)] + \
            [{"turn": t, "winrate": 40} for t in range(9, 14)]
    assert _playstyle(late) == "economy"
    assert _playstyle(early) == "tempo"
    assert _playstyle([]) == "flexible"


def test_comp_normalization_and_tribe_inference():
    out = normalize_comps(RAW_COMPS)
    assert len(out) == 2                              # tiny_sample filtered
    names = {c["name"] for c in out}
    assert "Murloc Scam" in names and "Mech Magnet" in names
    murloc = next(c for c in out if c["name"] == "Murloc Scam")
    assert murloc["tribe"] == "Murloc"
    assert murloc["tier"] == "S"                      # 2.82 avg


def test_archetype_tribe_inference():
    assert _archetype_tribe("beast_token") == "Beast"
    assert _archetype_tribe("neutral_back_to_back") is None


def test_tier_buckets():
    assert _tier(3.5) == "S" and _tier(3.9) == "A" and _tier(4.2) == "B"
    assert _tier(4.5) == "C" and _tier(4.8) == "D"


RAW_CARDS = {
    "cardStats": [
        {"cardId": "BG_MURLOC_1", "totalPlayed": 50000,
         "averagePlacement": 3.2, "averagePlacementOther": 4.0},
        {"cardId": "BG_MECH_1", "totalPlayed": 40000,
         "averagePlacement": 3.5, "averagePlacementOther": 4.1},
        {"cardId": "BG_RARE", "totalPlayed": 100,            # filtered by min_play
         "averagePlacement": 1.0, "averagePlacementOther": 4.5},
    ],
}
CARD_META = {
    "BG_MURLOC_1": {"name": "Murloc Warleader", "tribes": ["Murloc"], "techLevel": 3},
    "BG_MECH_1": {"name": "Junkbot", "tribes": ["Mech"], "techLevel": 4},
}

RAW_TRINKETS = {
    "trinketStats": [
        {"trinketCardId": "BG_T1", "dataPoints": 5000, "pickRate": 0.3, "averagePlacement": 3.1},
        {"trinketCardId": "BG_T2", "dataPoints": 50, "pickRate": 0.9, "averagePlacement": 2.0},  # filtered
    ],
}


def test_card_normalization_computes_impact():
    out = normalize_cards(RAW_CARDS, CARD_META)
    assert len(out) == 2                                  # rare card filtered
    warleader = next(c for c in out if c["cardId"] == "BG_MURLOC_1")
    assert warleader["name"] == "Murloc Warleader"
    assert warleader["tribes"] == ["Murloc"]
    assert warleader["impact"] == 0.8                     # 4.0 - 3.2 (placed better with it)


def test_inject_core_cards_by_tribe():
    cards = normalize_cards(RAW_CARDS, CARD_META)
    comps = [{"name": "Murlocs", "tribe": "Murloc", "coreCards": []},
             {"name": "Neutral", "tribe": None, "coreCards": []}]
    inject_core_cards(comps, cards)
    assert comps[0]["coreCards"] == ["Murloc Warleader"]
    assert comps[1]["coreCards"] == []                    # no tribe -> unchanged


def test_trinket_normalization_and_filter():
    out = normalize_trinkets(RAW_TRINKETS, {"BG_T1": {"name": "Ironforge Anvil"}})
    assert len(out) == 1                                  # low-data trinket filtered
    assert out[0]["name"] == "Ironforge Anvil"
    assert out[0]["averagePosition"] == 3.1
    assert out[0]["tier"] == "S"
