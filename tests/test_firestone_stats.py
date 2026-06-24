"""Firestone stats normalization tests (offline, on small recorded fixtures).

The fixtures mirror the real Firestone schema confirmed live on 2026-06-24.
"""

from hsbg_coach.firestone_stats import (
    normalize_heroes, normalize_comps, _playstyle, _archetype_tribe, _tier,
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
