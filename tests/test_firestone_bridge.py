"""Firestone bridge tests — conversion shape + transparent fallback (headless).

These run without Node installed: they exercise the board->BgsBattleInfo
conversion and confirm the pure-Python fallback is used when the sidecar isn't
available.
"""

from hsbg_coach import firestone_bridge as fb
from hsbg_coach.sim import Combatant as C, SimResult


def test_conversion_shape_matches_firestone_contract():
    info = fb.to_bgs_battle_info(
        [C(3, 4, taunt=True, name="A")],
        [C(5, 5, divine_shield=True, name="B")],
        turn=7, my_tier=3, enemy_tier=4, my_hp=25, enemy_hp=30, runs=500,
    )
    assert set(info) == {"playerBoard", "opponentBoard", "options", "gameState"}
    assert info["gameState"]["currentTurn"] == 7
    assert info["options"]["numberOfSimulations"] == 500

    pe = info["playerBoard"]["board"][0]
    assert pe["attack"] == 3 and pe["health"] == 4 and pe["taunt"] is True
    assert "enchantments" in pe and "entityId" in pe
    oe = info["opponentBoard"]["board"][0]
    assert oe["divineShield"] is True
    assert info["playerBoard"]["player"]["tavernTier"] == 3
    assert info["opponentBoard"]["player"]["hpLeft"] == 30


def test_card_id_carried_into_entity():
    class MV:
        def __init__(self):
            self.name = "Harvest Golem"
            self.card_id = "BG_FP1_024"
            self.attack, self.health, self.tags = 2, 3, {}
    info = fb.to_bgs_battle_info([MV()], [])
    assert info["playerBoard"]["board"][0]["cardId"] == "BG_FP1_024"


def test_unique_entity_ids_per_side():
    info = fb.to_bgs_battle_info([C(1, 1), C(1, 1)], [C(1, 1)])
    ids = [e["entityId"] for e in info["playerBoard"]["board"]]
    assert len(set(ids)) == 2                       # no dup ids on a side
    # player and opponent id ranges don't collide
    opp_ids = [e["entityId"] for e in info["opponentBoard"]["board"]]
    assert set(ids).isdisjoint(opp_ids)


def test_fallback_used_when_node_absent(monkeypatch=None):
    # Force the fallback and confirm we still get a valid SimResult.
    r = fb.simulate([C(5, 5)], [C(1, 1)], runs=100, seed=1, force_fallback=True)
    assert isinstance(r, SimResult)
    assert r.win_pct == 1.0                         # matches pure-sim behavior


def test_backend_name_reports_fallback_without_node():
    # In this headless test env Node/node_modules aren't installed.
    if not fb.is_available():
        assert fb.backend_name() == "python-fallback"


def test_raw_result_maps_to_sim_result():
    raw = {"wonPercent": 60.0, "tiedPercent": 10.0, "lostPercent": 30.0,
           "averageDamageWon": 4.0, "averageDamageLost": 2.0}
    res = fb._to_sim_result(raw, runs=1000)
    assert res.wins == 600 and res.ties == 100 and res.losses == 300
    assert res.avg_damage_dealt == 4.0 and res.avg_damage_taken == 2.0
