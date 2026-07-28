"""Recommendation facade tests — economy + positioning merge and combat odds."""

from hsbg_coach.recommend import recommend, combat_odds, overlay_payload
from hsbg_coach.bg import ActionType
from hsbg_coach.sim import Combatant as C


def snap(**over):
    base = {
        "turn": 2, "tavern_tier": 1, "gold": 4, "hero_health": 30,
        "board": [{"name": "Tabbycat", "attack": 1, "health": 1}],
        "shop": [{"name": "Alleycat", "attack": 1, "health": 1}],
        "notes": [],
    }
    base.update(over)
    return base


def test_economy_only_when_no_enemy_known():
    recs = recommend(snap())
    assert recs
    assert all(r.source == "economy" for r in recs)       # no positioning w/o enemy


def test_positioning_added_when_enemy_known():
    recs = recommend(snap(board=[{"attack": 3, "health": 3}, {"attack": 2, "health": 4}]),
                     enemy_boards=[[C(2, 2)]], runs=60)
    assert any(r.source == "positioning" for r in recs)


def test_recs_sorted_by_priority():
    recs = recommend(snap(turn=1, gold=3))
    prios = [r.priority for r in recs]
    assert prios == sorted(prios, reverse=True)


def test_combat_odds_string():
    odds = combat_odds(snap(board=[{"attack": 5, "health": 5}]),
                       [[C(1, 1)]], runs=50)
    assert "win" in odds and "loss" in odds


def test_combat_odds_none_without_enemy():
    assert combat_odds(snap(), None) is None


def test_overlay_payload_shape():
    payload = overlay_payload(snap(board=[{"attack": 3, "health": 3}]),
                              enemy_boards=[[C(1, 1)]], runs=50)
    assert "snapshot" in payload and "odds" in payload
    assert "recommendations" in payload
    # Top recs are surfaced as overlay notes.
    assert any(":" in n for n in payload["snapshot"]["notes"])


def test_works_with_snapshot_object():
    from hsbg_coach.bg import Snapshot, MinionView
    s = Snapshot(game_counter=1, turn=2, phase="recruit", tavern_tier=1,
                 gold=4, hero_health=30,
                 board=[MinionView(1, "X", "Tabbycat", 1, 1, 1)],
                 shop=[MinionView(2, "Y", "Alleycat", 1, 1, None)])
    recs = recommend(s)
    assert any(r.action == ActionType.BUY.value for r in recs)
