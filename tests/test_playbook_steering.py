"""The lobby playbook is the one steering direction (no legacy / Naga pulls)."""
from hsbg_coach import lobby_playbook as lp
from hsbg_coach.build_path import load_archetypes, path_value
from hsbg_coach.tribe_policy import direction_buy_penalty, playbook_direction

LOBBY = ["Mech", "Murloc", "Elemental", "Dragon", "Pirate"]


def _m(name, tribe, tier=2):
    return {"name": name, "attack": 3, "health": 3, "tribes": [tribe], "tier": tier}


def _snap(board, lobby=LOBBY, **kw):
    s = {"phase": "recruit", "turn": 5, "tavern_tier": 3, "gold": 7,
         "available_tribes": lobby, "board": board, "shop": [], "hand": []}
    s.update(kw)
    return lp.ensure(s)


def test_board_built_a_tribe_joins_strong_and_steers():
    # Real-log case: three Pirates on board, Elemental is the lobby's S tribe.
    snap = _snap([_m("p1", "Pirate"), _m("p2", "Pirate"), _m("p3", "Pirate")])
    assert "Pirate" in snap["playbook"]["strong"]
    assert playbook_direction(snap) == "Pirate"


def test_no_direction_until_the_board_supports_a_strong_tribe():
    snap = _snap([_m("d1", "Dragon")])
    assert "Dragon" not in snap["playbook"]["strong"]
    assert playbook_direction(snap) is None
    # …and the legacy board inference (which would say Dragon) stays off.
    pen, why = direction_buy_penalty(_m("m1", "Murloc", 3), snap)
    assert not why or "Dragon" not in why


def test_locked_plan_is_the_direction():
    snap = _snap([_m("p1", "Pirate"), _m("p2", "Pirate")],
                 playbook_plan="Elementals - Unbound Tempest")
    assert playbook_direction(snap) == "Elemental"


def test_build_path_never_targets_naga_and_follows_playbook_tribes():
    assert all(a.tribe != "Naga" for a in load_archetypes())
    board = [_m("x", "Pirate")]
    for card in ("Tarecgosa", "Gunpowder Courier", "Surfing Sylvar"):
        _, why = path_value(board, card, 4, tribes=["Elemental", "Pirate"])
        assert not why or "Naga" not in why
        _, why = path_value(board, card, 4)
        assert not why or "Naga" not in why
