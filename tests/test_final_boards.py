"""Winning-board extraction tests + the committed real snapshot."""

from hsbg_coach.final_boards import extract, core_cards_by_archetype
from hsbg_coach.stats import load_final_boards, example_boards


# Minimal raw comp payload in Firestone's real finalBoards shape.
def _board(*minions):
    return {"finalComp": {"board": list(minions)}, "mmr": 8000}


def _m(card_id, race=None, pos=1, atk=5, hp=5, **kw):
    tags = {"ZONE_POSITION": pos, "ATK": atk, "HEALTH": hp}
    if race is not None:
        tags["CARDRACE"] = race
    for k, v in kw.items():
        tags[k] = v
    return {"cardID": card_id, "tags": tags}


RAW = {
    "compStats": [{
        "archetype": "murloc_scam",
        "heroStats": [{
            "heroCardId": "H1",
            "finalBoards": [
                _board(_m("M1", race=14, pos=1), _m("M2", race=14, pos=2, REBORN=1)),
                _board(_m("M1", race=14, pos=1), _m("M3", race=14, pos=3)),
                _board(_m("M1", race=14, pos=1), _m("M2", race=14, pos=2)),
            ],
        }],
    }],
}
NAMES = {"M1": "Murloc One", "M2": "Murloc Two", "M3": "Murloc Three"}


def test_core_cards_ranked_by_frequency():
    out = extract(RAW, NAMES, min_boards=1)
    assert len(out) == 1
    core = out[0]["coreCards"]
    # M1 is in all 3 boards -> frequency 1.0 and ranked first.
    assert core[0]["name"] == "Murloc One"
    assert core[0]["frequency"] == 1.0
    assert core[0]["commonPosition"] == 1


def test_example_boards_carry_minion_detail():
    out = extract(RAW, NAMES, min_boards=1, examples=2)
    ex = out[0]["examples"]
    assert len(ex) == 2
    m = ex[0]["minions"][0]
    assert m["name"] == "Murloc One" and m["tribe"] == "Murloc"
    assert "keywords" in m


def test_keywords_extracted_from_board_tags():
    out = extract(RAW, NAMES, min_boards=1)
    # M2 has REBORN in one board.
    flat = [mi for e in out[0]["examples"] for mi in e["minions"]]
    assert any("REBORN" in mi["keywords"] for mi in flat if mi["name"] == "Murloc Two")


def test_core_cards_by_archetype_helper():
    mapping = core_cards_by_archetype(extract(RAW, NAMES, min_boards=1), top=2)
    assert mapping["murloc_scam"][0] == "Murloc One"


def test_committed_snapshot_has_real_boards():
    boards = load_final_boards()
    assert len(boards) > 10                           # real committed data present
    assert all("coreCards" in b for b in boards)
    # at least one comp has example boards with positioned minions
    assert any(b.get("examples") and b["examples"][0]["minions"] for b in boards)
