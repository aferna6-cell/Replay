"""In-game choice detection tests — drive the parser with synthetic
DebugPrintEntityChoices log blocks (the format we calibrate against a real log)."""

from hsbg_coach.choices import ChoiceParser, classify, name_for, rank_offer
from hsbg_coach import cards


def _block(card_ids, choice_type="GENERAL"):
    pre = "D 09:25:43.1 GameState.DebugPrintEntityChoices() - "
    lines = [f"{pre}id=1 Player=Bob ChoiceType={choice_type} CountMin=1 CountMax=1",
             f"{pre}  Source=GAME_005"]
    for i, cid in enumerate(card_ids):
        lines.append(f"{pre}  Entities[{i}]=[entityName=Foo id={70+i} "
                     f"zone=SETASIDE zonePos=0 cardId={cid} player=1]")
    lines.append("D 09:25:43.2 GameState.DebugPrintPower() - TAG_CHANGE Entity=x tag=Y value=1")
    return lines


def _run(lines):
    p = ChoiceParser()
    offers = [o for o in (p.feed(ln) for ln in lines) if o is not None]
    return offers


# --- classification ---------------------------------------------------------
def test_classify_by_cardid():
    assert classify(["BG30_MagicItem_403"]) == "trinket"
    assert classify(["BG26_HERO_101"]) == "hero"
    assert classify(["BGS_039", "BG24_001"]) == "discover"


# --- parsing ----------------------------------------------------------------
def test_detects_discover_offer():
    ids = ["BGS_039", "BGS_041", "BGS_043"]
    offers = _run(_block(ids))
    assert len(offers) == 1
    assert offers[0].kind == "discover" and offers[0].card_ids == ids


def test_detects_trinket_offer():
    offers = _run(_block(["BG30_MagicItem_403", "BG30_MagicItem_401"]))
    assert offers[0].kind == "trinket"


def test_send_choices_clears_open_block():
    p = ChoiceParser()
    for ln in _block(["BGS_039", "BGS_041"])[:-1]:      # header + entities, no end
        p.feed(ln)
    # player picks before any terminator -> SendChoices clears, no offer leaks
    assert p.feed("D 09:25:44 GameState.SendChoices() - id=1 ChoiceType=GENERAL") is None


# --- names + ranking --------------------------------------------------------
def test_name_resolution_uses_committed_data():
    cid = next(iter(cards.load_kb()))                   # a real minion id
    assert name_for(cid) == cards.load_kb()[cid].name


def test_rank_offer_returns_best_first():
    # Use two real minion names so the discover ranker can score them.
    kb = cards.load_kb()
    names = [c.name for c in list(kb.values())[:3]]
    from hsbg_coach.choices import ChoiceOffer
    offer = ChoiceOffer("discover", ["x", "y", "z"], names)
    picks = rank_offer(offer, board=[], kb=kb)
    assert len(picks) == 3 and picks[0].name in names
