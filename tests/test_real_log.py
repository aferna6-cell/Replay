"""Live-parsing regression tests built from REAL macOS Hearthstone log lines
(captured from a player's client). These pin the calibration so it doesn't break.
"""

from hsbg_coach.bg import BGTracker
from hsbg_coach.parser import parse_line
from hsbg_coach.choices import ChoiceParser

# Verbatim lines from a real client (Battlegrounds game start + hero mulligan).
REAL_LINES = """\
D 23:18:29.3 GameState.DebugPrintPower() - FULL_ENTITY - Creating ID=35 CardID=TB_BaconShop_HERO_PH
D 23:18:29.3 GameState.DebugPrintPower() - FULL_ENTITY - Creating ID=74 CardID=TB_BaconShopBob
D 23:18:29.3 PowerTaskList.DebugPrintPower() -     FULL_ENTITY - Updating [entityName=Bartender Bob id=74 zone=PLAY zonePos=0 cardId=TB_BaconShopBob player=14] CardID=TB_BaconShopBob
D 23:18:29.5 GameState.DebugPrintEntityChoices() - id=1 Player=QuirkyTurtle#1118798 TaskList=7 ChoiceType=MULLIGAN CountMin=1 CountMax=1
D 23:18:29.5 GameState.DebugPrintEntityChoices() -   Source=GameEntity
D 23:18:29.5 GameState.DebugPrintEntityChoices() -   Entities[0]=[entityName=A. F. Kay id=113 zone=HAND zonePos=1 cardId=TB_BaconShop_HERO_16 player=6]
D 23:18:29.5 GameState.DebugPrintEntityChoices() -   Entities[1]=[entityName=Murloc Holmes id=114 zone=HAND zonePos=2 cardId=BG23_HERO_303 player=6]
D 23:18:29.5 GameState.DebugPrintEntityChoices() -   Entities[2]=[entityName=Lich Baz'hial id=115 zone=HAND zonePos=3 cardId=TB_BaconShop_HERO_25 player=6]
D 23:18:29.5 GameState.DebugPrintEntityChoices() -   Entities[3]=[entityName=Ysera id=116 zone=HAND zonePos=4 cardId=TB_BaconShop_HERO_53 player=6]
D 23:18:30.0 GameState.DebugPrintPower() - TAG_CHANGE Entity=GameEntity tag=TURN value=1
""".splitlines()


def test_detects_battlegrounds_from_cardids():
    # No LoadingScreen scene line in this client — detection must come from the
    # TB_Bacon* entity cardIds.
    t = BGTracker()
    for ln in REAL_LINES:
        ev = parse_line(ln)
        if ev:
            t.feed(ev)
    assert t.in_bg is True


def test_hero_select_offer_uses_log_entity_names():
    cp = ChoiceParser()
    offer = None
    for ln in REAL_LINES:
        o = cp.feed(ln)
        if o:
            offer = o
    assert offer is not None and offer.kind == "hero"
    assert offer.names == ["A. F. Kay", "Murloc Holmes", "Lich Baz'hial", "Ysera"]
