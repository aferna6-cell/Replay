"""Live-parsing regression tests built from REAL macOS Hearthstone log lines
(captured from a player's client). These pin the calibration so it doesn't break.
"""

from pathlib import Path

import pytest

from hsbg_coach.bg import BGTracker
from hsbg_coach.parser import parse_line
from hsbg_coach.choices import ChoiceParser

# Full captured client log, committed in logs/ (the collected-log archive —
# see logs/README.md). The end-to-end test below replays it and pins the live
# snapshot fields (local player, board, shop, tier, gold, hp). The assertions
# are calibrated to THIS specific capture, so it's pinned by name rather than
# "newest log in the archive". Skipped gracefully if the fixture isn't present.
REAL_LOG = (Path(__file__).resolve().parent.parent
            / "logs" / "Power_20260625-232054_f9b95962.log")

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


def _replay(tracker: BGTracker, sample_every=0):
    """Replay the full real log, optionally collecting recruit snapshots."""
    samples = []
    with REAL_LOG.open(errors="ignore") as f:
        for i, line in enumerate(f):
            ev = parse_line(line)
            if ev:
                tracker.feed(ev)
            if sample_every and i % sample_every == 0:
                samples.append(tracker.snapshot())
    return samples


@pytest.mark.skipif(not REAL_LOG.exists(), reason="real Power.log fixture absent")
def test_full_log_identifies_local_player_and_names():
    t = BGTracker()
    _replay(t)
    # The human is the only seat with a real GameAccountId (hi != 0). In the
    # captured log that's QuirkyTurtle, PlayerID 3 in the final game.
    assert t.in_bg is True
    assert t.local_player == 3
    assert t.player_names.get(3) == "QuirkyTurtle#1118798"


@pytest.mark.skipif(not REAL_LOG.exists(), reason="real Power.log fixture absent")
def test_full_log_snapshot_reads_board_shop_tier_gold_hp():
    t = BGTracker()
    samples = _replay(t, sample_every=300)
    recruit = [s for s in samples if s.phase == "recruit" and s.shop]
    assert recruit, "expected at least one recruit snapshot with a shop"

    # Every shop entry is a real minion (no Bartender Bob / trinkets / spells) and
    # resolves to a real card id.
    for s in recruit:
        for m in s.shop:
            assert m.card_id and m.tags.get("CARDTYPE") == "MINION"

    # A mid-game recruit with a populated board: pins board filtering (real
    # minions only, no trinkets/placeholders) and the economy reads.
    with_board = [s for s in recruit if s.board]
    assert with_board, "expected a recruit snapshot with our own minions"
    s = with_board[-1]
    assert all(m.card_id and "Trinket" not in (m.card_id or "") for m in s.board)
    assert all("UNKNOWN ENTITY" not in (m.name or "") for m in s.board)
    assert s.tavern_tier and 1 <= s.tavern_tier <= 6
    assert s.gold is not None and 0 <= s.gold <= 12
    assert s.hero_health is not None and s.hero_health > 0


def test_phase_from_events_not_turn_parity():
    """Recruit/combat come from definitive events (attack = combat, the tavern's
    DragBuy = recruit), not TURN parity — which is offset in some games (anomalies)
    and wrongly showed 'combat' while the player was shopping."""
    from hsbg_coach.bg import BGTracker, Phase

    def feed(t, line):
        ev = parse_line(line)
        if ev:
            t.feed(ev)

    t = BGTracker()
    t.in_bg = True
    ts = "D 16:42:49.1129590 GameState.DebugPrintPower() - "
    # An attack means combat, even on an "odd" turn that parity would call recruit.
    feed(t, ts + "TAG_CHANGE Entity=GameEntity tag=TURN value=3")
    feed(t, ts + "BLOCK_START BlockType=ATTACK Entity=[id=5] Target=[id=9]")
    assert t.phase == Phase.COMBAT
    # The tavern's buy mechanic being dealt means recruit, even on an "even" turn.
    feed(t, ts + "TAG_CHANGE Entity=GameEntity tag=TURN value=4")
    feed(t, ts + "FULL_ENTITY - Creating ID=274 CardID=TB_BaconShop_DragBuy")
    assert t.phase == Phase.RECRUIT


@pytest.mark.skipif(not REAL_LOG.exists(), reason="real Power.log fixture absent")
def test_full_log_tavern_tier_never_regresses_within_a_game():
    """Tavern tier only goes up inside a game. Catches the bug where trinkets
    (which also carry PLAYER_TECH_LEVEL) leak in and make the tier jump around."""
    t = BGTracker()
    last_game = None
    peak = 0
    with REAL_LOG.open(errors="ignore") as f:
        for line in f:
            ev = parse_line(line)
            if not ev:
                continue
            t.feed(ev)
            if t.state.game_counter != last_game:
                last_game, peak = t.state.game_counter, 0
            tier = t._tavern_tier()
            if tier is not None:
                assert tier >= peak, f"tier regressed {peak}->{tier}"
                peak = tier
