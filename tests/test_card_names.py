"""Every Battlegrounds card resolves to a name (and tribes) — no raw ids."""
import re
from pathlib import Path

import pytest

from hsbg_coach.bg import BGTracker, _card_name
from hsbg_coach.lobby_playbook import card_tribes
from hsbg_coach.parser import parse_line
from hsbg_coach.tribe_policy import _is_quarantined_minion

REAL_LOG = Path(__file__).resolve().parent.parent / "Power.log"


def test_cards_outside_the_live_pool_kb_resolve():
    # Seen as raw ids in a real shop before the HearthstoneJSON fallback.
    assert _card_name("BG21_018") == "Defiant Shipwright"
    assert _card_name("BG33_894") == "Coldlight Diver"
    assert _card_name("BG28_521") == "Planar Telescope"
    assert _card_name("BG_EX1_014t") == "Bananas"
    assert card_tribes("Defiant Shipwright") == ("Pirate",)


def test_dual_tribe_naga_card_plays_as_its_live_tribe():
    seer = {"name": "Ominous Seer", "card_id": "BG31_330"}
    assert card_tribes(seer) == ("Demon",)
    assert not _is_quarantined_minion(seer)
    assert _is_quarantined_minion({"name": "x", "tribes": ["Naga"]})


@pytest.mark.skipif(not REAL_LOG.is_file(), reason="real Power.log not present")
def test_real_log_shows_no_raw_card_ids():
    t, raw = BGTracker(), set()
    for i, line in enumerate(REAL_LOG.open(encoding="utf-8", errors="replace")):
        ev = parse_line(line)
        if ev:
            t.feed(ev)
        if i % 3000 == 0:
            snap = t.snapshot().to_dict()
            for zone in ("shop", "board", "hand"):
                for m in snap.get(zone) or []:
                    name = m.get("name") if isinstance(m, dict) else None
                    if name and re.fullmatch(r"(BG|TB_)\w+", name):
                        raw.add(name)
    assert not raw, raw
