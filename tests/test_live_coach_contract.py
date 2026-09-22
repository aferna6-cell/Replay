"""Contract tests for the live in-game coach surface.

Locks the Snapshot → ranked #1 move (+ rationale) → overlay/terminal formatter
path that the player sees while in-game. Pure/headless — no Tk, no ML training.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hsbg_coach.bg import BGTracker
from hsbg_coach.live import advice_lines, _combat_odds_for, _key
from hsbg_coach.overlay import format_next, format_overlay_text
from hsbg_coach.parser import parse_line
from hsbg_coach.board_value import HeuristicScorer

FIXTURES = Path(__file__).parent / "fixtures"
RECRUIT_SNAP = FIXTURES / "live_recruit_snapshot.json"

EMB = {
    "b1": [1.0, 0.0], "b2": [1.0, 0.0], "b3": [1.0, 0.0],
    "good": [1.0, 0.0], "bad": [0.0, 1.0],
}


def _load_recruit_fixture() -> dict:
    return json.loads(RECRUIT_SNAP.read_text())


def test_fixture_top_recommendation_is_deterministic_and_explainable():
    """Given a fixed Snapshot, #1 move is stable and carries a short why."""
    snap = _load_recruit_fixture()
    assert snap["phase"] == "recruit"
    assert snap["board"] and snap["shop"]

    a = advice_lines(snap, kb=None, scorer=HeuristicScorer(EMB))
    b = advice_lines(snap, kb=None, scorer=HeuristicScorer(EMB))
    assert a and b
    assert a[0] == b[0], "top recommendation must be deterministic for a fixture"
    # Action + short rationale (finish / why) — not a bare verb.
    top = a[0]
    assert any(tok in top for tok in ("Buy", "Roll", "Tier", "Sell", "Play", "Freeze"))
    assert ("—" in top) or ("finish" in top) or ("(" in top)


def test_format_next_leads_with_action_and_short_alternates():
    snap = _load_recruit_fixture()
    recs = advice_lines(snap, kb=None, scorer=HeuristicScorer(EMB))
    odds = _combat_odds_for(snap, runs=40, seed=0)
    text = format_next(snap, odds, recs)
    assert text.startswith("→ ")
    # Default view: primary move only (no finish/rationale clutter).
    primary = recs[0].split(" (finish ")[0].split(" — ")[0]
    assert primary in text
    # No board/shop/odds/status spam on the minimal overlay.
    assert "Your board" not in text
    assert "Combat:" not in text
    assert "gold" not in text


def test_format_overlay_text_shows_next_then_board():
    snap = _load_recruit_fixture()
    recs = ["Buy Gunpowder Courier (finish 3.8) — pirate synergy",
            "Roll the shop"]
    odds = "win 55% / tie 10% / loss 35%"
    text = format_overlay_text(snap, odds, recs)
    assert "NEXT → Buy Gunpowder Courier" in text
    assert text.index("NEXT") < text.index("Your board")
    assert "Combat: win 55%" in text
    assert "then:" in text and "Roll the shop" in text


def test_combat_odds_from_fixture_opponents_seen():
    snap = _load_recruit_fixture()
    assert snap.get("opponents_seen"), "fixture must carry a last-fight board"
    odds = _combat_odds_for(snap, runs=40, seed=0)
    assert odds is not None
    assert "win" in odds and "loss" in odds


def test_combat_odds_none_without_enemy_or_board():
    assert _combat_odds_for({"board": [], "opponents_seen": [[{"attack": 1, "health": 1}]]}) is None
    assert _combat_odds_for({"board": [{"attack": 2, "health": 2}], "opponents_seen": []}) is None


def test_feed_remembers_opponent_board_without_snapshot_during_combat():
    """Regression: from-start replay used to miss opponents_seen because
    last_opponent_board was only updated inside snapshot()."""
    log = Path(__file__).resolve().parents[1] / "Power.log"
    if not log.is_file():
        pytest.skip("repo Power.log sample not present")
    t = BGTracker()
    with log.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            ev = parse_line(line)
            if ev:
                t.feed(ev)  # deliberately never call snapshot() mid-stream
    snap = t.snapshot().to_dict()
    assert getattr(t, "last_opponent_board", None), "combat feed must remember the enemy board"
    assert snap.get("opponents_seen"), "recruit snapshot must expose opponents_seen"


def test_advice_key_changes_when_shop_or_gold_changes():
    snap = _load_recruit_fixture()
    a = _key(snap)
    b = _key(dict(snap, gold=(snap.get("gold") or 0) + 1))
    shop = list(snap.get("shop") or [])
    if shop:
        c = _key(dict(snap, shop=shop[:1]))
        assert a != c
    assert a != b
