"""End-to-end Battlegrounds state regression using a privacy-safe synthetic log."""

from pathlib import Path

from hsbg_coach.bg import BGTracker
from hsbg_coach.parser import parse_line


SYNTHETIC_LOG = Path(__file__).resolve().parent / "fixtures" / "synthetic_bg_power.log"


def _events():
    with SYNTHETIC_LOG.open() as f:
        for line in f:
            event = parse_line(line)
            if event is not None:
                yield event


def test_synthetic_full_log_preserves_state_regression_contract():
    """Exercise the semantic coverage formerly supplied only by captured Power.log."""
    tracker = BGTracker()
    for event in _events():
        tracker.feed(event)

    snapshot = tracker.snapshot()
    assert tracker.in_bg is True
    assert tracker.local_player == 3
    assert tracker.player_names.get(3) == "ReplaySynthetic#0001"
    assert snapshot.phase == "recruit"

    # Economy + survivability are resolved through the synthetic player -> hero link.
    assert snapshot.tavern_tier == 4
    assert snapshot.gold == 6
    assert snapshot.hero_health == 31

    # Board filtering keeps only our real in-play minion.
    assert [m.card_id for m in snapshot.board] == ["BGS_039"]
    assert all(m.tags.get("CARDTYPE") == "MINION" for m in snapshot.board)

    # Shop filtering keeps revealed foreign minions and excludes the trinket.
    assert [m.card_id for m in snapshot.shop] == ["BGS_010"]
    assert all(m.tags.get("CARDTYPE") == "MINION" for m in snapshot.shop)


def test_synthetic_full_log_tavern_tier_never_regresses():
    """Pin the monotonic tier invariant without relying on a captured account log."""
    tracker = BGTracker()
    peak = 0
    observed = []
    for event in _events():
        tracker.feed(event)
        tier = tracker._tavern_tier()
        if tier is not None:
            assert tier >= peak, f"tier regressed {peak}->{tier}"
            peak = tier
            observed.append(tier)

    assert 3 in observed
    assert observed[-1] == 4
