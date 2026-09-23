"""Adherence check: metrics, frame rules and a real-log replay."""
from pathlib import Path

import pytest

from hsbg_coach.adherence import (GameAudit, TurnAudit, _check_frame, audit_log,
                                  find_logs, render_text, summarize)

REAL_LOG = Path(__file__).resolve().parent.parent / "Power.log"
PLAN = "Undead - Attack Scaling"


def _game(placement, bought_on, bought_off, up=True, pieces=2):
    t = TurnAudit(turn=6, plan=PLAN, frames=4,
                  plan_cards_up=["x"] if up else [],
                  bought=bought_on + bought_off,
                  bought_on_plan=bought_on, bought_off_plan=bought_off)
    return GameAudit(log="l", game=1, placement=placement, finished=True,
                     final_plan=PLAN, final_plan_pieces=pieces, turns=[t])


def test_player_score_combines_hunt_discipline_and_final_comp():
    g = _game(2, ["a"], ["b"], pieces=4)
    assert g.hunt_rate == 1.0 and g.discipline == 0.5 and g.final_comp == 1.0
    assert g.player_score == pytest.approx(83.3, abs=0.1)
    g2 = _game(7, [], ["b", "c"], pieces=0)
    assert g2.hunt_rate == 0.0 and g2.discipline == 0.0 and g2.player_score == 0.0


def test_summary_splits_placement_by_adherence():
    games = [_game(1, ["a"], [], pieces=4), _game(2, ["a"], [], pieces=3),
             _game(7, [], ["b"], pieces=0), _game(8, [], ["b"], pieces=1)]
    s = summarize(games)
    assert s["split"]["followed"]["avg_placement"] == 1.5
    assert s["split"]["strayed"]["avg_placement"] == 7.5
    text = render_text(games, s)
    assert "Followed the plan" in text and "too few" in text


def test_frame_rules_flag_coach_violations():
    snap = {"playbook_plan": PLAN, "gold": 5, "board": [], "hand": [],
            "shop": [{"name": "Eternal Knight", "tribes": ["Undead"]},
                     {"name": "Tunnel Blaster", "tribes": []}]}
    from hsbg_coach.lobby_playbook import comp_by_name
    core = next(n for n in comp_by_name(PLAN).core)
    snap["shop"][0]["name"] = core
    v = _check_frame(snap, ["Roll the shop (finish 5.0) — x"], None)
    assert v and v[0].startswith("roll as NEXT")
    v = _check_frame(snap, ["Buy Ominous Seer (finish 8.0) — never buy"], None)
    assert any("never to buy" in x for x in v)
    assert _check_frame(snap, [f"Buy {core} (finish 3.0) — PLAN"], None) == []


def test_find_logs_searches_folders(tmp_path):
    (tmp_path / "Hearthstone_2026_09_01").mkdir()
    f = tmp_path / "Hearthstone_2026_09_01" / "Power.log"
    f.write_text("")
    assert find_logs([str(tmp_path)]) == [str(f.resolve())]


@pytest.mark.skipif(not REAL_LOG.is_file(), reason="real Power.log not present")
def test_real_log_replays_without_coach_violations():
    games = audit_log(str(REAL_LOG))
    assert len(games) == 2
    assert all(not g.finished and g.placement is None for g in games)   # log ends mid-game
    assert games[1].hero == "Marin the Manager"
    assert len(games[1].lobby) == 5
    turns = [t.turn for t in games[1].turns if t.turn]
    assert turns == sorted(turns)                        # no half-turn flip-flop
    assert sum(len(g.coach_violations) for g in games) == 0, \
        [g.coach_violations for g in games]
