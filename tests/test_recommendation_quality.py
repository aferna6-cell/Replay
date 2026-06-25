"""Recommendation-quality regressions from real play feedback (2026-06-25):
tech cards were over-recommended, and positioning advice was always generic
because the live opponent board wasn't threaded into the recommender.
"""

from hsbg_coach import cards
from hsbg_coach.board_value import get_scorer
from hsbg_coach.card_roles import is_tech, tech_note
from hsbg_coach.game_value import rank_actions


def _snap(shop, board=None, **kw):
    base = {"phase": "recruit", "turn": 7, "tavern_tier": 4, "gold": 6,
            "hero_health": 30, "board": board or [], "shop": shop,
            "opponents_seen": []}
    base.update(kw)
    return base


def test_tech_cards_are_flagged():
    assert is_tech("BG_DAL_775", None)          # Tunnel Blaster
    assert is_tech(None, "Deadly Spore")
    assert tech_note("BG_DAL_775", None)
    assert not is_tech("BG26_135", "Southsea Busker")


def test_tech_card_does_not_outrank_a_comparable_on_tribe_minion():
    kb, scorer = cards.load_kb(), get_scorer()
    beater = {"name": "Roaring Recruiter", "card_id": "X1", "attack": 6, "health": 6}
    snap = _snap(
        shop=[
            {"name": "Tunnel Blaster", "card_id": "BG_DAL_775", "attack": 3, "health": 7},
            dict(beater),
        ],
        board=[dict(beater),
               {"name": "Ancestral Automaton", "card_id": "X2", "attack": 7, "health": 7}],
    )
    recs, _ = rank_actions(snap, kb=kb, scorer=scorer)
    buys = [r for r in recs if r.action.describe().startswith("Buy")]
    # The on-tribe beater should be ranked at least as well as the tech card.
    tunnel = next(r for r in buys if "Tunnel" in r.action.describe())
    recruiter = next(r for r in buys if "Recruiter" in r.action.describe())
    assert recruiter.placement <= tunnel.placement
    assert "tech" in tunnel.reason.lower()


def test_reposition_uses_the_opponent_board_when_present():
    kb, scorer = cards.load_kb(), get_scorer()
    my = [{"name": "A", "card_id": "a", "attack": 5, "health": 5},
          {"name": "B", "card_id": "b", "attack": 2, "health": 8},
          {"name": "C", "card_id": "c", "attack": 8, "health": 2}]
    enemy = [{"name": "E1", "attack": 4, "health": 4},
             {"name": "E2", "attack": 3, "health": 6}]
    snap = _snap(shop=[{"name": "x", "card_id": "z", "attack": 1, "health": 1}],
                 board=my, opponents_seen=[enemy])
    recs, _ = rank_actions(snap, kb=kb, scorer=scorer)
    repo = next(r for r in recs if r.action.describe().startswith("Reposition"))
    # With a real opponent the reposition reason is concrete (an order or a
    # "current order is ~best (NN% win)" verdict), never the generic no-opponent
    # fallback.
    assert "no opponent" not in repo.reason.lower()
    assert "win" in repo.reason.lower()
