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


def test_tech_card_is_promoted_when_the_matchup_wants_it():
    kb, scorer = cards.load_kb(), get_scorer()
    board = [{"name": "Roaring Recruiter", "card_id": "X1", "attack": 6, "health": 6},
             {"name": "Ancestral Automaton", "card_id": "X2", "attack": 7, "health": 7}]
    shop = [{"name": "Tunnel Blaster", "card_id": "BG_DAL_775", "attack": 3, "health": 7},
            {"name": "Roaring Recruiter", "card_id": "X1", "attack": 6, "health": 6}]
    shielded = [{"name": f"DS{i}", "attack": 3, "health": 2,
                 "tags": {"DIVINE_SHIELD": "1"}} for i in range(5)]
    snap = _snap(shop=shop, board=board, opponents_seen=[shielded])
    recs, _ = rank_actions(snap, kb=kb, scorer=scorer)
    tunnel = next(r for r in recs if "Tunnel" in r.action.describe())
    recruiter = next(r for r in recs if "Recruiter" in r.action.describe())
    # Against a wall of Divine Shields, the board-clear should now win.
    assert tunnel.placement < recruiter.placement
    assert "divine shield" in tunnel.reason.lower()


def test_naked_sell_is_not_a_top_recommendation():
    kb, scorer = cards.load_kb(), get_scorer()
    board = [{"name": f"M{i}", "card_id": f"c{i}", "attack": 5, "health": 5}
             for i in range(7)]
    snap = _snap(shop=[{"name": "weakling", "card_id": "w", "attack": 1, "health": 1}],
                 board=board, gold=1)
    recs, _ = rank_actions(snap, kb=kb, scorer=scorer)
    # You want a full board of 7; selling a minion for nothing shouldn't be #1.
    assert not recs[0].action.describe().startswith("Sell")


def test_tavern_spells_are_offered_as_buy_recommendations():
    kb, scorer = cards.load_kb(), get_scorer()
    snap = _snap(
        shop=[{"name": "Vanilla", "card_id": "v", "attack": 2, "health": 2}],
        board=[{"name": "A", "card_id": "a", "attack": 3, "health": 3}],
        tavern_tier=2, gold=5,
        shop_spells=[{"name": "Pointy Arrow", "card_id": "EBG_Spell_014", "cost": 1},
                     {"name": "Mystery Spell", "card_id": "EBG_Spell_999", "cost": 2}],
    )
    recs, _ = rank_actions(snap, kb=kb, scorer=scorer)
    spell_recs = [r for r in recs if r.action.kind == "buy_spell"]
    assert len(spell_recs) == 2
    # The spell line names the spell and its gold cost.
    assert any("Pointy Arrow" in r.action.describe() and "1g" in r.action.describe()
               for r in spell_recs)


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
