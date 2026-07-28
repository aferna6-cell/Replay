"""recommend() with a card knowledge base produces synergy-aware buy advice."""

from hsbg_coach.cards import CardKnowledge
from hsbg_coach.recommend import recommend
from hsbg_coach.economy import HeroContext


def _kb(cards):
    return {c.card_id: c for c in cards}


def K(name, tribes=None, keywords=None, text=""):
    return CardKnowledge(card_id=name, name=name, tier=1, attack=2, health=2,
                         tribes=tribes or [], keywords=keywords or [], text=text)


def test_recommend_adds_synergy_buy():
    kb = _kb([
        K("Murloc Warleader", tribes=["Murloc"], text="Give your other Murlocs +2/+1."),
        K("Annoy-o-Tron", tribes=["Mech"]),
        K("Murloc Scout", tribes=["Murloc"]),
    ])
    snap = {
        "turn": 6, "tavern_tier": 2, "gold": 6, "hero_health": 30,
        "board": [{"name": "Murloc Scout"}],
        "shop": [{"name": "Annoy-o-Tron"}, {"name": "Murloc Warleader"}],
    }
    recs = recommend(snap, hero_ctx=HeroContext(target_tribe="Murloc"), kb=kb)
    syn = [r for r in recs if r.source == "synergy"]
    assert syn, "expected a synergy recommendation"
    assert "Murloc Warleader" in syn[0].rationale       # the on-tribe buffer
    assert syn[0].detail["score"] > 0


def test_no_kb_means_no_synergy_source():
    snap = {"turn": 3, "tavern_tier": 1, "gold": 3, "hero_health": 30,
            "board": [], "shop": [{"name": "X"}]}
    recs = recommend(snap)                                # no kb
    assert all(r.source != "synergy" for r in recs)


def test_unknown_cards_skip_synergy_gracefully():
    kb = _kb([K("Known", tribes=["Beast"])])
    snap = {"turn": 3, "tavern_tier": 1, "gold": 3, "hero_health": 30,
            "board": [], "shop": [{"name": "Totally Unknown Card"}]}
    recs = recommend(snap, kb=kb)                         # shop card not in kb
    assert all(r.source != "synergy" for r in recs)      # no crash, no bogus rec
