"""Card knowledge + synergy tests."""

from hsbg_coach.cards import CardKnowledge, load_kb, BG_CARDS
from hsbg_coach.synergy import derive_tags, score_card, rank_shop


def K(name, tier=1, attack=1, health=1, tribes=None, keywords=None, text=""):
    return CardKnowledge(card_id=name, name=name, tier=tier, attack=attack,
                         health=health, tribes=tribes or [], keywords=keywords or [],
                         text=text)


# --- tag derivation ----------------------------------------------------------
def test_keyword_tags():
    tags = derive_tags(K("X", keywords=["BATTLECRY", "DIVINE_SHIELD"]))
    assert "is:battlecry" in tags and "is:divine shield" in tags


def test_buffs_tribe_tag():
    c = K("Warleader", tribes=["Murloc"], text="Give your other Murlocs +2/+1.")
    tags = derive_tags(c)
    assert "buffs:Murloc" in tags


def test_cares_about_tag():
    c = K("Pack Leader", tribes=["Beast"], text="After you play a Beast, give it +2/+1.")
    tags = derive_tags(c)
    assert "cares:Beast" in tags


def test_doubler_tag():
    c = K("Brann", text="Your Battlecries trigger twice.")
    assert "doubles:battlecry" in derive_tags(c)


# --- scoring -----------------------------------------------------------------
def test_on_tribe_scores_higher():
    cand = K("Murloc A", tribes=["Murloc"])
    board = [K("Murloc B", tribes=["Murloc"]), K("Murloc C", tribes=["Murloc"])]
    on = score_card(cand, board)
    off = score_card(K("Mech", tribes=["Mech"]), board)
    assert on.score > off.score
    assert any("on-tribe" in r for r in on.reasons)


def test_buffer_candidate_rewarded():
    cand = K("Warleader", tribes=["Murloc"], text="Give your other Murlocs +2/+1.")
    board = [K("Murloc B", tribes=["Murloc"])]
    v = score_card(cand, board)
    assert any("buffs your Murloc" in r for r in v.reasons)


def test_battlecry_doubler_synergy():
    board = [K("Brann", text="Your Battlecries trigger twice.")]
    cand = K("BC Minion", keywords=["BATTLECRY"], text="Battlecry: do a thing.")
    v = score_card(cand, board)
    assert any("battlecry doubled" in r.lower() for r in v.reasons)


def test_target_comp_bonus():
    cand = K("Dragon X", tribes=["Dragon"])
    v = score_card(cand, [], target_tribe="Dragon")
    assert any("target comp" in r for r in v.reasons)


def test_hero_power_tribe_care():
    cand = K("Murloc A", tribes=["Murloc"])
    v = score_card(cand, [], hero_power_text="Give a Murloc +1/+1.")
    assert any("hero power" in r.lower() for r in v.reasons)


def test_rank_shop_orders_by_synergy():
    board = [K("Murloc B", tribes=["Murloc"]), K("Murloc C", tribes=["Murloc"])]
    shop = [K("Mech", tribes=["Mech"]),
            K("Warleader", tribes=["Murloc"], text="Give your other Murlocs +2/+1.")]
    ranked = rank_shop(shop, board)
    assert ranked[0][0].name == "Warleader"           # on-tribe buffer wins


# --- real committed knowledge base ------------------------------------------
def test_real_kb_loads_and_has_tier_tribe_keywords():
    kb = load_kb(BG_CARDS)
    assert len(kb) > 200                              # committed snapshot present
    # every card has a tier; at least some carry tribes and keywords
    assert all(c.tier is not None for c in kb.values())
    assert any(c.tribes for c in kb.values())
    assert any(c.keywords for c in kb.values())
