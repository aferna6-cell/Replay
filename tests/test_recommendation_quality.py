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
    # Against a wall of Divine Shields, the board-clear should now win, and the
    # reason is grounded in the simulated combat-win swing.
    assert tunnel.placement < recruiter.placement
    assert "win" in tunnel.reason.lower() and "sim" in tunnel.reason.lower()


def test_sim_models_tunnel_blaster_aoe_deathrattle():
    from hsbg_coach.sim import simulate, Combatant

    def mk(n, a, h, tags=None):
        return {"name": n, "attack": a, "health": h, "tags": tags or {}}

    mine = [mk("A", 5, 5), mk("B", 5, 5), mk("C", 5, 5), mk("D", 5, 5)]
    shields = [mk(f"DS{i}", 2, 2, {"DIVINE_SHIELD": "1"}) for i in range(6)]

    def win(my):
        return simulate([Combatant.from_minion(x) for x in my],
                        [Combatant.from_minion(x) for x in shields], runs=400).win_pct

    # Without the AOE deathrattle the swarm of shields wins; with it, it's cleared.
    assert win(mine) < 0.1
    assert win(mine + [mk("Tunnel Blaster", 3, 7)]) > 0.8


def test_sim_reads_dict_minions():
    # Regression: from_minion must read dicts (live snapshots), not just objects —
    # a dict used to silently become a 0/0 with no keywords.
    from hsbg_coach.sim import Combatant
    c = Combatant.from_minion({"name": "Deadly Spore", "attack": 1, "health": 1})
    assert c.attack == 1 and c.poisonous is True


def test_tech_not_promoted_when_the_matchup_does_not_want_it():
    kb, scorer = cards.load_kb(), get_scorer()
    board = [{"name": f"M{i}", "card_id": f"c{i}", "attack": 5, "health": 5} for i in range(4)]
    shop = [{"name": "Tunnel Blaster", "card_id": "BG_DAL_775", "attack": 3, "health": 7}]
    giant = [{"name": "Giant", "attack": 12, "health": 12}]      # one tall threat
    snap = _snap(shop=shop, board=board, opponents_seen=[giant])
    recs, _ = rank_actions(snap, kb=kb, scorer=scorer)
    tunnel = next(r for r in recs if "Tunnel" in r.action.describe())
    # 3-to-all does nothing to a lone 12/12 — the read should call it situational.
    assert "situational" in tunnel.reason.lower() or "no combat" in tunnel.reason.lower()


def test_low_tier_minion_not_bought_late_game():
    # A tier-1 minion at tier 6 is filler however it's buffed — roll instead.
    kb, scorer = cards.load_kb(), get_scorer()
    board = [{"name": f"G{i}", "card_id": f"g{i}", "attack": 34, "health": 38}
             for i in range(6)]
    shop = [{"name": "Ominous Seer", "card_id": "BG31_330", "attack": 10, "health": 10}]
    snap = _snap(shop=shop, board=board, gold=7, tavern_tier=6)
    top = rank_actions(snap, kb=kb, scorer=scorer)[0][0]
    assert not top.action.describe().startswith("Buy Ominous Seer")


def test_targeted_spell_recommends_best_minion():
    from hsbg_coach.spell_target import best_buff_target, is_targeted
    from hsbg_coach.live import _hand_spell_lines
    board = [{"name": "Big Vanilla", "attack": 30, "health": 30},
             {"name": "Shielded", "attack": 5, "health": 5, "tags": {"DIVINE_SHIELD": "1"}}]
    # A +stats buff should go on the Divine-Shield minion, not the bigger vanilla.
    target, why = best_buff_target(board)
    assert target["name"] == "Shielded" and "divine shield" in why.lower()
    # Targeting classification: buffs target, coins/gold don't.
    assert is_targeted({"name": "Tavern Dish Banana"})
    assert not is_targeted({"name": "Tavern Coin", "coin": True})
    snap = {"hand_spells": [{"name": "Tavern Dish Banana"},
                            {"name": "Tavern Coin", "coin": True}], "board": board}
    lines = _hand_spell_lines(snap)
    assert any("Play Tavern Dish Banana on Shielded" in l for l in lines)
    assert any(l == "Play Tavern Coin" for l in lines)   # no bogus target


def test_full_board_hand_play_names_the_minion_to_sell():
    # 'Play X from hand' on a full board must name the weakest minion to sell,
    # not say a vague 'sell your weakest'.
    from hsbg_coach.live import _hand_play_lines
    board = [{"name": f"Big{i}", "card_id": f"b{i}", "attack": 10, "health": 10}
             for i in range(6)] + [{"name": "Runt", "card_id": "r", "attack": 1, "health": 1}]
    hand = [{"name": "Freebie", "card_id": "fb", "attack": 5, "health": 5,
             "tags": {"CARDTYPE": "MINION"}}]
    lines = _hand_play_lines({"board": board, "hand": hand}, cards.load_kb())
    assert any("sell Runt first" in l for l in lines)


def test_minion_added_to_hand_is_suggested_to_play():
    # A free minion in hand (e.g. one a combat effect generated) should surface as
    # a play, so the user doesn't leave it sitting there.
    from hsbg_coach.live import advice_lines
    kb, scorer = cards.load_kb(), get_scorer()
    hand = [{"name": "Freebie", "card_id": "fb", "attack": 3, "health": 4,
             "tags": {"CARDTYPE": "MINION"}}]
    snap = _snap(shop=[], board=[{"name": "A", "card_id": "a", "attack": 2, "health": 2}],
                 hand=hand, gold=4)
    lines = advice_lines(snap, kb, scorer)
    assert any("Play Freebie from hand" in l for l in lines)


def test_magnetic_mech_in_hand_suggests_a_fuse_target():
    # A Magnetic mech should be magnetized onto the best board mech (here the
    # Divine-Shield one), not played as a standalone body.
    from hsbg_coach.live import advice_lines
    from hsbg_coach.magnetize import best_magnetize_target
    kb, scorer = cards.load_kb(), get_scorer()
    board = [{"name": "Shielded", "card_id": "h", "attack": 6, "health": 6,
              "tags": {"CARDRACE": "MECH", "DIVINE_SHIELD": "1"}},
             {"name": "Plain", "card_id": "o", "attack": 5, "health": 5,
              "tags": {"CARDRACE": "MECH"}}]
    host, why = best_magnetize_target(board, kb)
    assert host["name"] == "Shielded" and "divine shield" in why.lower()
    hand = [{"name": "Magno", "card_id": "mg", "attack": 2, "health": 2,
             "tags": {"CARDTYPE": "MINION", "CARDRACE": "MECH", "MODULAR": "1"}}]
    snap = _snap(shop=[], board=board, hand=hand, gold=4)
    lines = advice_lines(snap, kb, scorer)
    assert any(l.startswith("Magnetize Magno onto Shielded") for l in lines)


def test_status_line_shows_a_sync_counter():
    # The panel must signal it re-read the tavern after a roll (not stuck): a
    # sync counter that ticks up on each ingested board/shop change.
    from hsbg_coach.overlay import format_next
    snap = {"turn": 7, "phase": "recruit", "tavern_tier": 4, "gold": 5,
            "hero_health": 30, "sync_seq": 3}
    out = format_next(snap, None, ["Buy X"])
    assert "synced" in out and "#3" in out


def test_late_game_does_not_manufacture_a_roll():
    # At tier 6 with a mediocre shop the coach must not synthetically boost "roll"
    # to the top — that's what locked the panel onto refresh late game.
    kb, scorer = cards.load_kb(), get_scorer()
    giants = [{"name": f"G{i}", "card_id": f"g{i}", "attack": 20, "health": 20}
              for i in range(5)]
    snap = _snap(shop=[{"name": "Tiny", "card_id": "t", "attack": 2, "health": 2}],
                 board=giants, gold=10, tavern_tier=6)
    recs, _ = rank_actions(snap, kb=kb, scorer=scorer)
    manufactured = "shop is only okay — roll for a stronger minion"
    assert all(r.reason != manufactured for r in recs)


def test_no_manufactured_roll_when_gold_is_tight():
    # Turn 8, tier 4, gold 4: not enough headroom to roll AND buy a result, so the
    # coach must not push 'roll for a stronger minion' (the recurring stuck-on-roll).
    kb, scorer = cards.load_kb(), get_scorer()
    board = [{"name": f"M{i}", "card_id": f"m{i}", "attack": 7, "health": 7}
             for i in range(6)]
    snap = _snap(shop=[{"name": "Okay", "card_id": "ok", "attack": 6, "health": 6}],
                 board=board, gold=4, tavern_tier=4, turn=8)
    recs, _ = rank_actions(snap, kb=kb, scorer=scorer)
    manufactured = "shop is only okay — roll for a stronger minion"
    assert all(r.reason != manufactured for r in recs)


def test_tier_two_on_turn_two_is_legal_and_recommended():
    # The classic aggressive line. On turn 2 the discounted tier-up cost is 4, so
    # with 4 gold it must be a LEGAL action AND surfaced as a strong move.
    from hsbg_coach.actions import legal_actions, LEVEL
    kb, scorer = cards.load_kb(), get_scorer()
    snap = _snap(shop=[{"name": "x", "card_id": "z", "attack": 2, "health": 2}],
                 board=[{"name": "A", "card_id": "a", "attack": 1, "health": 1}],
                 gold=4, tavern_tier=1, turn=2, level_cost=4, hero_health=30)
    levels = [a for a in legal_actions(snap) if a.kind == LEVEL]
    assert levels and levels[0].cost == 4          # affordable at 4 gold
    recs, _ = rank_actions(snap, kb=kb, scorer=scorer)
    lvl = next((r for r in recs if r.action.kind == LEVEL), None)
    assert lvl is not None and "aggressive leveling" in lvl.reason
    # The aggressive bonus should put it among the top moves, not buried.
    top3 = {r.action.kind for r in recs[:3]}
    assert LEVEL in top3


def test_discounted_level_cost_drops_one_per_turn_on_tier():
    from hsbg_coach.bg import BGTracker
    t = BGTracker(); t.local_player = 3; t.player_names = {3: "Me"}
    from hsbg_coach.state import Entity
    hero = Entity(id=9, card_id="H"); hero.tags = {"CARDTYPE": "HERO",
        "CONTROLLER": "3", "ZONE": "PLAY", "PLAYER_TECH_LEVEL": "1"}
    player = Entity(id=-1, name="Me"); player.tags = {"HERO_ENTITY": "9"}
    t.state.entities = {9: hero, -1: player}
    t._recruit_phases = 1                       # turn 1 on tier 1
    assert t._level_cost() == 5                 # base, no discount
    t._recruit_phases = 2                       # turn 2 still tier 1
    assert t._level_cost() == 4                 # discounted → enables 2-on-2


def test_all_offered_heroes_are_ranked_not_capped_to_two():
    # Heroes are ranked in full (the old 'first two are free' cap mis-picked a
    # padlocked hero). Paywalled-hero detection isn't possible from the log yet, so
    # the overlay shows the whole ranking + a caveat to skip locked ones.
    from hsbg_coach.choices import ChoiceOffer, rank_offer
    offer = ChoiceOffer("hero",
                        ["BG_A", "BG_B", "BG_C", "BG_D"],
                        ["Lord Jaraxxus", "Vol'jin", "Scabbs Cutterbutter", "Thorim"])
    picks = rank_offer(offer)
    assert len(picks) == 4          # all offered ranked, none silently dropped


def test_intrepid_botanist_recommends_attack_or_health_specifically():
    from hsbg_coach.choose_one import choose_one_advice
    botanist = {"name": "Intrepid Botanist", "card_id": "BG32_237"}
    # Healthy: take the +Attack half.
    adv = choose_one_advice(botanist, {"hero_health": 30})
    assert "+Attack" in adv and "Pristine Lilies" in adv
    # Low HP: survival half.
    adv_low = choose_one_advice(botanist, {"hero_health": 8})
    assert "+Health" in adv_low and "Giant Dewdrop" in adv_low
    # Unknown choose-one → no curated pick (caller uses a generic hint).
    assert choose_one_advice({"name": "Mystery", "card_id": "ZZ"}) is None


def test_shop_spells_only_count_the_real_tavern_row():
    # A 'buy spell' must be an actual tavern offering — in zone=PLAY under the same
    # shop controller as the shop minions. Pool spells (SETASIDE) and your own
    # spellcraft spell (controller=you) must NOT show as buyable (the recurring
    # 'wrong spell on the board' report).
    from hsbg_coach.bg import BGTracker, Phase
    from hsbg_coach.state import Entity
    t = BGTracker(); t.local_player = 12; t.phase = Phase.RECRUIT
    shop_min = Entity(id=1, card_id="BG_M")
    shop_min.tags = {"CARDTYPE": "MINION", "ZONE": "PLAY", "CONTROLLER": "5"}
    real_spell = Entity(id=2, card_id="BG_REAL_SPELL")
    real_spell.tags = {"CARDTYPE": "BATTLEGROUND_SPELL", "ZONE": "PLAY",
                       "CONTROLLER": "5", "COST": "1"}
    my_spellcraft = Entity(id=3, card_id="BG28_504")        # Recruit a Trainee (yours)
    my_spellcraft.tags = {"CARDTYPE": "BATTLEGROUND_SPELL", "ZONE": "PLAY",
                          "CONTROLLER": "12", "COST": "2"}
    pool_spell = Entity(id=4, card_id="BG28_897")           # set-aside, not in tavern
    pool_spell.tags = {"CARDTYPE": "BATTLEGROUND_SPELL", "ZONE": "SETASIDE",
                       "CONTROLLER": "5", "COST": "1"}
    t.state.entities = {1: shop_min, 2: real_spell, 3: my_spellcraft, 4: pool_spell}
    spells = t._shop_spells()
    ids = {s["card_id"] for s in spells}
    assert ids == {"BG_REAL_SPELL"}        # only the true tavern offering
    # With no shop minion to anchor the row, don't risk a phantom spell.
    t.state.entities = {3: my_spellcraft, 4: pool_spell}
    assert t._shop_spells() == []


def test_advice_key_changes_when_you_act():
    # The overlay recomputes advice only when the state key changes; if the key is
    # too coarse it stays stuck on an action already taken. Rolling (gold drops),
    # playing a hand card (hand shrinks), and a stat buff must each change the key.
    from hsbg_coach.live import _key
    base = {"board": [{"entity_id": 1, "name": "A", "attack": 3, "health": 3}],
            "shop": [{"entity_id": 9, "name": "S", "attack": 2, "health": 2}],
            "hand": [{"entity_id": 5, "name": "Freebie", "attack": 4, "health": 4}],
            "gold": 5, "tavern_tier": 3, "phase": "recruit", "hero_health": 30}
    k0 = _key(base)
    assert _key({**base, "gold": 4}) != k0                       # rolled
    assert _key({**base, "hand": []}) != k0                      # played the hand card
    buffed = {**base, "board": [{"entity_id": 1, "name": "A", "attack": 6, "health": 5}]}
    assert _key(buffed) != k0                                    # a stat buff landed


def test_played_spell_is_not_still_in_hand():
    # A spell you already played leaves a SETASIDE/pool copy; only a spell in
    # zone=HAND is castable. Accepting SETASIDE made the coach stay stuck telling
    # you to play a spell you'd just used.
    from hsbg_coach.bg import BGTracker
    from hsbg_coach.state import Entity
    t = BGTracker(); t.local_player = 7
    in_hand = Entity(id=1, card_id="BG28_897")
    in_hand.tags = {"CARDTYPE": "BATTLEGROUND_SPELL", "ZONE": "HAND",
                    "CONTROLLER": "7", "COST": "1"}
    spent = Entity(id=2, card_id="BG28_897")            # the played/pool copy
    spent.tags = {"CARDTYPE": "BATTLEGROUND_SPELL", "ZONE": "SETASIDE",
                  "CONTROLLER": "7", "COST": "1"}
    t.state.entities = {2: spent}
    assert t._hand_spells() == []                       # nothing castable in hand
    t.state.entities = {1: in_hand, 2: spent}
    assert len(t._hand_spells()) == 1                   # only the HAND copy counts


def test_passive_hero_power_is_not_offered():
    from hsbg_coach.bg import BGTracker
    from hsbg_coach.state import Entity
    t = BGTracker(); t.local_player = 3; t.player_names = {3: "Me"}
    hero = Entity(id=90, card_id="BG_HERO_X")
    hero.tags = {"CARDTYPE": "HERO", "CONTROLLER": "3", "ZONE": "PLAY",
                 "HAS_ACTIVATE_POWER": "0", "HEALTH": "30"}
    player = Entity(id=-1, name="Me"); player.tags = {"HERO_ENTITY": "90", "RESOURCES": "5"}
    hp = Entity(id=122, name="Wingmen")
    # Passive powers hide their cost and carry no COST tag (Illidan's Wingmen).
    hp.tags = {"CARDTYPE": "HERO_POWER", "CONTROLLER": "3", "HIDE_COST": "1"}
    t.state.entities = {90: hero, -1: player, 122: hp}
    assert t._hero_power() is None          # passive — never offered as "use"
    # An activatable power (real COST, no HIDE_COST) is offered.
    hp.tags = {"CARDTYPE": "HERO_POWER", "CONTROLLER": "3", "COST": "1"}
    assert t._hero_power() is not None


def test_hero_power_is_a_recommendable_action_when_usable():
    from hsbg_coach.actions import legal_actions, HERO_POWER
    snap = _snap(shop=[], board=[], gold=2,
                 hero_power={"name": "Trade Up", "card_id": "HP", "cost": 1, "usable": True})
    kinds = [a.kind for a in legal_actions(snap)]
    assert HERO_POWER in kinds
    # Not offered when exhausted/unusable.
    snap2 = dict(snap, hero_power={"name": "Trade Up", "cost": 1, "usable": False})
    assert HERO_POWER not in [a.kind for a in legal_actions(snap2)]


def test_recruit_always_has_a_next_move_even_with_no_shop():
    # After combat the shop may not be parsed yet — roll/tier/end are still legal,
    # so there must always be a move to show (fixes the post-combat blank).
    kb, scorer = cards.load_kb(), get_scorer()
    from hsbg_coach.live import advice_lines
    snap = _snap(shop=[], board=[{"name": "x", "card_id": "z", "attack": 3, "health": 3}],
                 gold=4, tavern_tier=2)
    lines = advice_lines(snap, kb, scorer)
    assert lines, "recruit with no shop should still recommend roll/tier/end"


def test_minimal_view_shows_one_move_and_status():
    from hsbg_coach.overlay import format_next
    snap = {"turn": 5, "phase": "recruit", "tavern_tier": 3, "gold": 7,
            "hero_health": 40, "anomaly": "Marin's Treasure Box",
            "hero_power": {"usable": True}}
    text = format_next(snap, None, ["Buy Titus Rivendare (finish 2.3) — core Mech"])
    assert text.startswith("→ Buy Titus Rivendare")
    assert "anomaly: Marin's Treasure Box" in text and "hero power ready" in text
    assert "Your board" not in text          # no board dump in the minimal view


def test_hero_fallback_covers_heroes_thin_at_top_mmr():
    # Ysera / Lich Baz'hial are popular overall but thin at top-10%; the all-MMR
    # fallback should give them a real placement instead of "no stats".
    from hsbg_coach.stats import StatsDB
    from hsbg_coach.draft import rank_heroes
    db = StatsDB.load()
    names = {h.name for h in db.heroes}
    if "Ysera" not in names:
        import pytest as _pt
        _pt.skip("hero stats not refreshed in this environment")
    ranked = rank_heroes(["Ysera", "Lich Baz'hial"], db)
    for c in ranked:
        assert "no stats" not in c.reason
        assert "avg" in c.reason


def test_effect_synergy_matches_producers_to_payoffs():
    from hsbg_coach.effect_synergy import card_profile, board_synergy
    from hsbg_coach import cards
    kb = cards.load_kb()
    idx = {c.name: c for c in kb.values()}
    payoff = idx.get("Scavenging Hyena")        # "whenever a friendly Beast dies…"
    if payoff is None:
        import pytest as _pt
        _pt.skip("expected card not in KB")
    assert "tribe:beast" in card_profile(payoff).wants
    # A Beast added next to the Hyena should register a generalized text combo.
    beast = next((c for c in kb.values()
                  if "beast" in [t.lower() for t in (c.tribes or [])] and c.name != payoff.name), None)
    assert beast is not None
    score, reasons = board_synergy(beast, [payoff])
    assert score > 0 and any("beast" in r.lower() for r in reasons)


def test_full_board_buy_names_the_minion_to_sell():
    kb, scorer = cards.load_kb(), get_scorer()
    board = ([{"name": "Weakling", "card_id": "w", "attack": 1, "health": 1}]
             + [{"name": f"M{i}", "card_id": f"c{i}", "attack": 6, "health": 6}
                for i in range(6)])                      # full board of 7
    shop = [{"name": "Titus Rivendare", "card_id": "t", "attack": 6, "health": 6}]
    snap = _snap(shop=shop, board=board, gold=6)
    recs, _ = rank_actions(snap, kb=kb, scorer=scorer)
    buy = next(r for r in recs if r.action.describe().startswith("Buy"))
    # The buy must tell you which minion to sell to make room (the weakest).
    assert "sell Weakling for room" in buy.reason


def test_freeze_not_recommended_for_a_single_okay_minion():
    # gold 0, strong board, one mediocre shop minion — freezing is a trap.
    kb, scorer = cards.load_kb(), get_scorer()
    board = [{"name": f"M{i}", "card_id": f"c{i}", "attack": 7, "health": 9}
             for i in range(7)]
    snap = _snap(shop=[{"name": "Scarlet Skull", "card_id": "BG26_178",
                        "attack": 3, "health": 4}],
                 board=board, gold=0, tavern_tier=3)
    top = rank_actions(snap, kb=kb, scorer=scorer)[0][0]
    assert not top.action.describe().startswith("Freeze")


def test_weak_filler_is_not_bought_over_rolling():
    # A minion far weaker than your board is slot-filler, not an upgrade — on a
    # board of giants the coach should roll for a real one, not buy the small minion.
    kb, scorer = cards.load_kb(), get_scorer()
    giants = [{"name": f"G{i}", "card_id": f"g{i}", "attack": 30, "health": 38}
              for i in range(6)]
    snap = _snap(shop=[{"name": "Tiny", "card_id": "t", "attack": 4, "health": 6}],
                 board=giants, gold=10, tavern_tier=6)
    top = rank_actions(snap, kb=kb, scorer=scorer)[0][0]
    assert not top.action.describe().startswith("Buy Tiny")


def test_discover_prefers_on_tribe_over_high_stat_off_tribe():
    from hsbg_coach.build_path import load_archetypes
    from hsbg_coach.draft import rank_discover
    if not load_archetypes():
        import pytest as _pt
        _pt.skip("archetype data not present")
    kb = cards.load_kb()
    idx = {c.name: c for c in kb.values()}
    murlocs = [c.name for c in kb.values()
               if "murloc" in [t.lower() for t in (c.tribes or [])]]
    nagas = [c.name for c in kb.values()
             if "naga" in [t.lower() for t in (c.tribes or [])]]
    if len(murlocs) < 3 or not nagas:
        import pytest as _pt
        _pt.skip("need murloc + naga cards")
    board = [{"name": murlocs[0]}, {"name": murlocs[1]}]   # committed Murloc board
    ranked = rank_discover([murlocs[2], nagas[0]], board, kb, tier=4)
    assert "murloc" in ranked[0].reason.lower() or ranked[0].name == murlocs[2]


def test_off_comp_minion_is_not_recommended_over_rolling():
    # A committed Murloc board should not buy an off-tribe Undead with no synergy
    # just because the eval net likes its stats — roll for a piece that fits.
    kb, scorer = cards.load_kb(), get_scorer()
    murlocs = [c for c in kb.values()
               if "murloc" in [t.lower() for t in (c.tribes or [])]]
    undead = [c for c in kb.values()
              if "undead" in [t.lower() for t in (c.tribes or [])]
              and "murloc" not in [t.lower() for t in (c.tribes or [])]]
    if len(murlocs) < 4 or not undead:
        import pytest as _pt
        _pt.skip("need murloc + undead cards")
    board = [{"name": murlocs[i].name, "card_id": f"m{i}", "attack": 4, "health": 4}
             for i in range(4)]
    off = undead[0]
    snap = _snap(shop=[{"name": off.name, "card_id": "u", "attack": 6, "health": 6}],
                 board=board, gold=5, tavern_tier=3)
    recs, _ = rank_actions(snap, kb=kb, scorer=scorer)
    top = recs[0]
    assert not top.action.describe().startswith(f"Buy {off.name}")


def test_panel_shows_alternative_moves():
    from hsbg_coach.overlay import format_next
    snap = {"turn": 5, "phase": "recruit", "tavern_tier": 3, "gold": 7,
            "hero_health": 46}
    out = format_next(snap, None, ["Buy A", "Level up", "Roll the shop"])
    assert "→ Buy A" in out and "or:" in out
    assert "Level up" in out and "Roll the shop" in out


def test_late_game_rewards_a_real_scaling_upgrade():
    # Late game (turn 11, tier 6): a shop minion well above your board average is a
    # scaling upgrade and should rank above rolling.
    kb, scorer = cards.load_kb(), get_scorer()
    board = [{"name": f"B{i}", "card_id": f"b{i}", "attack": 8, "health": 8}
             for i in range(6)]
    shop = [{"name": "Huge", "card_id": "h", "attack": 30, "health": 30}]
    snap = _snap(shop=shop, board=board, gold=6, tavern_tier=6, turn=11)
    recs, _ = rank_actions(snap, kb=kb, scorer=scorer)
    buy = next((r for r in recs if r.action.target == "Huge"), None)
    assert buy is not None and "scales" in buy.reason.lower()
    assert recs[0].action.target == "Huge"           # the scaling buy leads


def test_freeze_fires_when_out_of_gold_with_a_good_card():
    from hsbg_coach.advisor import advise_actions
    from hsbg_coach.actions import FREEZE
    kb, scorer = cards.load_kb(), get_scorer()
    murlocs = [c for c in kb.values()
               if "murloc" in [t.lower() for t in (c.tribes or [])]]
    if len(murlocs) < 4:
        import pytest as _pt
        _pt.skip("need murloc cards")
    from hsbg_coach.synergy import score_card, load_embeddings
    board_cks = murlocs[:3]
    board = [{"name": m.name, "card_id": m.card_id} for m in board_cks]
    # The strongest on-tribe card in the shop; we're out of gold → freeze to keep it.
    emb = load_embeddings()
    want = max(murlocs[3:], key=lambda c: score_card(c, board_cks,
                                                     target_tribe="murloc", embeddings=emb).score)
    snap = _snap(shop=[{"name": want.name, "card_id": want.card_id,
                        "attack": want.attack or 4, "health": want.health or 4}],
                 board=board, gold=0, tavern_tier=4)
    plan = advise_actions(snap, kb=kb, scorer=scorer)
    frz = next((s for s in plan.ranked if s.action.kind == FREEZE), None)
    assert frz is not None and frz.priority >= 0.4 and "out of gold" in frz.reason
    # With gold to act, freeze stays buried.
    plan2 = advise_actions(dict(snap, gold=5), kb=kb, scorer=scorer)
    frz2 = next((s for s in plan2.ranked if s.action.kind == FREEZE), None)
    assert frz2 is not None and frz2.priority < 0.2


def test_timewarped_anomaly_prioritizes_supercharged_minions():
    # Under Timewarped, a shop minion flagged HAS_TIMEWARPED_TAVERN_ALT_TEXT has
    # its effect supercharged — the coach must surface buying it, not ignore the
    # anomaly. Tag-grounded from the real Power.log.
    kb, scorer = cards.load_kb(), get_scorer()
    shop = [{"name": "Plain", "card_id": "p", "attack": 4, "health": 4},
            {"name": "Warped", "card_id": "w", "attack": 3, "health": 3,
             "tags": {"HAS_TIMEWARPED_TAVERN_ALT_TEXT": "1"}}]
    snap = _snap(shop=shop, board=[], gold=3, tavern_tier=2,
                 anomaly="Timewarped")
    recs, _ = rank_actions(snap, kb=kb, scorer=scorer)
    warped = next((r for r in recs if r.action.target == "Warped"), None)
    plain = next((r for r in recs if r.action.target == "Plain"), None)
    assert warped is not None and "Timewarped" in warped.reason
    assert warped.placement <= plain.placement       # supercharged buy ranks ahead


def test_anomaly_note_surfaces_for_known_anomaly():
    from hsbg_coach.anomaly import anomaly_note
    assert anomaly_note("Timewarped") and "Timewarped" in anomaly_note("Timewarped")
    assert anomaly_note("Timewarped Minions")        # loose match
    assert anomaly_note(None) is None
    assert anomaly_note("Some Unknown Anomaly") is None


def test_keep_value_protects_a_synergistic_comp_piece():
    # On a committed Murloc board, a small on-tribe Murloc is worth more to KEEP
    # than a fat off-tribe vanilla — so 'sell for room' targets the vanilla, not
    # the synergy piece.
    from hsbg_coach.game_value import _keep_value
    from hsbg_coach.board_value import _val
    kb = cards.load_kb()
    murlocs = [c.name for c in kb.values()
               if "murloc" in [t.lower() for t in (c.tribes or [])]]
    if len(murlocs) < 3:
        import pytest as _pt
        _pt.skip("need murloc cards")
    # Equal stats, so synergy is the tiebreaker: an on-tribe murloc vs an off-tribe
    # vanilla of the same size. The committed comp piece must be worth more to keep.
    board = [{"name": murlocs[0], "card_id": "m0", "attack": 3, "health": 3},
             {"name": murlocs[1], "card_id": "m1", "attack": 3, "health": 3},
             {"name": murlocs[2], "card_id": "m2", "attack": 3, "health": 3},
             {"name": "Vanilla", "card_id": "v", "attack": 3, "health": 3}]
    # Synergy raises the on-tribe murloc's keep-value above its bare stats…
    assert _keep_value(board[0], board, kb) > _val(board[0])
    # …and the off-tribe vanilla is the one to sell for room, not a murloc.
    sell = min(board, key=lambda m: _keep_value(m, board, kb))
    assert sell["name"] == "Vanilla"


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


def test_build_path_steers_toward_a_reachable_winning_comp():
    from hsbg_coach.build_path import infer_target, path_value, load_archetypes
    if not load_archetypes():
        import pytest as _pt
        _pt.skip("archetype data not present")
    board = [{"name": "Ingenious Inventor"}, {"name": "Deflect-o-Bot"}]
    fit = infer_target(board)
    assert fit is not None and fit.arch.tribe == "Mech"
    # A missing core Mech piece should be valued as advancing the build (negative
    # = better finish); an off-tribe minion should not.
    core_adj, core_reason = path_value(board, "Titus Rivendare", 4, candidate_tribe="Mech")
    off_adj, _ = path_value(board, "Murloc Tidehunter", 4, candidate_tribe="Murloc")
    assert core_adj < 0 and "core" in (core_reason or "").lower()
    assert off_adj >= core_adj          # scatter is never valued above a core piece


def test_build_path_changes_buy_ranking_toward_the_comp():
    from hsbg_coach.build_path import load_archetypes
    if not load_archetypes():
        import pytest as _pt
        _pt.skip("archetype data not present")
    kb, scorer = cards.load_kb(), get_scorer()
    board = [{"name": "Ingenious Inventor", "card_id": "i", "attack": 4, "health": 4},
             {"name": "Deflect-o-Bot", "card_id": "d", "attack": 3, "health": 3}]
    shop = [{"name": "Titus Rivendare", "card_id": "t", "attack": 5, "health": 5},
            {"name": "Murloc Tidehunter", "card_id": "m", "attack": 5, "health": 5}]
    snap = _snap(shop=shop, board=board, tavern_tier=4, gold=6)
    recs, _ = rank_actions(snap, kb=kb, scorer=scorer)
    titus = next(r for r in recs if "Titus" in r.action.describe())
    murloc = next(r for r in recs if "Tidehunter" in r.action.describe())
    # Same stats, but the on-comp Mech piece should finish ahead of the off-comp one.
    assert titus.placement < murloc.placement


def test_discover_is_board_aware_via_build_path():
    from hsbg_coach.build_path import load_archetypes
    from hsbg_coach.draft import rank_discover
    if not load_archetypes():
        import pytest as _pt
        _pt.skip("archetype data not present")
    kb = cards.load_kb()
    board = [{"name": "Ingenious Inventor"}, {"name": "Deflect-o-Bot"}]
    ranked = rank_discover(["Titus Rivendare", "Murloc Tidehunter"], board, kb, tier=4)
    # On a Mech board, the core Mech discover should be picked over the off-tribe one.
    assert ranked[0].name == "Titus Rivendare"
    assert "mech" in ranked[0].reason.lower()


def test_completing_a_triple_beats_rolling():
    # Two copies on board + a third in the shop = a triple: buying it golds the
    # minion and Discovers a higher-tier one. That's a premium tempo play and
    # should take the line over "Roll the shop" — even late with a stocked board.
    kb, scorer = cards.load_kb(), get_scorer()
    pair = [{"name": "Deflect-o-Bot", "card_id": "d", "attack": 9, "health": 9}
            for _ in range(2)]
    filler = [{"name": f"M{i}", "card_id": f"m{i}", "attack": 8, "health": 8}
              for i in range(4)]
    snap = _snap(shop=[{"name": "Deflect-o-Bot", "card_id": "d", "attack": 9, "health": 9},
                       {"name": "Vanilla", "card_id": "v", "attack": 3, "health": 3}],
                 board=pair + filler, gold=10, tavern_tier=6)
    recs, _ = rank_actions(snap, kb=kb, scorer=scorer)
    top = recs[0]
    assert top.action.describe().startswith("Buy Deflect-o-Bot")
    assert "triple" in top.reason.lower()


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
