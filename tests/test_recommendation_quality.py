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
