"""Aidan's playbook as HARD NEXT behavior: LOBBY → ENABLERS → FILL → COMMIT.

Proves (all from committed HSReplay data under data/hsreplay_guides/):
  1. Tribe ranking uses ONLY the lobby's tribes (HSReplay comp tiers)
  2. Enablers preload from comps.json for the strong lobby tribes (live pool,
     no Naga / out-of-pool)
  3. Before commit: fill beats roll on a solid shop; garbage can't lead
  4. Enabler in shop ⇒ Buy it is NEXT and PLAN → {comp} locks
  5. After the lock, NEXT hunts ONLY the comp's HSReplay core/key cards:
     core buys lead, off-plan buys are never NEXT or the first alternate, and
     the live coach keeps the lock for the rest of the game
  6. The overlay stays quiet: no LOBBY / ENABLERS / FILL / PLAN / NEED lines
"""
from __future__ import annotations

from hsbg_coach.actions import BUY, ROLL, SELL
from hsbg_coach.board_value import HeuristicScorer
from hsbg_coach.game_value import rank_actions
from hsbg_coach.pace import load_pace
from hsbg_coach.lobby_playbook import (
    PHASE_COMMIT, PHASE_FILL, PHASE_LOBBY, discover_pick, evaluate,
    live_comp_infos, overlay_lines, preload_enablers, rank_lobby_tribes,
    strong_tribes,
)
from hsbg_coach.hsreplay_guides import locked_comp, load_comps
from hsbg_coach.live import build_note_for
from hsbg_coach.overlay import format_next
from hsbg_coach.playstyle_prior import live_pool_names

PACE = load_pace()
SC = HeuristicScorer({})

# Beast + Demon are HSReplay S-tier tribes this patch; Mech/Murloc are A-tier;
# Aberration has no live HSReplay comp. Undead/Elemental are NOT in this lobby.
LOBBY = ["Mech", "Aberration", "Murloc", "Demon", "Beast"]


def _rank(snap):
    recs, _ = rank_actions(snap, kb=None, scorer=SC, pace=PACE,
                           include_reposition=False)
    return recs


def _m(name, a, h, tribe, tier):
    return {"name": name, "attack": a, "health": h, "tribes": [tribe], "tier": tier}


# ---------------------------------------------------------------- 1. LOBBY

def test_tribe_ranking_uses_only_lobby_tribes():
    ranked = rank_lobby_tribes(LOBBY)
    tribes = [r.tribe for r in ranked]
    assert sorted(tribes) == sorted(LOBBY)
    # S-tier HSReplay tribes first, no-comp Aberration last.
    assert set(tribes[:2]) == {"Beast", "Demon"}, tribes
    assert tribes[-1] == "Aberration"
    assert ranked[0].label == "S" and ranked[-1].best_tier is None
    # Undead / Elemental are S-tier globally but absent from this lobby.
    assert "Undead" not in tribes and "Elemental" not in tribes
    assert strong_tribes(ranked) == ["Beast", "Demon"]


def test_ranking_never_includes_naga():
    ranked = rank_lobby_tribes(["Naga", "Beast", "Pirate", "Dragon", "Quilboar"])
    assert "Naga" not in [r.tribe for r in ranked]


def test_strong_tribes_topped_up_when_lobby_has_no_s_tier():
    ranked = rank_lobby_tribes(["Mech", "Murloc", "Pirate", "Quilboar", "Aberration"])
    strong = strong_tribes(ranked)
    assert len(strong) == 2
    assert "Aberration" not in strong


# ------------------------------------------------------------- 2. ENABLERS

def test_enablers_preload_from_hsreplay_comps():
    enablers = preload_enablers(["Beast", "Demon"])
    comps = {c["name"]: c for c in load_comps()["comps"]}
    beetles = comps["Beasts - Beetles"]
    # HSReplay "when to commit" for Beetles: Scorpid / Skitterer.
    assert "Ravaging Scorpid" in enablers["Beast"]
    assert "Turquoise Skitterer" in enablers["Beast"]
    assert "Ravaging Scorpid" in (beetles.get("when_to_commit") or "")
    assert "Wrath Weaver" in enablers["Demon"]
    pool = live_pool_names()
    for names in enablers.values():
        for n in names:
            assert n in pool, f"{n} is not in the live pool"


def test_enablers_exclude_naga_and_out_of_pool():
    pool = live_pool_names()
    for ci in live_comp_infos():
        assert "naga" not in ci.name.lower()
        for n in ci.triggers + ci.cards:
            assert n in pool, (ci.name, n)


# ------------------------------------------------------------------ 3. FILL

def test_fill_phase_roll_not_next_on_solid_shop():
    snap = {
        "turn": 4, "tavern_tier": 2, "gold": 6, "hero_health": 30,
        "phase": "recruit", "available_tribes": LOBBY,
        "board": [_m("mech_a", 2, 3, "Mech", 1), _m("mech_b", 3, 2, "Mech", 1)],
        "shop": [_m("solid", 3, 4, "Murloc", 2), _m("ok", 2, 3, "Mech", 2)],
    }
    st = evaluate(snap)
    assert st.phase == PHASE_FILL and not st.committed
    recs = _rank(snap)
    assert recs[0].action.kind == BUY, [r.line() for r in recs[:3]]
    assert recs[0].action.kind != ROLL


def test_fill_phase_garbage_cannot_lead_when_solid_is_up():
    """A far-below-tier body is not a fill — the solid buy leads instead."""
    snap = {
        "turn": 9, "tavern_tier": 5, "gold": 6, "hero_health": 25,
        "phase": "recruit", "available_tribes": LOBBY,
        "board": [_m("m1", 8, 8, "Mech", 4), _m("m2", 7, 9, "Mech", 4),
                  _m("m3", 9, 7, "Mech", 5)],
        "shop": [_m("chaff", 30, 30, "Pirate", 1), _m("solid", 8, 8, "Mech", 5)],
    }
    recs = _rank(snap)
    assert recs[0].action.kind == BUY
    assert recs[0].action.target == "solid", [r.line() for r in recs[:3]]


# ---------------------------------------------------------------- 4. COMMIT

def _enabler_snap():
    return {
        "turn": 5, "tavern_tier": 3, "gold": 7, "hero_health": 28,
        "phase": "recruit", "available_tribes": LOBBY,
        "board": [_m("mech_a", 4, 4, "Mech", 2), _m("murloc_a", 3, 5, "Murloc", 2)],
        "shop": [
            _m("Mech Stat Stick", 7, 7, "Mech", 3),
            {"name": "Ravaging Scorpid", "attack": 3, "health": 3,
             "tribes": ["Beast"], "tier": 2},
            _m("Murloc Body", 5, 5, "Murloc", 3),
        ],
    }


def test_enabler_in_shop_locks_plan_and_is_next():
    snap = _enabler_snap()
    st = evaluate(snap)
    assert st.phase == PHASE_COMMIT
    assert st.plan == "Beasts - Beetles"
    assert st.trigger == "Ravaging Scorpid" and st.trigger_zone == "shop"
    recs = _rank(snap)
    assert recs[0].action.kind == BUY
    assert recs[0].action.target == "Ravaging Scorpid", [r.line() for r in recs[:3]]
    assert "HSReplay" in recs[0].reason and "Beetles" in recs[0].reason
    assert (locked_comp(snap) or {}).get("name") == "Beasts - Beetles"


def test_enabler_gate_changes_next(monkeypatch):
    """Control: with the hard gates disabled the bigger body leads; with them,
    the enabler does — the playbook really changes NEXT."""
    import hsbg_coach.lobby_playbook as lp

    def snap():
        s = _enabler_snap()
        # Mech-leaning board: pre-playbook direction logic calls Scorpid scatter.
        s["board"] = [_m("mech_a", 4, 4, "Mech", 2), _m("mech_b", 4, 4, "Mech", 2),
                      _m("mech_c", 4, 4, "Mech", 3)]
        return s

    # Pre-playbook behavior (#95): no phase state, no gates.
    monkeypatch.setattr(lp, "ensure", lambda s, kb=None: dict(
        s, playbook={"phase": lp.PHASE_FILL, "plan": None}))
    monkeypatch.setattr(lp, "apply_next_gates", lambda recs, *a, **k: recs)
    ungated = _rank(snap())
    monkeypatch.undo()
    gated = _rank(snap())
    assert ungated[0].action.target != "Ravaging Scorpid", [r.line() for r in ungated[:3]]
    assert gated[0].action.target == "Ravaging Scorpid", [r.line() for r in gated[:3]]


def test_unaffordable_enabler_is_frozen_not_rolled():
    from hsbg_coach.actions import FREEZE
    snap = dict(_enabler_snap(), gold=1)
    recs = _rank(snap)
    assert recs[0].action.kind == FREEZE, [r.line() for r in recs[:3]]
    assert "Ravaging Scorpid" in recs[0].reason


def test_enablers_and_core_are_one_set():
    """Aidan: an enabler is a core card and a core card is an enabler."""
    for ci in live_comp_infos():
        assert set(ci.triggers) == set(ci.core), ci.name
        assert set(ci.keys) <= set(ci.core), ci.name


def test_core_card_in_shop_locks_like_an_enabler():
    snap = _enabler_snap()
    snap["shop"] = [_m("Mech Stat Stick", 7, 7, "Mech", 3),
                    {"name": "Banana Slamma", "attack": 3, "health": 4,
                     "tribes": ["Beast"], "tier": 4}]
    st = evaluate(snap)
    assert st.committed and st.plan == "Beasts - Beetles", st
    recs = _rank(snap)
    assert recs[0].action.target == "Banana Slamma", [r.line() for r in recs[:3]]


def test_s_tier_comp_preferred_over_a_tier():
    """Headhunter Gryphon is in Beetles (S) and Summons / Lobstah (A): even
    with an A-comp piece on board, the S comp is the lock."""
    snap = _enabler_snap()
    snap["board"] = [{"name": "Titus Rivendare", "attack": 1, "health": 7,
                      "tribes": [], "tier": 5}, _m("mech_a", 4, 4, "Mech", 2)]
    snap["shop"] = [{"name": "Headhunter Gryphon", "attack": 4, "health": 4,
                     "tribes": ["Beast"], "tier": 4}]
    st = evaluate(snap)
    assert st.plan == "Beasts - Beetles", st


_A_LOBBY = ["Mech", "Murloc", "Pirate", "Quilboar", "Aberration"]


def test_b_tier_comps_are_not_preloaded():
    strong = strong_tribes(rank_lobby_tribes(_A_LOBBY))
    assert "Mech" in strong
    pre = preload_enablers(strong)
    # Scrap Scraper / Spark Snapper only appear in B-tier Mechs - Magnetics.
    assert "Scrap Scraper" not in pre["Mech"] and "Spark Snapper" not in pre["Mech"]
    assert "Glambot" in pre["Mech"]           # A-tier Mechs - Magnetics/Spells


def test_b_tier_comp_locks_only_on_a_high_roll():
    scraper = {"name": "Scrap Scraper", "attack": 6, "health": 5,
               "tribes": ["Mech"], "tier": 4}
    base = {"turn": 7, "tavern_tier": 4, "gold": 7, "hero_health": 25,
            "phase": "recruit", "available_tribes": _A_LOBBY, "shop": [scraper]}
    thin = dict(base, board=[_m("mech_a", 4, 4, "Mech", 3)])
    assert not evaluate(thin).committed
    highroll = dict(base, board=[
        {"name": "Drone Duplicator", "attack": 4, "health": 4, "tribes": ["Mech"], "tier": 3},
        {"name": "Spark Snapper", "attack": 5, "health": 5, "tribes": ["Mech"], "tier": 4},
    ])
    st = evaluate(highroll)
    assert st.committed and st.plan == "Mechs - Magnetics", st


def test_cross_tribe_enabler_needs_board_support():
    """Neutral Brann alone must not lock Demons; with 2 Demons on board it can."""
    base = {
        "turn": 7, "tavern_tier": 5, "gold": 8, "hero_health": 25,
        "phase": "recruit", "available_tribes": LOBBY,
        "shop": [{"name": "Brann Bronzebeard", "attack": 2, "health": 4,
                  "tribes": [], "tier": 5}],
    }
    thin = dict(base, board=[_m("mech_a", 5, 5, "Mech", 3)])
    assert not evaluate(thin).committed
    demons = dict(base, board=[_m("imp_a", 5, 5, "Demon", 3),
                               _m("imp_b", 5, 5, "Demon", 3)])
    st = evaluate(demons)
    assert st.committed and st.plan_tribe == "Demon", st


def test_non_strong_tribe_enabler_does_not_lock():
    """Mech is A-tier here (Beast/Demon are strong): Glambot doesn't commit."""
    snap = {
        "turn": 6, "tavern_tier": 4, "gold": 6, "hero_health": 25,
        "phase": "recruit", "available_tribes": LOBBY,
        "board": [_m("mech_a", 4, 4, "Mech", 3)],
        "shop": [{"name": "Glambot", "attack": 3, "health": 4,
                  "tribes": ["Mech"], "tier": 4}],
    }
    assert not evaluate(snap).committed


def test_discover_enabler_is_the_pick():
    snap = dict(_enabler_snap(), shop=[])
    pick = discover_pick(["Mech Body", "Wrath Weaver", "Murloc Body"], snap)
    assert pick == ("Wrath Weaver", "Demons - Self Damage")


# ------------------------------------------------------- 5. AFTER THE LOCK

def _committed_snap():
    return {
        "turn": 8, "tavern_tier": 4, "gold": 6, "hero_health": 22,
        "phase": "recruit", "available_tribes": LOBBY,
        "playbook_plan": "Beasts - Beetles",
        "board": [
            {"name": "Ravaging Scorpid", "attack": 4, "health": 4,
             "tribes": ["Beast"], "tier": 2},
            _m("mech_a", 6, 6, "Mech", 3), _m("mech_b", 6, 6, "Mech", 3),
            _m("murloc_a", 5, 5, "Murloc", 3),
        ],
        "shop": [
            _m("Big Mech", 9, 9, "Mech", 4),
            {"name": "Turquoise Skitterer", "attack": 3, "health": 4,
             "tribes": ["Beast"], "tier": 3},
        ],
    }


def test_after_lock_next_builds_the_hsreplay_comp():
    snap = _committed_snap()
    st = evaluate(snap)
    assert st.committed and st.plan == "Beasts - Beetles"
    recs = _rank(snap)
    assert recs[0].action.kind == BUY
    assert recs[0].action.target == "Turquoise Skitterer", [r.line() for r in recs[:3]]
    assert "PLAN Beasts - Beetles" in recs[0].reason


def test_after_lock_on_plan_tribe_buy_beats_off_plan():
    snap = _committed_snap()
    snap["shop"] = [_m("Big Mech", 9, 9, "Mech", 4), _m("Beast Body", 5, 5, "Beast", 4)]
    recs = _rank(snap)
    assert recs[0].action.kind == BUY
    assert recs[0].action.target == "Beast Body", [r.line() for r in recs[:3]]


def test_promoted_full_board_buy_still_names_the_sell():
    snap = _enabler_snap()
    snap["board"] = [_m(f"mech_{i}", 6, 6, "Mech", 3) for i in range(6)] + [
        _m("Weakling", 1, 1, "Mech", 1)]
    recs = _rank(snap)
    assert recs[0].action.target == "Ravaging Scorpid"
    assert "for room" in recs[0].reason and "HSReplay" in recs[0].reason, recs[0].reason


def test_after_lock_plan_piece_is_never_the_sell():
    snap = _committed_snap()
    snap["board"] = snap["board"] + [_m("mech_c", 6, 6, "Mech", 3),
                                     _m("mech_d", 6, 6, "Mech", 3),
                                     _m("mech_e", 6, 6, "Mech", 3)]
    recs = _rank(snap)
    assert not (recs[0].action.kind == SELL
                and recs[0].action.target == "Ravaging Scorpid")
    for r in recs:
        if r.action.kind == BUY and "for room" in (r.reason or ""):
            assert "sell Ravaging Scorpid" not in r.reason, r.reason


def test_live_coach_keeps_plan_after_enabler_leaves_shop():
    from hsbg_coach.live import LiveCoach

    coach = LiveCoach.__new__(LiveCoach)
    coach.kb = None
    coach._plan_game = None
    coach._plan = None
    coach._plan_trigger = None

    class _State:
        game_counter = 1

    class _Tracker:
        state = _State()

    coach.tracker = _Tracker()
    first = coach._with_playbook(_enabler_snap())
    assert first["playbook"]["plan"] == "Beasts - Beetles"
    rolled = dict(_enabler_snap(), shop=[_m("Mech Stat Stick", 7, 7, "Mech", 3)])
    later = coach._with_playbook(rolled)
    assert later["playbook"]["plan"] == "Beasts - Beetles"
    assert later["playbook_plan"] == "Beasts - Beetles"
    # New game → lock resets.
    _State.game_counter = 2
    fresh = coach._with_playbook(rolled)
    assert fresh["playbook"]["plan"] is None


def test_after_lock_core_buy_beats_level_and_roll():
    """Core card up ⇒ it is NEXT even with gold to level (economy waits)."""
    snap = _committed_snap()
    snap.update(gold=10, turn=6, tavern_tier=3)
    snap["shop"] = [_m("Big Mech", 9, 9, "Mech", 3),
                    {"name": "Headhunter Gryphon", "attack": 4, "health": 4,
                     "tribes": ["Beast"], "tier": 4}]
    recs = _rank(snap)
    assert recs[0].action.kind == BUY
    assert recs[0].action.target == "Headhunter Gryphon", [r.line() for r in recs[:3]]
    assert "core/key" in recs[0].reason or "commit trigger" in recs[0].reason


def test_after_lock_off_plan_is_never_next_or_first_alt():
    """Off-tribe fills, neutral flex and other tribes' enablers don't lead once
    committed — roll/level/economy do when nothing on-plan is up."""
    snap = _committed_snap()
    snap["board"] = snap["board"][:2]            # sparse board: still no wander
    snap["shop"] = [
        _m("Big Mech", 12, 12, "Mech", 4),
        {"name": "Wrath Weaver", "attack": 1, "health": 4, "tribes": ["Demon"], "tier": 1},
        {"name": "Brann Bronzebeard", "attack": 2, "health": 4, "tribes": [], "tier": 5},
    ]
    recs = _rank(snap)
    off = {"Big Mech", "Wrath Weaver", "Brann Bronzebeard"}
    assert not (recs[0].action.kind == BUY and recs[0].action.target in off), \
        [r.line() for r in recs[:3]]
    first_buy = next(i for i, r in enumerate(recs) if r.action.kind == BUY)
    non_buy = [i for i, r in enumerate(recs) if r.action.kind != BUY]
    assert first_buy > max(non_buy), [r.line() for r in recs]
    assert "off-plan" in recs[first_buy].reason


def test_after_lock_advice_lines_lead_with_the_hunt():
    from hsbg_coach.live import advice_lines
    snap = _committed_snap()
    lines = advice_lines(snap, kb=None, scorer=SC)
    assert lines[0].startswith("Buy Turquoise Skitterer"), lines[:3]
    snap["shop"] = [_m("Big Mech", 12, 12, "Mech", 4)]
    lines = advice_lines(snap, kb=None, scorer=SC)
    assert not lines[0].startswith("Buy Big Mech"), lines[:3]
    assert not any(ln.startswith("Buy Big Mech") for ln in lines[:2]), lines[:3]


# --------------------------------------------------------------- 6. OVERLAY

_PHASE_TAGS = ("LOBBY →", "ENABLERS →", "FILL →", "COMMIT →", "PLAN →",
               "NEED →", "HERO →")


def test_overlay_has_no_phase_chrome():
    fill = {
        "turn": 3, "tavern_tier": 2, "gold": 5, "phase": "recruit",
        "available_tribes": LOBBY,
        "board": [_m("mech_a", 2, 3, "Mech", 1)],
        "shop": [_m("solid", 3, 4, "Murloc", 2)],
    }
    for snap in (fill, _enabler_snap(), _committed_snap()):
        # Even if a debug note leaks into the snapshot, the overlay drops it.
        note = build_note_for(snap)
        text = format_next(dict(snap, build_note=note), None,
                           ["Buy solid (finish 5.0) — x", "Roll", "Tier up"])
        for tag in _PHASE_TAGS:
            assert tag not in text, text
        assert text.startswith("→ Buy solid"), text


def test_live_coach_attaches_no_strategy_header_by_default():
    import inspect
    from hsbg_coach.live import LiveCoach
    sig = inspect.signature(LiveCoach.__init__)
    assert sig.parameters["strategy_header"].default is False


def test_playbook_summary_still_available_for_debugging():
    """The phase machine itself still runs (internal state), just not shown."""
    lines = overlay_lines(dict(_enabler_snap()))
    assert lines[0].startswith("PLAN → Beasts - Beetles"), lines
    fill = {"turn": 3, "tavern_tier": 2, "gold": 5, "phase": "recruit",
            "available_tribes": LOBBY, "board": [], "shop": []}
    lines = overlay_lines(fill)
    assert lines[0].startswith("LOBBY → Beast(S) > Demon(S)"), lines
    assert lines[1].startswith("ENABLERS → Beast: Ravaging Scorpid"), lines


def test_unknown_lobby_still_ranks_and_reports():
    snap = {"turn": 1, "tavern_tier": 1, "gold": 3, "phase": "recruit",
            "board": [], "shop": []}
    st = evaluate(snap)
    assert st.phase == PHASE_LOBBY and not st.lobby_known
    assert "Naga" not in st.lobby
    assert overlay_lines(snap)[0].startswith("LOBBY → tribes not detected")


# ------------------------------------------------ 7. HEROES + AIDAN'S NOTES

def test_hero_fit_reads_hsreplay_hero_guides():
    from hsbg_coach.hero_comps import fit_for_name
    gally = fit_for_name("Trade Prince Gallywix")      # "try not to go Beasts or Undead"
    assert {"Beast", "Undead"} <= set(gally.avoid)
    assert "River Skipper" in gally.buys
    mill = fit_for_name("Millhouse Manastorm")         # "…Elementals - NOT Undeads, Beast, or Mechs"
    assert {"Elemental", "Pirate"} <= set(mill.tribes)
    assert {"Beast", "Mech"} <= set(mill.avoid)
    jarax = fit_for_name("Lord Jaraxxus")
    assert "Demons - Shop Buff" in jarax.comps


def test_hero_steers_strong_tribes():
    lobby = ["Beast", "Demon", "Mech", "Murloc", "Pirate"]
    snap = {"hero_name": "Millhouse Manastorm", "available_tribes": lobby,
            "turn": 2, "tavern_tier": 1, "gold": 4, "phase": "recruit",
            "board": [], "shop": []}
    st = evaluate(snap)
    assert "Beast" not in st.strong, st.strong          # hero guide: not Beasts
    assert "Pirate" in st.strong, st.strong             # hero favors Pirates (A)
    assert "Demon" in st.strong                         # S tribe stays


def test_hero_breaks_ties_between_same_tier_comps():
    """Felboar is core in two A-tier Demon comps; Jaraxxus' guide names
    Demons - Shop Buff, so that is the lock."""
    base = {"turn": 6, "tavern_tier": 4, "gold": 7, "phase": "recruit",
            "available_tribes": LOBBY, "board": [_m("imp", 3, 3, "Demon", 2)],
            "shop": [{"name": "Felboar", "attack": 4, "health": 4,
                      "tribes": ["Demon"], "tier": 3}]}
    plain = evaluate(base)
    jarax = evaluate(dict(base, hero_name="Lord Jaraxxus"))
    assert plain.committed and jarax.committed
    assert jarax.plan == "Demons - Shop Buff", jarax.plan
    assert plain.plan != jarax.plan or plain.plan == "Demons - Shop Buff"


def test_aidan_note_support_buy_beats_roll_after_lock():
    """Shop Buff Demons: a spell / Blood Gem generator is on-plan (Aidan's
    note) and is bought over rolling when no core card is up."""
    from hsbg_coach.comp_notes import notes_for, support_cards
    assert notes_for("Demons - Shop Buff")
    assert "Crater Miner" in support_cards("Demons - Shop Buff")
    snap = {"turn": 6, "tavern_tier": 3, "gold": 6, "hero_health": 25,
            "phase": "recruit", "available_tribes": LOBBY,
            "playbook_plan": "Demons - Shop Buff",
            "board": [_m("imp_a", 4, 4, "Demon", 2), _m("imp_b", 4, 4, "Demon", 2),
                      _m("imp_c", 4, 4, "Demon", 3)],
            "shop": [_m("Big Mech", 9, 9, "Mech", 3),
                     {"name": "Crater Miner", "attack": 2, "health": 3,
                      "tribes": ["Quilboar"], "tier": 2}]}
    recs = _rank(snap)
    buys = [r for r in recs if r.action.kind == BUY]
    miner = next(r for r in buys if r.action.target == "Crater Miner")
    mech = next(r for r in buys if r.action.target == "Big Mech")
    assert miner.placement < mech.placement, [r.line() for r in recs[:4]]
    assert recs[0].action.kind != BUY or recs[0].action.target == "Crater Miner", \
        [r.line() for r in recs[:4]]
    assert "off-plan" in mech.reason
