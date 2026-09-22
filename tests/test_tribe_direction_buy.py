"""Lobby-tribe soft lean, buy demotion, Activate harden, Dark Gift > body."""

from __future__ import annotations

from hsbg_coach.activate_cards import is_activate_minion, ACTIVATE_CARD_IDS
from hsbg_coach.actions import legal_actions, ACTIVATE, BUY
from hsbg_coach.tribe_policy import (
    soft_lean_tribe, direction_buy_penalty, filter_lobby_tribes,
    tribe_win_priors, canonicalize,
)
from hsbg_coach.dark_gift import GiftedOption, rank_dark_gift_options, score_gift_text


def test_naga_quarantined_and_aberration_ok():
    assert canonicalize("Naga") is None
    assert canonicalize("ABERRATION") == "Aberration"
    assert filter_lobby_tribes(["Naga", "Dragon", "Aberration", "Beast"]) == [
        "Dragon", "Aberration", "Beast",
    ]


def test_soft_lean_prefers_prior_but_pivots_on_board():
    lobby = ["Murloc", "Dragon", "Beast", "Mech", "Pirate"]
    lean, why = soft_lean_tribe([], available_tribes=lobby, turn=1)
    assert lean in lobby
    assert "soft" in why.lower() or "prior" in why.lower()
    # Board commitment pivots away from prior.
    board = [{"name": "a", "tribes": ["Dragon"]}, {"name": "b", "tribes": ["Dragon"]}]
    lean2, why2 = soft_lean_tribe(board, available_tribes=lobby, turn=4)
    assert lean2 == "Dragon"
    assert "pivot" in why2.lower() or "board" in why2.lower()


def test_manual_aberration_prior_normalizes():
    ranked = tribe_win_priors(
        available=["Murloc", "Dragon", "Beast", "Aberration"],
        manual={"Aberration": 0.55},  # Aidan-typed first%/weight at lobby start
    )
    names = [t for t, _ in ranked]
    assert "Aberration" in names
    assert "Naga" not in names
    assert abs(sum(s for _, s in ranked) - 1.0) < 1e-6
    # Strong manual prior should lead (or tie top) until board evidence pivots.
    assert ranked[0][0] == "Aberration", ranked
    # Ranking-mode input (1=best) also works.
    ranked2 = tribe_win_priors(
        available=["Murloc", "Dragon", "Aberration"],
        manual={"Aberration": 1, "Murloc": 2, "Dragon": 3},
    )
    assert ranked2[0][0] == "Aberration"


def test_tavern5_does_not_top_buy_t2_off_direction():
    board = [
        {"name": "d1", "tribes": ["Dragon"], "attack": 5, "health": 5},
        {"name": "d2", "tribes": ["Dragon"], "attack": 4, "health": 4},
    ]
    snap = {
        "board": board,
        "tavern_tier": 5,
        "available_tribes": ["Dragon", "Beast", "Mech", "Murloc", "Pirate"],
        "gold": 10,
        "shop": [],
    }
    junk = {"name": "T2 Pirate", "tribes": ["Pirate"], "tier": 2, "attack": 8, "health": 8}
    ondir = {"name": "T5 Dragon", "tribes": ["Dragon"], "tier": 5, "attack": 3, "health": 3}
    pj, rj = direction_buy_penalty(junk, snap)
    po, ro = direction_buy_penalty(ondir, snap)
    assert pj >= 1.0, (pj, rj)
    assert po < pj


def test_activate_allowlist_rejects_joyous_lullabot_scout():
    assert not is_activate_minion("BG36_110", "Joyous")
    assert not is_activate_minion("BG26_146", "Lullabot")
    assert not is_activate_minion("BG32_330", "Flighty Scout")
    # Allowlisted Activate still works.
    assert any(cid.startswith("BG36_345") for cid in ACTIVATE_CARD_IDS) or is_activate_minion("BG36_345")
    snap = {
        "phase": "recruit", "gold": 6, "tavern_tier": 2,
        "board": [{"name": "Joyous", "card_id": "BG36_110"}],
        "shop": [], "activatable": [
            {"name": "Joyous", "card_id": "BG36_110", "cost": 0, "usable": True},
        ],
    }
    assert ACTIVATE not in [a.kind for a in legal_actions(snap)]


def test_dark_gift_ranks_gift_over_body():
    opts = [
        GiftedOption("Big Body", attack=12, health=12,
                     gift_text="+2/+2", tribes=["Beast"]),
        GiftedOption("Tiny Golden", attack=1, health=1,
                     gift_text="This is Golden, but doesn't give a Triple Reward.",
                     tribes=["Beast"]),
    ]
    board = [{"name": "b", "tribes": ["Beast"]}, {"name": "b2", "tribes": ["Beast"]}]
    ranked = rank_dark_gift_options(
        opts, board=board, available_tribes=["Beast", "Dragon", "Murloc"])
    assert ranked[0][0].name == "Tiny Golden"
    gpow, _ = score_gift_text("Divine Shield, Windfury")
    assert gpow >= 1.0
