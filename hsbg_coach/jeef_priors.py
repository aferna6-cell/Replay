"""Soft recommend priors distilled from Jeef (and sparse RDU) VOD labels.

Advisory only — never hard rules. Labels under
`/workspace/hsreplay-tier7/jeef_vod_pilot/labels` are thin: most `decision`
rows lack gold/tier/board, and there is no dedicated ActionKind for pressing
the Dark Gift button (gift moments appear as ASR notes, discover picks with
`dark_gift` midgame-effect ids, or `comp_plan` strings). Treat these numbers
as gentle nudges so good Jeef-like spots can win #1 when the action is already
legal, without burying alternatives when the sample does not apply.

Sources (JeefHS unless noted):
- Dark gift over early level: ASR "Should have just dark gifted. Level is kind
  of dumb." (z9PfvDgMqAU)
- Gift discover midgame: turn≈4, tier≈2, gold≈3, pick Treasure Parrot +
  Replication (V0ZyTYC9gr8)
- Combo plan: `dark_gift_chef_hp` (g8kCF5Z-UIs)
- Activate / HP: ASR treats 0-cost Activate and clickable HP as real spends
  (z9PfvDgMqAU, Ry-Zn2sPP2k cues)
- Spells: Corrupted Coin (BG36_303) appears repeatedly in Aberration VODs;
  `spell_power` appears once as a labeled plan. BUY_SPELL cardId frequency is
  otherwise nearly empty — unknown spells stay demoted via spell_roles.
"""

from __future__ import annotations

from typing import Optional, Tuple

# Placement adjustments: negative = better expected finish (same axis as game_value).

# Jeef-named / VOD-seen tavern spells (cardId -> placement bonus).
# Keep the table small; do not invent out-of-pool Naga spells.
JEEF_SPELL_BONUS = {
    "BG36_303": (-0.35, "Jeef/Aberration VOD — Corrupted Coin often played"),
}

# Name substrings for when cardId is missing from the shop parse.
JEEF_SPELL_NAME_HINTS = (
    ("corrupted coin", -0.30, "Jeef VOD — Corrupted Coin"),
    ("strike oil", -0.15, "Jeef VOD OCR — Strike Oil seen in hand/shop"),
    ("warband whistle", -0.12, "Jeef VOD — Warband Whistle tooltip"),
)


def _get(snap, key, default=None):
    return snap.get(key, default) if isinstance(snap, dict) else getattr(snap, key, default)


def dark_gift_adjust(snapshot, cost: int = 3) -> Tuple[float, Optional[str]]:
    """Soft placement nudge when DARK_GIFT is legal.

    Boost: midgame (turn 3–7), tier 1–3, enough gold, shop not stacked with
    obvious upgrades, and/or HP also usable (gift+HP plan).
    Demote: panic spend (very low HP + need board bodies) or late game with a
    full strong board where 3g is better on activates/buys.
    """
    turn = _get(snapshot, "turn")
    tier = _get(snapshot, "tavern_tier") or 1
    gold = int(_get(snapshot, "gold") or 0)
    hp = _get(snapshot, "hero_health")
    board = list(_get(snapshot, "board", []) or [])
    shop = list(_get(snapshot, "shop", []) or [])
    hero_power = _get(snapshot, "hero_power") or {}

    adj = -0.25  # baseline: gift>body when legal (Jeef regret + denser 36.6.1 ASR)
    reason = "Dark Gift — Jeef prior: take when gold allows"

    # Midgame discover window (Replication pick was turn 4 / tier 2 / 3g).
    if turn is not None and 3 <= int(turn) <= 7 and tier <= 3:
        adj -= 0.20
        reason = "Dark Gift — Jeef midgame window (≈T3–7 / low tier)"

    # Prefer gift over dumping gold into a weak early level (ASR regret).
    level_cost = _get(snapshot, "level_cost")
    if level_cost is not None and gold >= cost and gold < int(level_cost) + cost:
        # Can gift but leveling would empty the turn — nudge gift.
        adj -= 0.10
        reason = "Dark Gift — Jeef: gift over a thin early level"

    # Combo: gift + clickable hero power in the same shop (labeled plan).
    if isinstance(hero_power, dict) and hero_power.get("usable"):
        hp_cost = int(hero_power.get("cost") or 0)
        if gold >= cost + hp_cost:
            adj -= 0.12
            reason = "Dark Gift — Jeef plan: gift + hero power same turn"

    # Weak shop → gift discover is better than rolling filler.
    if shop and _shop_looks_weak(shop, board):
        adj -= 0.08

    # Panic / late demotions (soft).
    if hp is not None and int(hp) <= 12 and len(board) < 4 and gold < cost + 3:
        adj += 0.25
        reason = "Dark Gift — demoted: low HP, need board/tempo first"
    if turn is not None and int(turn) >= 12 and len(board) >= 6:
        adj += 0.15
        reason = "Dark Gift — demoted: late full board, spend elsewhere"

    # Unaffordable should not happen (legal_actions gates), but be safe.
    if gold < cost:
        return 0.0, None

    return adj, reason


def activate_adjust(snapshot, cost: int = 0) -> Tuple[float, Optional[str]]:
    """Activate-keyword clicks — Jeef ASR treats 0-cost Activate as free value."""
    gold = int(_get(snapshot, "gold") or 0)
    if gold < cost:
        return 0.0, None
    if cost <= 0:
        return -0.40, "Activate (0g) — Jeef: press free Activate"
    if cost <= 2 and gold >= cost + 3:
        return -0.28, "Activate — cheap ability with gold left over"
    if gold >= cost + 3:
        return -0.22, "Activate — affordable ability"
    return -0.12, "Activate — usable but tight on gold"


def hero_power_adjust(snapshot, cost: int = 0) -> Tuple[float, Optional[str]]:
    """Clickable hero power. Sparse labels (2 HP decisions); ASR still values it."""
    gold = int(_get(snapshot, "gold") or 0)
    turn = _get(snapshot, "turn")
    dg = _get(snapshot, "dark_gift") or {}
    if gold < cost:
        return 0.0, None
    adj = -0.35
    reason = "Hero power ready — Jeef: click when usable"
    # Same-turn gift+HP plan.
    if isinstance(dg, dict) and dg.get("usable"):
        dg_cost = int(dg.get("cost") or 3)
        if gold >= cost + dg_cost:
            adj -= 0.10
            reason = "Hero power — Jeef: stack with Dark Gift"
    if cost == 0:
        adj -= 0.08
        reason = "Hero power (0g) — free click"
    if turn is not None and int(turn) <= 2 and cost >= 3:
        adj += 0.10  # early expensive HP can wait for board fill
    return adj, reason


def spell_prior_adjust(card_id: Optional[str], name: Optional[str]
                       ) -> Tuple[float, Optional[str]]:
    """Extra placement nudge for Jeef-frequent shop spells (on top of spell_roles)."""
    if card_id and card_id in JEEF_SPELL_BONUS:
        return JEEF_SPELL_BONUS[card_id]
    blob = (name or "").lower()
    for needle, bonus, note in JEEF_SPELL_NAME_HINTS:
        if needle in blob:
            return bonus, note
    return 0.0, None


def _shop_looks_weak(shop, board) -> bool:
    """True when shop avg stats are clearly below board (gift > buy filler)."""
    def stats(m):
        if isinstance(m, dict):
            return int(m.get("attack") or 0) + int(m.get("health") or 0)
        return int(getattr(m, "attack", 0) or 0) + int(getattr(m, "health", 0) or 0)
    if not shop:
        return True
    shop_avg = sum(stats(m) for m in shop) / max(1, len(shop))
    if not board:
        return shop_avg <= 6
    board_avg = sum(stats(m) for m in board) / max(1, len(board))
    return shop_avg + 2 < board_avg


# Human-readable sample size notes for PR / docs.
SAMPLE_NOTES = {
    "dark_gift_decisions": 3,
    "hero_power_decisions": 2,
    "activate_asr_hits": "sparse (cues + 1 HP/activate ASR)",
    "buy_spell_labeled_card_ids": "≈0 explicit BUY_SPELL decisions; Corrupted Coin from frame/ASR",
    "label_gap": "VOD decisions lack ActionKind for dark_gift button; gold/tier often null",
}


# ---------------------------------------------------------------------------
# Soft board-fill prior (Aidan playtest): fill solid bodies before hard rolling.
# Advisory only — never absolute. Direction-only good buys still count as fills.
# ---------------------------------------------------------------------------

_MAX_BOARD_SLOTS = 7
# Sparse thresholds: early/mid prefer bodies; late high-roll tempo is exempt.
_SPARSE_HARD = 4          # always treat <4 as sparse when gold remains
_SPARSE_SOFT = 5          # <5 still sparse early/mid
_LATE_HIGHROLL_TURN = 10
_LATE_HIGHROLL_TIER = 5


def _stats(m) -> int:
    if isinstance(m, dict):
        return int(m.get("attack") or 0) + int(m.get("health") or 0)
    return int(getattr(m, "attack", 0) or 0) + int(getattr(m, "health", 0) or 0)


def _board_is_sparse(snapshot) -> bool:
    board = list(_get(snapshot, "board", []) or [])
    n = len(board)
    if n >= _MAX_BOARD_SLOTS:
        return False
    if n < _SPARSE_HARD:
        return True
    turn = _get(snapshot, "turn")
    tier = int(_get(snapshot, "tavern_tier") or 1)
    # Soft sparse: not full and still early/mid curve.
    if n < _SPARSE_SOFT:
        if turn is not None and int(turn) >= _LATE_HIGHROLL_TURN and tier >= _LATE_HIGHROLL_TIER:
            return False
        return True
    return False


def _late_highroll_ok(snapshot) -> bool:
    """Intentional late high-roll tempo — don't demote rolling for fillers."""
    turn = _get(snapshot, "turn")
    tier = int(_get(snapshot, "tavern_tier") or 1)
    board = list(_get(snapshot, "board", []) or [])
    if len(board) >= 6 and tier >= _LATE_HIGHROLL_TIER:
        return True
    if turn is not None and int(turn) >= _LATE_HIGHROLL_TURN and tier >= _LATE_HIGHROLL_TIER:
        return True
    return False


def shop_unit_is_acceptable_fill(minion, snapshot, kb=None) -> bool:
    """True when a shop minion is a reasonable stabilize/fill buy (soft).

    On-direction / flex / low direction-penalty units count even with mediocre
    stats. Hard off-direction junk and far-below-tavern chaff do not.
    """
    if minion is None:
        return False
    try:
        from .tribe_policy import direction_buy_penalty, card_tier
        snap = snapshot if isinstance(snapshot, dict) else (
            snapshot.to_dict() if hasattr(snapshot, "to_dict") else snapshot)
        dpen, _ = direction_buy_penalty(minion, snap, kb=kb)
        mt = card_tier(minion, kb)
    except Exception:
        dpen = 0.0
        mt = minion.get("tier") if isinstance(minion, dict) else None
    # Hard demotion from direction/tier → not an acceptable fill.
    if dpen >= 0.8:
        return False
    tavern = int(_get(snapshot, "tavern_tier") or 1)
    if mt is not None and tavern >= 4 and int(tavern) - int(mt) >= 3:
        return False
    # On-direction / soft prior: accept as board filler.
    if dpen < 0.5:
        return True
    # Mild off-direction: need solid-ish stats vs board (or empty board body).
    board = list(_get(snapshot, "board", []) or [])
    s = _stats(minion)
    if not board:
        return s >= 4
    avg = sum(_stats(m) for m in board) / max(1, len(board))
    return avg <= 0 or (s / avg) >= 0.45


def shop_has_acceptable_fill(snapshot, kb=None) -> bool:
    shop = list(_get(snapshot, "shop", []) or [])
    return any(shop_unit_is_acceptable_fill(m, snapshot, kb) for m in shop)


def board_fill_roll_adjust(snapshot, kb=None) -> Tuple[float, Optional[str]]:
    """Soft placement nudge for ROLL. Positive = demote (prefer buy/stabilize).

    Demote when board is sparse, gold can buy, and the shop has any reasonable
    filler/on-direction unit — unless late high-roll tempo or shop is trash.
    """
    gold = int(_get(snapshot, "gold") or 0)
    if gold < 3:  # can't buy a minion (cost 3)
        return 0.0, None
    if not _board_is_sparse(snapshot):
        return 0.0, None
    if _late_highroll_ok(snapshot):
        return 0.0, None
    if not shop_has_acceptable_fill(snapshot, kb):
        return 0.0, None  # all trash → roll OK
    board = list(_get(snapshot, "board", []) or [])
    n = len(board)
    # Soft strength scales with how empty the board is.
    if n < 3:
        adj = 0.55
    elif n < 4:
        adj = 0.40
    else:
        adj = 0.28
    return adj, "board sparse — buy/stabilize before hard rolling"


def board_fill_buy_adjust(action, snapshot, kb=None) -> Tuple[float, Optional[str]]:
    """Soft placement nudge for BUY when stabilizing a thin board. Negative=better.

    Only boosts acceptable fills (incl. direction-only mediocre bodies). Does not
    rescue hard off-direction junk — that stays demoted by tribe_policy.
    """
    from .actions import BUY
    if getattr(action, "kind", None) != BUY:
        return 0.0, None
    if not _board_is_sparse(snapshot):
        return 0.0, None
    if _late_highroll_ok(snapshot):
        return 0.0, None
    detail = getattr(action, "detail", None) or {}
    minion = detail.get("minion")
    if not shop_unit_is_acceptable_fill(minion, snapshot, kb):
        return 0.0, None
    n = len(list(_get(snapshot, "board", []) or []))
    if n < 3:
        adj = -0.35
    elif n < 4:
        adj = -0.28
    else:
        adj = -0.18
    return adj, "fill board — stabilize before hard rolling"


# ---------------------------------------------------------------------------
# Soft mid-game solid-vs-trash prior (Aidan playtest): once the board is
# reasonably filled, stop endless re-rolling past solid on-direction pieces.
# Hard-roll only when hunting known keys with a full/stable board + dead shop.
# ---------------------------------------------------------------------------

_STABLE_BOARD = 5          # "reasonably filled" for mid-game stabilize
_SOLID_TIER_GAP = 1        # shop tier within 1 of tavern counts as on-curve


def _direction_for(snapshot, kb=None):
    try:
        from .tribe_policy import soft_lean_tribe, infer_direction
        board = list(_get(snapshot, "board", []) or [])
        avail = _get(snapshot, "available_tribes")
        lean, _ = soft_lean_tribe(board, available_tribes=avail, kb=kb,
                                  turn=_get(snapshot, "turn"))
        if lean:
            return lean
        return infer_direction(board, avail, kb=kb)
    except Exception:
        return None


def _board_is_stable(snapshot) -> bool:
    """Enough bodies that we should buy solid / cut chaff, not endlessly roll."""
    n = len(list(_get(snapshot, "board", []) or []))
    return n >= _STABLE_BOARD


def shop_unit_is_solid_midgame(minion, snapshot, kb=None) -> bool:
    """True when a shop unit is a solid mid-game keep for our direction (soft).

    Uses lobby direction + on-curve tier + meta quality when available. Trash /
    hard off-direction / far-below-tavern chaff return False.
    """
    if minion is None:
        return False
    try:
        from .tribe_policy import direction_buy_penalty, card_tier, is_on_direction
        snap = snapshot if isinstance(snapshot, dict) else (
            snapshot.to_dict() if hasattr(snapshot, "to_dict") else snapshot)
        dpen, _ = direction_buy_penalty(minion, snap, kb=kb)
        if dpen >= 0.8:
            return False
        direction = _direction_for(snapshot, kb)
        on_dir = is_on_direction(minion, direction, kb) if direction else (dpen < 0.5)
        if not on_dir and dpen >= 0.5:
            return False
        tavern = int(_get(snapshot, "tavern_tier") or 1)
        mt = card_tier(minion, kb)
        if mt is not None and int(tavern) - int(mt) > _SOLID_TIER_GAP and tavern >= 4:
            return False
    except Exception:
        on_dir = True
        mt = minion.get("tier") if isinstance(minion, dict) else None
        tavern = int(_get(snapshot, "tavern_tier") or 1)
        if mt is not None and tavern >= 4 and int(tavern) - int(mt) > _SOLID_TIER_GAP:
            return False
    # Meta quality: known-strong is always solid; known-weak on-curve still OK
    # as a direction body (pool knowledge), but pure trash stats are not.
    try:
        from .card_quality import placement, _WEAK
        cid = minion.get("card_id") if isinstance(minion, dict) else None
        name = minion.get("name") if isinstance(minion, dict) else None
        ap = placement(cid, name)
        if ap is not None and ap >= _WEAK + 0.35:
            return False  # clearly below-average filler
        if ap is not None and ap <= 3.2:
            return True   # meta-solid
    except Exception:
        pass
    # On-curve body with acceptable stats vs board average.
    s = _stats(minion)
    board = list(_get(snapshot, "board", []) or [])
    if not board:
        return s >= 5
    avg = sum(_stats(m) for m in board) / max(1, len(board))
    return avg <= 0 or (s / avg) >= 0.55


def shop_has_solid_midgame(snapshot, kb=None) -> bool:
    shop = list(_get(snapshot, "shop", []) or [])
    return any(shop_unit_is_solid_midgame(m, snapshot, kb) for m in shop)


def anti_stuck_roll_adjust(snapshot, kb=None) -> Tuple[float, Optional[str]]:
    """Soft demote ROLL when board is filled-enough and shop has solid on-direction.

    Prevents endless re-rolling past real pieces. Still allows hard-roll when the
    shop is dead, or late high-roll hunting with a full/stable board.
    """
    gold = int(_get(snapshot, "gold") or 0)
    if gold < 3:
        return 0.0, None
    # Sparse board handled by board_fill_roll_adjust.
    if _board_is_sparse(snapshot):
        return 0.0, None
    if not _board_is_stable(snapshot):
        return 0.0, None
    if _late_highroll_ok(snapshot) and not shop_has_solid_midgame(snapshot, kb):
        # Intentional key-hunt with a dead shop — roll OK.
        return 0.0, None
    if not shop_has_solid_midgame(snapshot, kb):
        return 0.0, None  # shop is trash / off-plan — roll OK
    return 0.35, "shop has solid on-direction — buy/cut instead of endless rolling"


def midgame_solid_buy_adjust(action, snapshot, kb=None) -> Tuple[float, Optional[str]]:
    """Soft promote BUY of solid on-direction mid-game pieces. Negative=better."""
    from .actions import BUY
    if getattr(action, "kind", None) != BUY:
        return 0.0, None
    if _board_is_sparse(snapshot):
        return 0.0, None  # board_fill handles sparse
    if not _board_is_stable(snapshot) and len(list(_get(snapshot, "board", []) or [])) < 4:
        return 0.0, None
    detail = getattr(action, "detail", None) or {}
    minion = detail.get("minion")
    if not shop_unit_is_solid_midgame(minion, snapshot, kb):
        return 0.0, None
    return -0.30, "solid on-direction mid-game — take it over endless rolling"


def direction_cut_sell_adjust(action, snapshot, kb=None) -> Tuple[float, Optional[str]]:
    """Soft promote SELL of useless off-direction pieces once direction is set.

    Negative = better (encourage the cut). Does not fire on trinket enablers or
    flex keys. Soft only — never forces a sell.
    """
    from .actions import SELL
    if getattr(action, "kind", None) != SELL:
        return 0.0, None
    board = list(_get(snapshot, "board", []) or [])
    if len(board) < 4:
        return 0.0, None  # too early to hard-cut
    detail = getattr(action, "detail", None) or {}
    minion = detail.get("minion")
    if minion is None:
        return 0.0, None
    try:
        from .tribe_policy import (soft_lean_tribe, is_on_direction, is_flex_key,
                                   direction_buy_penalty)
        avail = _get(snapshot, "available_tribes")
        direction, why = soft_lean_tribe(board, available_tribes=avail, kb=kb,
                                         turn=_get(snapshot, "turn"))
        if not direction:
            return 0.0, None
        # Need commitment (2+ on-direction) before cutting.
        on_count = sum(1 for m in board if is_on_direction(m, direction, kb))
        if on_count < 2:
            return 0.0, None
        name = minion.get("name") if isinstance(minion, dict) else getattr(minion, "name", None)
        cid = minion.get("card_id") if isinstance(minion, dict) else getattr(minion, "card_id", None)
        if is_flex_key(name, cid, kb):
            return 0.0, None
        if is_on_direction(minion, direction, kb):
            return 0.0, None
        # Don't cut pieces that enable equipped trinkets.
        try:
            from .comp_signals import sell_trinket_penalty
            if sell_trinket_penalty(minion, snapshot, kb) > 0.2:
                return 0.0, None
        except Exception:
            pass
        # Off-direction dead weight — soft promote the cut.
        return -0.40, f"cut off-{direction} — make room for your plan"
    except Exception:
        return 0.0, None
