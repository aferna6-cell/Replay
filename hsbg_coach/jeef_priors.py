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
