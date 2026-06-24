"""Synergy layer — does a card work WITH my board, trinkets, and hero power?

Stats tell you a card is strong on average; synergy tells you it's strong *for
your board right now*. We derive synergy **tags** from each card's keywords and
rules text (e.g. a card that buffs Murlocs, one that cares about Battlecries, a
Battlecry-doubler), then score a candidate against the current context.

HONEST SCOPE: this is **text/keyword-derived heuristic** synergy — it captures
the obvious, high-signal interactions (tribe buffs, keyword payoffs, doublers,
hero-power/trinket tribe care). It is NOT learned statistical co-occurrence;
that's an ML step on real trajectories later (specs §6). It composes with the
population stats: stats rank cards globally, synergy re-ranks them for *your*
board.

Reads `CardKnowledge` from `cards.py`. Trinkets/hero powers are passed as text
(their effects also mention tribes/keywords).
"""

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set

from .cards import CardKnowledge

TRIBES = ["Murloc", "Beast", "Dragon", "Mech", "Elemental", "Undead",
          "Demon", "Pirate", "Quilboar", "Naga"]
_KEYWORDS = ["battlecry", "deathrattle", "divine shield", "taunt", "poisonous",
             "reborn", "windfury", "magnetic", "frenzy", "avenge"]

_BUFF_WORDS = ("give", "gain", "+", "buff", "grant")
_CARES_WORDS = ("after you play", "whenever you play", "whenever you summon",
                "after you summon", "whenever a friendly")


def _tribe_mentions(text: str) -> Set[str]:
    low = text.lower()
    return {t for t in TRIBES if t.lower() in low}


def derive_tags(card: CardKnowledge) -> Set[str]:
    """Synergy tags for a card: is:<kw>, buffs:<Tribe>/all, cares:<Tribe>/<kw>,
    doubles:<kw>."""
    tags: Set[str] = set()
    for kw in card.keywords:
        tags.add(f"is:{kw.lower().replace('_', ' ')}")
    text = card.text.lower()
    if not text:
        return tags

    has_buff = any(w in text for w in _BUFF_WORDS)
    has_cares = any(w in text for w in _CARES_WORDS)
    if ("your other" in text or "friendly minions" in text) and has_buff:
        tags.add("buffs:all")
    for tribe in _tribe_mentions(card.text):
        if has_buff:
            tags.add(f"buffs:{tribe}")
        if has_cares:
            tags.add(f"cares:{tribe}")
    for kw in _KEYWORDS:
        if kw in text and has_cares:
            tags.add(f"cares:{kw}")
    # Doublers ("trigger twice"). Match stems so plurals count
    # ("Battlecries"/"Deathrattles").
    if "twice" in text:
        for kw, stem in (("battlecry", "battlecr"), ("deathrattle", "deathrattle")):
            if stem in text:
                tags.add(f"doubles:{kw}")
    return tags


@dataclass
class SynergyVerdict:
    score: float
    reasons: List[str] = field(default_factory=list)


def _board_tribes(board: Sequence[CardKnowledge]) -> Set[str]:
    out: Set[str] = set()
    for c in board:
        out.update(c.tribes)
    return out


def score_card(
    candidate: CardKnowledge,
    board: Sequence[CardKnowledge],
    target_tribe: Optional[str] = None,
    hero_power_text: str = "",
    trinket_texts: Iterable[str] = (),
) -> SynergyVerdict:
    """Score how well `candidate` synergizes with the current context."""
    score = 0.0
    reasons: List[str] = []
    cand_tribes = set(candidate.tribes)
    board_tribes = _board_tribes(board)
    board_tags: Set[str] = set()
    for c in board:
        board_tags |= derive_tags(c)
    cand_tags = derive_tags(candidate)

    # 1. Shares a tribe with the board you're building.
    shared = cand_tribes & board_tribes
    if shared:
        score += 1.0
        reasons.append(f"on-tribe with board ({', '.join(sorted(shared))})")
    if target_tribe and target_tribe in cand_tribes:
        score += 0.8
        reasons.append(f"fits target comp ({target_tribe})")

    # 2. Candidate buffs a tribe you already have (or all).
    for tribe in board_tribes:
        if f"buffs:{tribe}" in cand_tags:
            score += 1.2
            reasons.append(f"buffs your {tribe}s")
    if "buffs:all" in cand_tags and board:
        score += 0.6
        reasons.append("buffs your whole board")

    # 3. A board minion buffs the candidate's tribe.
    for tribe in cand_tribes:
        if any(f"buffs:{tribe}" in derive_tags(c) for c in board):
            score += 1.0
            reasons.append(f"your board buffs {tribe}s")

    # 4. Keyword payoffs: doublers + cares-about.
    for kw in ("battlecry", "deathrattle"):
        if f"is:{kw}" in cand_tags and f"doubles:{kw}" in board_tags:
            score += 1.3
            reasons.append(f"{kw} doubled by your board")
    for tribe in cand_tribes:
        if f"cares:{tribe}" in board_tags:
            score += 0.7
            reasons.append(f"your board cares about {tribe}s")

    # 5. Hero power / trinkets that care about the candidate's tribe/keywords.
    extra = " ".join([hero_power_text, *trinket_texts]).lower()
    if extra:
        for tribe in cand_tribes:
            if tribe.lower() in extra:
                score += 0.5
                reasons.append(f"hero power/trinket cares about {tribe}s")
                break

    return SynergyVerdict(round(score, 2), reasons)


def rank_shop(
    shop: Sequence[CardKnowledge],
    board: Sequence[CardKnowledge],
    target_tribe: Optional[str] = None,
    hero_power_text: str = "",
    trinket_texts: Iterable[str] = (),
) -> List[tuple]:
    """Rank shop minions by synergy. Returns [(card, SynergyVerdict), …] best-first."""
    scored = [
        (c, score_card(c, board, target_tribe, hero_power_text, trinket_texts))
        for c in shop
    ]
    scored.sort(key=lambda x: x[1].score, reverse=True)
    return scored


def resolve(names_or_ids: Iterable[str], kb: Dict[str, CardKnowledge],
            kb_by_name: Optional[Dict[str, CardKnowledge]] = None
            ) -> List[CardKnowledge]:
    """Map snapshot minion names/card-ids to CardKnowledge (skips unknowns)."""
    by_name = kb_by_name or {c.name: c for c in kb.values()}
    out = []
    for key in names_or_ids:
        c = kb.get(key) or by_name.get(key)
        if c:
            out.append(c)
    return out
