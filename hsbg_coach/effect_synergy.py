"""Effect-text synergy: infer which cards combo from *what they do*, not from
co-occurrence data — so it generalizes to cards card2vec has never seen.

Model each card as two concept sets read from its rules text + keywords:

  * PRODUCES — what it puts into the system: summons of a tribe, granted keywords
    (Divine Shield/Taunt/…), Blood Gems, tavern spells, gold, stat buffs to a
    tribe, and its own tribe membership (a body payoffs can care about).
  * WANTS — what it pays off: "whenever you summon", "after you cast a spell",
    "your Murlocs", Blood-Gem / Divine-Shield / Battlecry / Deathrattle payoffs,
    menagerie (different tribes).

Synergy(candidate, board) = how much the candidate PRODUCES what the board WANTS,
plus how much the candidate WANTS what the board PRODUCES. Directional and
symmetric, so a token-maker scores high next to a "whenever you summon" payoff —
and vice-versa — even if neither is in the embedding vocabulary.

Heuristic, text-derived (not learned), and complementary to card2vec: card2vec
captures co-occurrence the text misses; this captures mechanics the data is too
thin to have seen yet.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Set, Tuple

from .cards import CardKnowledge

_TRIBES = ["murloc", "beast", "dragon", "mech", "elemental", "undead",
           "demon", "pirate", "quilboar", "naga"]
_GRANTABLE = ["divine shield", "taunt", "windfury", "poisonous", "venomous",
              "reborn", "stealth", "cleave"]

# WANT triggers: text that means "this card pays off when X happens / exists".
_CARE_PREFIXES = ("after you", "whenever you", "whenever a friendly", "whenever another",
                  "after a friendly", "at the end of your turn", "your other")
_PRODUCE_VERBS = ("summon", "give", "gain", "grant", "add", "generate", "get a")


@dataclass
class Profile:
    produces: Set[str] = field(default_factory=set)
    wants: Set[str] = field(default_factory=set)


def _t(card: CardKnowledge) -> str:
    return (getattr(card, "text", "") or "").lower()


def card_profile(card: CardKnowledge) -> Profile:
    """Read PRODUCES / WANTS concept tags from a card's text, keywords, tribes."""
    text = _t(card)
    p: Set[str] = set()
    w: Set[str] = set()

    # Its own identity is a "product" payoffs can consume.
    for tr in (t.lower() for t in getattr(card, "tribes", []) or []):
        if tr in _TRIBES:
            p.add(f"tribe:{tr}")
    for kw in (k.lower().replace("_", " ") for k in getattr(card, "keywords", []) or []):
        if kw in _GRANTABLE:
            p.add(f"keyword:{kw}")
        if kw in ("battlecry", "deathrattle"):
            p.add(f"trigger:{kw}")
    if not text:
        return Profile(p, w)

    has_produce = any(v in text for v in _PRODUCE_VERBS)
    has_care = any(pre in text for pre in _CARE_PREFIXES)

    # Tribe production / payoff.
    for tr in _TRIBES:
        if tr in text:
            if has_produce and ("summon" in text or "+" in text or "give" in text):
                p.add(f"tribe:{tr}")            # makes/buffs that tribe
            if has_care or "your " + tr in text:
                w.add(f"tribe:{tr}")            # cares about that tribe

    # Granted keywords (Divine Shield giver, etc.) and their payoffs.
    for kw in _GRANTABLE:
        if kw in text:
            if has_produce:
                p.add(f"keyword:{kw}")
            if has_care:
                w.add(f"keyword:{kw}")

    # Mechanic tokens / resources.
    if "blood gem" in text:
        (p if "gain" in text or "give" in text or "summon" in text else w).add("bloodgem")
        if has_care:
            w.add("bloodgem")
    if "spell" in text:
        if "add" in text or "get a" in text or "discover" in text:
            p.add("spell")
        if has_care or "cast" in text or "played a spell" in text:
            w.add("spell")
    if re.search(r"\bgain\b.*\bgold\b|\bcoin\b", text):
        p.add("gold")

    # Summon synergy (token producers <-> "whenever you summon").
    if "summon" in text:
        p.add("summon")
    if "whenever you summon" in text or "after you summon" in text:
        w.add("summon")

    # Trigger payoffs / doublers.
    for kw in ("battlecry", "deathrattle"):
        if kw in text and (has_care or "twice" in text):
            w.add(f"trigger:{kw}")

    # Menagerie payoff (cares about many different tribes).
    if "different" in text and ("tribe" in text or "minion type" in text):
        w.add("menagerie")
    return Profile(p, w)


def _menagerie_count(board_tribes: Set[str]) -> int:
    return len(board_tribes & set(_TRIBES))


def board_synergy(candidate: CardKnowledge, board: Sequence[CardKnowledge]
                  ) -> Tuple[float, List[str]]:
    """(score, reasons): how the candidate's effects mesh with the board's.

    score is unbounded-positive synergy points; callers scale it. Reasons name the
    concrete combo ("makes Beasts your board pays off")."""
    cand = card_profile(candidate)
    if not board:
        return 0.0, []
    bp: Set[str] = set()
    bw: Set[str] = set()
    btribes: Set[str] = set()
    for c in board:
        prof = card_profile(c)
        bp |= prof.produces
        bw |= prof.wants
        btribes |= {t.lower() for t in getattr(c, "tribes", []) or []}

    score = 0.0
    reasons: List[str] = []

    # A tribe-specific payoff (e.g. Banana Slamma cares about Beast summons) is dead
    # without that tribe on the board — don't credit a Beast payoff with no Beasts.
    cand_tribes = {t.split(":", 1)[1] for t in cand.wants if t.startswith("tribe:")}
    _GENERIC = {"summon", "bloodgem"}        # mechanic concepts a tribe gates

    def _tribe_ok(tag: str) -> bool:
        if tag in _GENERIC and cand_tribes:
            return bool(cand_tribes & btribes)
        return True

    # Candidate PRODUCES what the board WANTS.
    for tag in cand.produces & bw:
        score += 1.1
        reasons.append(f"feeds your board's {_pretty(tag)} payoff")
    # Candidate WANTS what the board PRODUCES.
    for tag in cand.wants & bp:
        if not _tribe_ok(tag):               # tribe-specific payoff, tribe absent
            continue
        score += 1.1
        reasons.append(f"pays off your board's {_pretty(tag)}")

    # Menagerie: the candidate cares about tribe variety and your board is varied.
    if "menagerie" in cand.wants and _menagerie_count(btribes) >= 3:
        score += 1.0
        reasons.append("menagerie payoff (your board spans many tribes)")

    # De-dup reasons, keep top few.
    seen, uniq = set(), []
    for r in reasons:
        if r not in seen:
            seen.add(r)
            uniq.append(r)
    return round(score, 2), uniq[:3]


def _pretty(tag: str) -> str:
    if ":" in tag:
        kind, val = tag.split(":", 1)
        if kind == "tribe":
            return val.capitalize()
        return val
    return tag
