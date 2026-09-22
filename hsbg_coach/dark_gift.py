"""Dark Gift discover ranking — the gift matters more than the body.

When Discover offers minions that each carry a Dark Gift enchantment, rank by:
  1. Gift power / synergy with the board (dominant)
  2. Soft on-direction / tribe fit
  3. Base minion stats / tier (weakest signal)

Aidan: "the gift is more important than the minion itself."
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .tribe_policy import (
    QUARANTINED_TRIBES, canonicalize, infer_direction, is_on_direction, _norm,
)

# Heuristic gift tiers from MidGameEffect text. Higher = better gift.
_GIFT_POWER_PATTERNS: List[Tuple[re.Pattern, float, str]] = [
    (re.compile(r"this is golden", re.I), 1.40, "Golden gift"),
    (re.compile(r"has all minion types|amalgam", re.I), 1.25, "Amalgam types"),
    (re.compile(r"reborn.*full stats|persisting horror", re.I), 1.20, "full Reborn"),
    (re.compile(r"divine shield.*windfury|windfury.*divine shield", re.I), 1.15, "DS+WF"),
    (re.compile(r"divine shield.*3 hits|toreth", re.I), 1.10, "triple DS"),
    (re.compile(r"start of combat:.*deathrattle|jaws of death", re.I), 1.05, "SoC Deathrattle"),
    (re.compile(r"end of your turn.*battlecry|echoing voice", re.I), 1.00, "EoT Battlecry"),
    (re.compile(r"also trigger at start of turn|time turning", re.I), 0.95, "double EoT"),
    (re.compile(r"get an? extra copy|double vision", re.I), 0.90, "extra copy"),
    (re.compile(r"for each (battlecry|deathrattle|tavern spell)", re.I), 0.95, "scaling gift"),
    (re.compile(r"rally:.*get a random minion|charisma", re.I), 0.85, "Rally generate"),
    (re.compile(r"at the end of every 2 turns.*copy|replication", re.I), 0.80, "Replication"),
    (re.compile(r"\+10 (attack|health)|sacrifice", re.I), 0.55, "flat +10 gift"),
    (re.compile(r"divine shield", re.I), 0.70, "Divine Shield gift"),
    (re.compile(r"windfury|reborn|taunt|venomous|poisonous", re.I), 0.60, "keyword gift"),
    (re.compile(r"\+\d+/\+\d+", re.I), 0.40, "stat gift"),
]


@dataclass
class GiftedOption:
    name: str
    card_id: Optional[str] = None
    gift_id: Optional[str] = None
    gift_name: Optional[str] = None
    gift_text: Optional[str] = None
    attack: Optional[int] = None
    health: Optional[int] = None
    tier: Optional[int] = None
    tribes: Optional[List[str]] = None


def score_gift_text(text: Optional[str], gift_name: Optional[str] = None) -> Tuple[float, str]:
    """Return (power, label) for a Dark Gift enchantment. Unknown gifts get a mild prior."""
    blob = f"{gift_name or ''} {text or ''}".strip()
    if not blob:
        return 0.35, "unknown gift"
    best = 0.35
    label = "generic gift"
    for pat, val, lab in _GIFT_POWER_PATTERNS:
        if pat.search(blob):
            if val > best:
                best, label = val, lab
    return best, label


def _body_score(opt: GiftedOption, kb=None) -> float:
    atk = opt.attack
    hp = opt.health
    if (atk is None or hp is None) and kb and opt.card_id and opt.card_id in kb:
        ck = kb[opt.card_id]
        atk = atk if atk is not None else ck.attack
        hp = hp if hp is not None else ck.health
    atk = int(atk or 0)
    hp = int(hp or 0)
    # Weak signal — gift dominates. Cap so a 20/20 body can't beat a great gift.
    return min(0.45, (atk + hp) / 40.0)


def rank_dark_gift_options(options: Sequence[GiftedOption], board=None, kb=None,
                           available_tribes=None, direction: Optional[str] = None
                           ) -> List[Tuple[GiftedOption, float, str]]:
    """Best-first ranking. Sort key lower = better (matches draft.Choice convention).

    Weighting: gift ~70%, direction fit ~20%, body ~10%.
    """
    direction = direction or infer_direction(board, available_tribes, kb=kb)
    ranked = []
    for opt in options:
        # Quarantine Naga bodies.
        tribes = opt.tribes or []
        if any(_norm(t) in QUARANTINED_TRIBES for t in tribes):
            ranked.append((opt, 9.0, "Naga out of pool — skip"))
            continue
        gpow, glabel = score_gift_text(opt.gift_text, opt.gift_name)
        body = _body_score(opt, kb)
        on_dir = False
        if direction:
            proxy = {"name": opt.name, "card_id": opt.card_id, "tribes": tribes}
            on_dir = is_on_direction(proxy, direction, kb)
        dir_term = -0.20 if on_dir else 0.10
        # Lower is better: invert gift/body so bigger gift → smaller rank_value.
        rank_value = -(0.70 * gpow + 0.10 * body) + dir_term
        bits = [f"gift: {glabel} ({gpow:.2f})"]
        if on_dir and direction:
            bits.append(f"on-{direction}")
        elif direction:
            bits.append(f"off-{direction}")
        bits.append(f"body {body:.2f}")
        ranked.append((opt, rank_value, " — ".join(bits)))
    ranked.sort(key=lambda x: x[1])
    return ranked


def enrich_discover_for_gifts(offered_names: List[str], gift_by_name: Optional[Dict[str, dict]] = None,
                              kb=None) -> List[GiftedOption]:
    """Build GiftedOptions from discover names + optional gift metadata map."""
    gift_by_name = gift_by_name or {}
    out = []
    for nm in offered_names:
        meta = gift_by_name.get(nm) or gift_by_name.get(nm.lower()) or {}
        tribes = meta.get("tribes")
        cid = meta.get("card_id")
        if kb and not tribes:
            from .cards import by_name
            ck = by_name(kb).get(nm) if not isinstance(kb, dict) else None
            if ck is None and isinstance(kb, dict):
                # try name index
                from .cards import by_name as bn
                try:
                    ck = bn(kb).get(nm)
                except Exception:
                    ck = None
            if ck is not None:
                tribes = list(ck.tribes or [])
                cid = cid or ck.card_id
        out.append(GiftedOption(
            name=nm, card_id=cid, tribes=tribes,
            gift_id=meta.get("gift_id"), gift_name=meta.get("gift_name"),
            gift_text=meta.get("gift_text"),
            attack=meta.get("attack"), health=meta.get("health"),
            tier=meta.get("tier"),
        ))
    return out

enrich_discover_with_gifts = enrich_discover_for_gifts
