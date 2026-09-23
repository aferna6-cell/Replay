"""What a hero's HSReplay guide says about its hero power.

Two readings of the guide text, used by hero select and by NEXT:

  * a *weak* verdict — the guide says the hero power itself is bad
    ("Don't expect much value from this hero power", "The hero power isn't
    that good", "Patchwerk 2.0", "isn't strong enough to play into"). Hero
    select demotes these heroes; NEXT never promotes their hero power.
  * a *restriction* — a line telling you when NOT to press it ("Do not hero
    power on tavern 2", "only when you have 1 gold left over"). Those are not
    reasons to make Hero Power the NEXT move.
"""

from __future__ import annotations

import re
from typing import List, Optional

_SENT = re.compile(r"(?<=[.!?])\s+")
_MARK = re.compile(r"\[\[([^\]|]+)(?:\|\|\d+)?\]\]")
_WEAK = re.compile(
    r"don'?t expect much (?:value )?from (?:this|the|your) hero power"
    r"|hero power (?:is|isn'?t|is not)\s+(?:not\s+)?(?:that |very |really )?"
    r"(?:good|strong|great)(?:\s+enough)?\b(?<!is good)(?<!is strong)(?<!is great)"
    r"|hero power (?:is )?(?:weak|bad|underwhelming|useless)"
    r"|hero power scales (?:a bit )?too slow"
    r"|patchwerk 2\.0",
    re.I)
_RESTRICT = re.compile(
    r"\b(?:do not|don'?t|never|stop|only)\b[^.]*\bhero ?power"
    r"|\bhero ?power\b[^.]*\b(?:only|unless|until|don'?t|do not|not)\b",
    re.I)


def _plain(text: str) -> str:
    return _MARK.sub(lambda m: m.group(1), text or "")


def _sentences(hero: Optional[dict]) -> List[str]:
    if not hero:
        return []
    text = " ".join(x for x in (hero.get("guide_text"),
                                *((hero.get("structured") or {}).get("hp") or [])) if x)
    return [s.strip() for s in _SENT.split(_plain(text)) if s.strip()]


def weak_quote(hero: Optional[dict]) -> Optional[str]:
    """The guide sentence that calls the hero power weak, else None."""
    for s in _sentences(hero):
        if _WEAK.search(s):
            return s
    return None


def is_restriction(line: str) -> bool:
    """A 'when not to press it' line rather than a reason to press it now."""
    return bool(_RESTRICT.search(_plain(line)))


def usable_hp_bullets(hero: Optional[dict]) -> List[str]:
    """HP bullets that actually recommend pressing it (no weak verdict)."""
    if not hero or weak_quote(hero):
        return []
    return [b for b in (hero.get("structured") or {}).get("hp") or []
            if not is_restriction(b) and not _WEAK.search(_plain(b))]


_WHEN = re.compile(r"\b(tavern|turn)\s+(\d+)\b", re.I)


def restricted_now(hero: Optional[dict], tavern_tier=None, turn=None) -> Optional[str]:
    """A 'do not hero power' line that names this tavern tier / turn."""
    if not hero:
        return None
    for b in (hero.get("structured") or {}).get("hp") or []:
        if not is_restriction(b):
            continue
        for kind, n in _WHEN.findall(_plain(b)):
            cur = tavern_tier if kind.lower() == "tavern" else turn
            if cur is not None and str(cur) == n:
                return _plain(b)
    return None
